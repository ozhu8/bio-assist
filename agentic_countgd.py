"""
Agentic counting pipeline: Claude orchestrates CountGD.

Claude turns a natural-language request into a short count target, CountGD
(via its hosted Gradio Space) does the counting, and Claude evaluates the
annotated result and retries with an adjusted prompt if it looks wrong.

Usage:
    python agentic_countgd.py --image cells.webp --prompt "count the individual cells"
"""
import argparse
import base64
import json
import mimetypes
import textwrap
from pathlib import Path
from typing import Any, Optional, cast

import anthropic # pyright: ignore[reportMissingImports]
from anthropic.types import ImageBlockParam # pyright: ignore[reportMissingImports]
import matplotlib.pyplot as plt # pyright: ignore[reportMissingModuleSource]
from matplotlib.backends.backend_pdf import PdfPages # pyright: ignore[reportMissingModuleSource]
from PIL import Image # pyright: ignore[reportMissingImports]
from gradio_client import Client, handle_file # pyright: ignore[reportMissingImports]

MODEL = "claude-opus-4-8"
COUNTGD_SPACE = "nikigoli/countgd"
PDF_NAME = "countgd_results.pdf"


ALLOWED_IMAGE_MIME_TYPES = ("image/jpeg", "image/png", "image/gif", "image/webp")


def image_to_content_block(image_path: str) -> ImageBlockParam:
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type not in ALLOWED_IMAGE_MIME_TYPES:
        mime_type = "image/png"
    data = base64.standard_b64encode(Path(image_path).read_bytes()).decode("utf-8")
    return cast(ImageBlockParam, {
        "type": "image",
        "source": {"type": "base64", "media_type": mime_type, "data": data},
    })


def unwrap(value: Any) -> Any:
    """Gradio event-driven outputs arrive as {'value': ..., '__type__': 'update'} dicts."""
    return value["value"] if isinstance(value, dict) and "value" in value else value


def run_countgd(countgd_client: Client, image_path: str, text: str):
    raw_image, raw_count = countgd_client.predict(
        image=handle_file(image_path),
        text=text,
        prompts={"image": handle_file(image_path), "points": []},
        api_name="/count_main",
    )
    return unwrap(raw_image), int(unwrap(raw_count))


def interpret_prompt(claude: anthropic.Anthropic, user_prompt: str, image_path: str) -> str:
    response = claude.messages.create(
        model=MODEL,
        max_tokens=2048,
        thinking={"type": "adaptive"},
        messages=[{
            "role": "user",
            "content": [
                image_to_content_block(image_path),
                {"type": "text", "text": (
                    f"The user wants to count objects in this image. Their request: "
                    f"\"{user_prompt}\"\n\n"
                    "Reply with ONLY a short noun phrase (1-3 words) naming the single "
                    "object type CountGD should count (e.g. 'cell', 'car', 'strawberry'). "
                    "No punctuation, no explanation, nothing else."
                )},
            ],
        }],
    )
    text = next(b.text for b in response.content if b.type == "text")
    return text.strip().strip('."\'')


def evaluate_result(
    claude: anthropic.Anthropic,
    user_prompt: str,
    count_target: str,
    predicted_count: int,
    annotated_image_path: str,
    history: list,
) -> dict:
    response = claude.messages.create(
        model=MODEL,
        max_tokens=4096,
        thinking={"type": "adaptive"},
        output_config={
            "format": {
                "type": "json_schema",
                "schema": {
                    "type": "object",
                    "properties": {
                        "accept": {"type": "boolean"},
                        "score": {"type": "integer"},
                        "feedback": {"type": "string"},
                        "revised_text": {"type": ["string", "null"]},
                    },
                    "required": ["accept", "score", "feedback", "revised_text"],
                    "additionalProperties": False,
                },
            }
        },
        messages=[{
            "role": "user",
            "content": [
                image_to_content_block(annotated_image_path),
                {"type": "text", "text": (
                    f"Original user request: \"{user_prompt}\"\n"
                    f"CountGD was asked to count: \"{count_target}\"\n"
                    f"CountGD's predicted count: {predicted_count}\n"
                    f"Prior attempts this session: {json.dumps(history)}\n\n"
                    "The attached image shows CountGD's detections as boxes/heatmap. "
                    "Evaluate: (1) do the boxes look visually accurate (no obvious "
                    "double-counts, missed objects, or false positives)? (2) is the count "
                    "physically/biologically plausible? (3) does this satisfy the user's "
                    "original request?\n"
                    "Score 0-10. If score < 7 and a different/more specific text prompt "
                    "would plausibly fix it, set accept=false and give revised_text to "
                    "retry with. Otherwise set accept=true and revised_text=null."
                )},
            ],
        }],
    )
    text = next(b.text for b in response.content if b.type == "text")
    return json.loads(text)


ACCEPT_SCORE_THRESHOLD = 7

SCORING_RUBRIC = (
    "Each iteration is scored 0-10 by evaluating the annotated image against three criteria:\n"
    "  1. Do the detection boxes/dots look visually accurate (no obvious double-counts,\n"
    "     missed objects, or false positives)?\n"
    "  2. Is the predicted count physically/biologically plausible for what's shown?\n"
    "  3. Does the result satisfy the user's original request?\n\n"
    f"A score >= {ACCEPT_SCORE_THRESHOLD} accepts the result. A score below that triggers a retry\n"
    "with a revised count target, if a more specific prompt would plausibly fix the issue. A\n"
    "result with zero detections on an image that clearly contains the target object fails all\n"
    "three criteria outright and scores 0, regardless of how the count target was worded.\n\n"
    "Loops to good result: the number of iterations run before a score reached the accept\n"
    "threshold (or the total number run, if none did) — a measure of how many retries the\n"
    "agentic loop needed, separate from the accuracy of the final result itself."
)


def loops_to_acceptance(history: list) -> tuple:
    """Return (iteration_count, reached) where iteration_count is the 1-indexed
    iteration that first met ACCEPT_SCORE_THRESHOLD, or len(history) if none did."""
    for entry in history:
        if entry["score"] >= ACCEPT_SCORE_THRESHOLD:
            return entry["iteration"], True
    return len(history), False


def save_pdf_report(
    pdf_path: Path,
    user_prompt: str,
    image_paths: list,
    history: list,
    evaluator_note: Optional[str] = None,
) -> None:
    """Render a methodology page, then one page per iteration (annotated image + feedback), into a single PDF."""
    with PdfPages(pdf_path) as pdf:
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis("off")
        ax.set_title("Scoring methodology", fontsize=15, fontweight="bold", loc="left")
        methodology = SCORING_RUBRIC
        if evaluator_note:
            methodology += f"\n\nNote on this run:\n{textwrap.fill(evaluator_note, 90)}"
        ax.text(0, 0.95, methodology, va="top", ha="left", fontsize=11, wrap=True)
        pdf.savefig(fig)
        plt.close(fig)

        for entry, image_path in zip(history, image_paths):
            fig, (ax_img, ax_text) = plt.subplots(
                2, 1, figsize=(8.5, 11), gridspec_kw={"height_ratios": [4, 1]}
            )
            ax_img.imshow(Image.open(image_path))
            ax_img.axis("off")
            ax_img.set_title(f"Iteration {entry['iteration']}: counting {entry['count_target']!r}")

            ax_text.axis("off")
            caption = (
                f"Request: {user_prompt}\n"
                f"Predicted count: {entry['predicted_count']}\n"
                f"Score: {entry['score']}/10\n"
                f"Feedback: {textwrap.fill(entry['feedback'], 100)}"
            )
            ax_text.text(0, 1, caption, va="top", ha="left", fontsize=10, wrap=True)

            pdf.savefig(fig)
            plt.close(fig)

        loops, reached = loops_to_acceptance(history)
        final = history[-1]
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis("off")
        ax.set_title("Summary", fontsize=15, fontweight="bold", loc="left")
        loops_line = (
            f"Loops to good result: {loops} of {len(history)} iterations run"
            if reached
            else f"Loops to good result: not reached (all {len(history)} iterations scored "
                 f"below {ACCEPT_SCORE_THRESHOLD}/10)"
        )
        summary = (
            f"Request: {user_prompt}\n\n"
            f"{loops_line}\n"
            f"Final count target: {final['count_target']!r}\n"
            f"Final predicted count: {final['predicted_count']}\n"
            f"Final score: {final['score']}/10"
        )
        ax.text(0, 0.95, summary, va="top", ha="left", fontsize=11, wrap=True)
        pdf.savefig(fig)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Run CountGD with Claude as orchestrator/evaluator")
    parser.add_argument("--image", required=True, help="Path to the input image")
    parser.add_argument("--prompt", required=True, help="What to count / user instruction")
    parser.add_argument("--max-iterations", type=int, default=5)
    parser.add_argument("--output-dir", default="./countgd_agent_output")
    parser.add_argument("--pdf-name", default=PDF_NAME, help="Filename for the saved PDF report")
    args = parser.parse_args()
    if args.max_iterations < 1:
        parser.error("--max-iterations must be at least 1")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    claude = anthropic.Anthropic()
    countgd = Client(COUNTGD_SPACE)

    count_target = interpret_prompt(claude, args.prompt, args.image)
    print(f"[Claude] counting target: {count_target!r}")

    history = []
    saved_paths = []
    saved_path = None
    predicted_count = None
    for i in range(1, args.max_iterations + 1):
        print(f"\n--- Iteration {i}: CountGD counting {count_target!r} ---")
        annotated_path, predicted_count = run_countgd(countgd, args.image, count_target)
        print(f"[CountGD] count={predicted_count}")

        saved_path = output_dir / f"iteration_{i}.png"
        saved_path.write_bytes(Path(annotated_path).read_bytes())
        saved_paths.append(saved_path)

        eval_result = evaluate_result(
            claude, args.prompt, count_target, predicted_count, str(saved_path), history
        )
        print(f"[Claude eval] score={eval_result['score']} accept={eval_result['accept']}")
        print(f"[Claude eval] feedback: {eval_result['feedback']}")

        history.append({
            "iteration": i,
            "count_target": count_target,
            "predicted_count": predicted_count,
            "score": eval_result["score"],
            "feedback": eval_result["feedback"],
        })

        if eval_result["accept"] or not eval_result.get("revised_text"):
            break
        count_target = eval_result["revised_text"]

    pdf_path = output_dir / args.pdf_name
    save_pdf_report(pdf_path, args.prompt, saved_paths, history)

    print("\n=== Final result ===")
    print(f"Count: {predicted_count}")
    print(f"Annotated image: {saved_path}")
    print(f"PDF report: {pdf_path}")
    print(f"History: {json.dumps(history, indent=2)}")


if __name__ == "__main__":
    main()
