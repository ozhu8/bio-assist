"""
Agentic cell-classification pipeline: a local Qwen3-VL manager orchestrates CellViT.

Unlike CountGD (a hosted Gradio Space taking a free-text object name), CellViT
(https://github.com/TIO-IKIM/CellViT) is a local, checkpoint-based nucleus
segmentation/classification model with a fixed five-class taxonomy (PanNuke:
Neoplastic, Inflammatory, Connective, Dead, Epithelial). So instead of turning
the user's request into free text, Qwen maps it onto a subset of those five
classes plus a confidence threshold, CellViT segments and classifies every
nucleus in the image, and Qwen evaluates the annotated result and retries
with an adjusted class selection/threshold if it looks wrong.

Qwen runs locally (transformers, same model/loading approach as
manager_agent.py's QwenVLM) rather than calling the Claude API -- no
ANTHROPIC_API_KEY is needed to run this script. There's no ground-truth expert
persona here (unlike manager_agent.py's CountGD/StarDist routing): PanNuke
supplies images but no per-nucleus-type ground truth to hand to one, so Qwen
always falls back to scoring its own result visually, 0-10.

CellViT's documented workflow (`cell_segmentation/inference/cell_detection.py`)
expects a whole-slide image already tiled into 1024x1024 patches with 64px
overlap. For a single input image (this script's unit of work, same as
CountGD's `--image`) we skip that WSI/tiling machinery and feed the image
directly through the loaded model as one patch, reusing CellViT's own
model-loading and per-patch post-processing code.

Requires a local clone of https://github.com/TIO-IKIM/CellViT (pass its path
via --cellvit-repo or put it on PYTHONPATH) and a downloaded model checkpoint
(e.g. CellViT-SAM-H, see the repo's README for download links).

Usage:
    python cellvit_agent.py --image tissue_patch.png \
        --prompt "count the neoplastic cells" \
        --checkpoint /path/to/CellViT-SAM-H-x40.pth \
        --cellvit-repo /path/to/CellViT
"""
import argparse
import json
import sys
import textwrap
from pathlib import Path
from typing import Optional, Protocol

import matplotlib.pyplot as plt # pyright: ignore[reportMissingModuleSource]
import numpy as np # pyright: ignore[reportMissingImports]
import torch # pyright: ignore[reportMissingImports]
from matplotlib.backends.backend_pdf import PdfPages # pyright: ignore[reportMissingModuleSource]
from PIL import Image, ImageDraw # pyright: ignore[reportMissingImports]

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
PDF_NAME = "cellvit_results.pdf"
PATCH_SIZE = 1024  # CellViT's required patch size (see cell_detection.py header comment)
NUCLEI_CLASSES = ["Neoplastic", "Inflammatory", "Connective", "Dead", "Epithelial"]


class QwenVLM:
    """Lazily loads Qwen3-VL and answers single image+text prompts with it.

    Same class as manager_agent.py's QwenVLM (trimmed to the ask/ask_json
    methods this script actually needs) -- duplicated rather than imported so
    this script stays standalone/self-contained like agentic_countgd.py and
    agentic_stardist.py, each with their own venv (see CLAUDE.md)."""

    def __init__(self, model_id: str = MODEL_ID, device_map: str = "auto"):
        self.model_id = model_id
        self.device_map = device_map
        self._model = None
        self._processor = None

    def _load(self):
        if self._model is None:
            from transformers import AutoModelForImageTextToText, AutoProcessor # pyright: ignore[reportMissingImports]
            self._model = AutoModelForImageTextToText.from_pretrained(
                self.model_id, dtype="auto", device_map=self.device_map
            )
            self._processor = AutoProcessor.from_pretrained(self.model_id)
        return self._model, self._processor

    def ask(self, image_path: str, prompt: str, max_new_tokens: int = 512) -> str:
        from qwen_vl_utils import process_vision_info # pyright: ignore[reportMissingImports]
        model, processor = self._load()
        assert processor is not None, "Processor failed to load"

        messages = [{
            "role": "user",
            "content": [{"type": "image", "image": image_path}, {"type": "text", "text": prompt}],
        }]
        chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[chat_text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
        ).to(model.device)

        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
        return processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()

    def ask_json(self, image_path: str, prompt: str, max_new_tokens: int = 512, required_keys: Optional[list] = None) -> dict:
        """Unlike the Claude calls this replaces, Qwen has no API-enforced JSON schema, so its
        free-text output can drop a requested key. Callers that will subscript the result (e.g.
        result["score"]) should pass required_keys so a malformed response fails here with a
        clear message instead of a bare KeyError deep in the caller."""
        raw = self.ask(image_path, prompt, max_new_tokens=max_new_tokens)
        start, end = raw.find("{"), raw.rfind("}")
        if start == -1 or end == -1:
            raise ValueError(f"Qwen response did not contain a JSON object: {raw!r}")
        result = json.loads(raw[start:end + 1])
        if required_keys:
            missing = [k for k in required_keys if k not in result]
            if missing:
                raise ValueError(f"Qwen JSON response is missing required key(s) {missing}: {result!r}")
        return result


class AsksJSON(Protocol):
    """Structural type for interpret_request/evaluate_result below -- both only ever call
    ask_json, so this accepts this module's own QwenVLM as well as manager_agent.QwenVLM (a
    structural superset with more methods this module doesn't need) without either module
    importing the other's class."""

    def ask_json(self, image_path: str, prompt: str, max_new_tokens: int = 512, required_keys: Optional[list] = None) -> dict: ...


def load_cellvit_module(cellvit_repo: Optional[str]):
    """Import CellViT's inference class, adding its repo to sys.path first if given."""
    if cellvit_repo:
        sys.path.insert(0, str(Path(cellvit_repo).resolve()))
    try:
        from cell_segmentation.inference.cell_detection import ( # pyright: ignore[reportMissingImports]
            CellSegmentationInference,
            COLOR_DICT,
        )
    except ImportError as exc:
        raise SystemExit(
            "Could not import CellViT. Clone https://github.com/TIO-IKIM/CellViT and pass "
            "--cellvit-repo /path/to/CellViT (or put it on PYTHONPATH)."
        ) from exc
    return CellSegmentationInference, COLOR_DICT


def load_patch(image_path: str) -> Image.Image:
    """Load an image as a PATCH_SIZE x PATCH_SIZE patch, preserving aspect ratio.

    Resizing directly to a square would stretch non-square inputs, distorting
    nucleus shapes before segmentation. Instead, downscale to fit within
    PATCH_SIZE and letterbox (center on a black canvas) so cells keep their
    true proportions.
    """
    image = Image.open(image_path).convert("RGB")
    image.thumbnail((PATCH_SIZE, PATCH_SIZE), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (PATCH_SIZE, PATCH_SIZE))
    offset = ((PATCH_SIZE - image.width) // 2, (PATCH_SIZE - image.height) // 2)
    canvas.paste(image, offset)
    return canvas


def autocast_device_type(inferer) -> str:
    """Derive the torch.autocast device_type from the inferer's actual device
    instead of assuming CUDA, so CPU-only inference doesn't crash."""
    device = inferer.device
    return getattr(device, "type", str(device).split(":")[0])


def run_cellvit(inferer, image_path: str, magnification: float, target_classes: set, prob_threshold: float, color_dict: dict):
    """Run one CellViT forward pass, treating the whole image as a single 1024x1024 patch.
    Returns (annotated, matched, counts_by_type, cells) -- matched/counts_by_type are filtered
    by target_classes/prob_threshold (what the user asked for), while cells is every detected
    nucleus regardless of that filter (contour in load_patch's 1024x1024 letterboxed-patch
    coordinate space, type_name, type_prob) -- needed by callers that score the underlying
    per-class classification quality against ground truth, independent of what was requested."""
    image = load_patch(image_path)
    patch = inferer.inference_transforms(image).unsqueeze(0).to(inferer.device)

    with torch.no_grad():
        if inferer.mixed_precision:
            with torch.autocast(device_type=autocast_device_type(inferer), dtype=torch.float16):
                predictions = inferer.model.forward(patch, retrieve_tokens=True)
        else:
            predictions = inferer.model.forward(patch, retrieve_tokens=True)
        instance_types, _ = inferer.get_cell_predictions_with_tokens(predictions, magnification=magnification)

    nuclei_types = inferer.run_conf["dataset_config"]["nuclei_types"]
    background_id = nuclei_types.get("Background", 0)
    name_by_id = {type_id: name for name, type_id in nuclei_types.items()}

    cells = []
    for cell in instance_types[0].values():
        if cell["type"] == background_id:
            continue
        cells.append({
            # CellViT stores contour points as (row, col); flip to (x, y) for PIL drawing.
            "contour": [(pt[1], pt[0]) for pt in cell["contour"]],
            "type": int(cell["type"]),
            "type_name": name_by_id.get(cell["type"], "Unknown"),
            "type_prob": float(cell["type_prob"]),
        })

    matched = [c for c in cells if c["type_prob"] >= prob_threshold and c["type_name"] in target_classes]
    counts_by_type = {}
    for c in cells:
        counts_by_type[c["type_name"]] = counts_by_type.get(c["type_name"], 0) + 1

    annotated = draw_annotations(image, cells, matched, color_dict)
    return annotated, matched, counts_by_type, cells


def draw_annotations(image: Image.Image, all_cells: list, matched_cells: list, color_dict: dict) -> Image.Image:
    """Outline every detected nucleus; matched (target-class, above-threshold) cells get a thicker outline."""
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    matched_ids = {id(c) for c in matched_cells}
    for cell in all_cells:
        contour = cell["contour"]
        if len(contour) < 2:
            continue
        color = tuple(color_dict.get(cell["type"], [255, 255, 255]))
        width = 4 if id(cell) in matched_ids else 1
        draw.line(contour + [contour[0]], fill=color, width=width)
    return annotated


def run_raw_inference(inferer, image_path: str, magnification: float):
    """One forward pass with no class-filtering or annotation overlay: returns
    CellViT's own raw output, the instance map (per-pixel nucleus IDs, 0 =
    background) and per-nucleus type dicts from model.calculate_instance_map."""
    image = load_patch(image_path)
    patch = inferer.inference_transforms(image).unsqueeze(0).to(inferer.device)

    with torch.no_grad():
        if inferer.mixed_precision:
            with torch.autocast(device_type=autocast_device_type(inferer), dtype=torch.float16):
                predictions = inferer.model.forward(patch, retrieve_tokens=True)
        else:
            predictions = inferer.model.forward(patch, retrieve_tokens=True)

        # calculate_instance_map expects post-softmax maps (mirrors
        # get_cell_predictions_with_tokens in cell_detection.py).
        predictions["nuclei_binary_map"] = torch.softmax(predictions["nuclei_binary_map"], dim=1)
        predictions["nuclei_type_map"] = torch.softmax(predictions["nuclei_type_map"], dim=1)
        instance_map, instance_types = inferer.model.calculate_instance_map(
            predictions, magnification=magnification
        )

    nuclei_types = inferer.run_conf["dataset_config"]["nuclei_types"]
    name_by_id = {type_id: name for name, type_id in nuclei_types.items()}
    instance_map = instance_map[0].cpu().numpy().astype(np.int32)  # (H, W), batch size 1

    nuclei = []
    for local_id, cell in instance_types[0].items():
        nuclei.append({
            "instance_id": int(local_id),
            "bbox": np.asarray(cell["bbox"]).tolist(),
            "centroid": np.asarray(cell["centroid"]).tolist(),
            "contour": np.asarray(cell["contour"]).tolist(),
            "type": int(cell["type"]),
            "type_name": name_by_id.get(cell["type"], "Unknown"),
            "type_prob": float(cell["type_prob"]),
        })

    return instance_map, nuclei


def interpret_request(qwen: AsksJSON, user_prompt: str, image_path: str) -> dict:
    prompt = (
        "CellViT classifies nuclei in histopathology images into exactly five "
        f"fixed classes: {', '.join(NUCLEI_CLASSES)}. The user wants: "
        f"\"{user_prompt}\"\n\n"
        "Map their request onto one or more of these five class names (pick every "
        "class that plausibly matches what they're asking about; if they clearly "
        "want everything counted, include all five). Also pick a type_prob "
        "confidence threshold in [0, 1] for keeping a detection - 0.5 is a "
        "reasonable default, raise it if the request implies only confident/obvious "
        "cells and lower it if it implies catching everything.\n\n"
        "Reply with ONLY a JSON object matching this schema: "
        f"{{\"target_classes\": [one or more of {json.dumps(NUCLEI_CLASSES)}], "
        "\"prob_threshold\": number}}"
    )
    result = qwen.ask_json(image_path, prompt, required_keys=["target_classes", "prob_threshold"])
    invalid = [c for c in result["target_classes"] if c not in NUCLEI_CLASSES]
    if invalid or not result["target_classes"]:
        raise ValueError(f"Qwen returned invalid target_classes: {result!r}")
    return result


def evaluate_result(
    qwen: AsksJSON,
    user_prompt: str,
    target_classes: list,
    prob_threshold: float,
    predicted_count: int,
    counts_by_type: dict,
    annotated_image_path: str,
    history: list,
) -> dict:
    prompt = (
        f"Original user request: \"{user_prompt}\"\n"
        f"CellViT was asked to highlight: {target_classes} (type_prob >= {prob_threshold})\n"
        f"Matched cell count: {predicted_count}\n"
        f"All detected cells by type: {json.dumps(counts_by_type)}\n"
        f"Prior attempts this session: {json.dumps(history, default=str)}\n\n"
        "The attached image shows every nucleus CellViT detected as a colored "
        "outline (thick outline = matches the target class(es) and threshold, thin "
        "outline = detected but excluded). Evaluate: (1) do the thick outlines look "
        "visually accurate for the requested class(es) (no obvious misclassifications, "
        "missed nuclei, or false positives)? (2) is the matched count histologically "
        "plausible for what's shown? (3) does this satisfy the user's original "
        "request?\n"
        "Score 0-10. If score < 7 and a different class selection or threshold would "
        "plausibly fix it, set accept=false and give revised_target_classes and/or "
        "revised_prob_threshold to retry with. Otherwise set accept=true and leave "
        "both revised fields null.\n\n"
        "Reply with ONLY a JSON object matching this schema: {\"accept\": bool, "
        "\"score\": int, \"feedback\": str (1-2 sentences), "
        "\"revised_target_classes\": array or null, \"revised_prob_threshold\": number or null}"
    )
    return qwen.ask_json(
        annotated_image_path, prompt, max_new_tokens=768,
        required_keys=["accept", "score", "feedback"],
    )


ACCEPT_SCORE_THRESHOLD = 7

SCORING_RUBRIC = (
    "Each iteration is scored 0-10 by evaluating the annotated image against three criteria:\n"
    "  1. Do the highlighted (thick-outline) nuclei look visually accurate for the requested\n"
    "     class(es) (no obvious misclassifications, missed nuclei, or false positives)?\n"
    "  2. Is the matched count histologically plausible for what's shown?\n"
    "  3. Does the result satisfy the user's original request?\n\n"
    f"A score >= {ACCEPT_SCORE_THRESHOLD} accepts the result. A score below that triggers a retry\n"
    "with a revised class selection and/or confidence threshold, if that would plausibly fix the\n"
    "issue. A result with zero matched cells on an image that clearly contains the target class\n"
    "fails all three criteria outright and scores 0, regardless of the threshold used.\n\n"
    "Loops to good result: the number of iterations run before a score reached the accept\n"
    "threshold (or the total number run, if none did) - a measure of how many retries the\n"
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
            classes_label = ", ".join(entry["target_classes"])
            ax_img.set_title(
                f"Iteration {entry['iteration']}: counting {classes_label} "
                f"(p>={entry['prob_threshold']:.2f})"
            )

            ax_text.axis("off")
            caption = (
                f"Request: {user_prompt}\n"
                f"Matched count: {entry['predicted_count']}\n"
                f"All detected cells by type: {json.dumps(entry['counts_by_type'])}\n"
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
            f"Final target classes: {', '.join(final['target_classes'])!r}\n"
            f"Final prob_threshold: {final['prob_threshold']:.2f}\n"
            f"Final matched count: {final['predicted_count']}\n"
            f"Final score: {final['score']}/10"
        )
        ax.text(0, 0.95, summary, va="top", ha="left", fontsize=11, wrap=True)
        pdf.savefig(fig)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Run CellViT with a local Qwen3-VL manager as orchestrator/evaluator")
    parser.add_argument("--image", required=True, help="Path to the input image (treated as one patch)")
    parser.add_argument("--prompt", default=None, help="What to count / user instruction (ignored with --raw-only)")
    parser.add_argument(
        "--raw-only", action="store_true",
        help="Skip the Qwen agent loop; run a single CellViT forward pass and dump raw output "
             "(instance mask + per-nucleus type predictions) with no evaluation/retries",
    )
    parser.add_argument("--checkpoint", required=True, help="Path to a CellViT model checkpoint (.pth)")
    parser.add_argument("--cellvit-repo", default=None, help="Path to a local clone of TIO-IKIM/CellViT")
    parser.add_argument("--gpu", type=int, default=0, help="CUDA GPU id for inference")
    parser.add_argument("--magnification", type=float, default=40, help="Network magnification")
    parser.add_argument("--enforce-amp", action="store_true", help="Force mixed-precision inference")
    parser.add_argument("--max-iterations", type=int, default=5)
    parser.add_argument("--output-dir", default="./cellvit_agent_output")
    parser.add_argument("--pdf-name", default=PDF_NAME, help="Filename for the saved PDF report")
    parser.add_argument("--model-id", default=MODEL_ID, help="Hugging Face repo id for the Qwen manager")
    args = parser.parse_args()
    if args.max_iterations < 1:
        parser.error("--max-iterations must be at least 1")
    if not args.raw_only and not args.prompt:
        parser.error("--prompt is required unless --raw-only is set")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    CellSegmentationInference, COLOR_DICT = load_cellvit_module(args.cellvit_repo)
    inferer = CellSegmentationInference(
        model_path=args.checkpoint, gpu=args.gpu, enforce_mixed_precision=args.enforce_amp
    )

    if args.raw_only:
        instance_map, nuclei = run_raw_inference(inferer, args.image, args.magnification)

        print(f"Instance mask: shape={instance_map.shape}, dtype={instance_map.dtype}")
        print(f"Unique instance IDs (excluding background=0): {len(np.unique(instance_map)) - 1}")
        print(f"Per-nucleus predictions: {len(nuclei)} nuclei detected")
        if nuclei:
            print("Example nucleus record (first of the list):")
            print(json.dumps(nuclei[0], indent=2))

        mask_path = output_dir / "instance_mask.npy"
        np.save(mask_path, instance_map)
        nuclei_path = output_dir / "nuclei.json"
        with open(nuclei_path, "w") as f:
            json.dump(nuclei, f, indent=2)

        print(f"\nSaved raw instance mask (.npy, shape {instance_map.shape}): {mask_path}")
        print(f"Saved raw per-nucleus labels (.json, {len(nuclei)} records): {nuclei_path}")
        return

    qwen = QwenVLM(args.model_id)
    request = interpret_request(qwen, args.prompt, args.image)
    target_classes = set(request["target_classes"])
    prob_threshold = request["prob_threshold"]
    print(f"[Qwen] target classes: {sorted(target_classes)}, prob_threshold={prob_threshold:.2f}")

    history = []
    saved_paths = []
    saved_path = None
    predicted_count = None
    counts_by_type = {}
    for i in range(1, args.max_iterations + 1):
        print(f"\n--- Iteration {i}: CellViT highlighting {sorted(target_classes)} (p>={prob_threshold:.2f}) ---")
        annotated_image, matched_cells, counts_by_type, _cells = run_cellvit(
            inferer, args.image, args.magnification, target_classes, prob_threshold, COLOR_DICT
        )
        predicted_count = len(matched_cells)
        print(f"[CellViT] matched count={predicted_count}, all detected by type={counts_by_type}")

        saved_path = output_dir / f"iteration_{i}.png"
        annotated_image.save(saved_path)
        saved_paths.append(saved_path)

        eval_result = evaluate_result(
            qwen, args.prompt, sorted(target_classes), prob_threshold,
            predicted_count, counts_by_type, str(saved_path), history,
        )
        print(f"[Qwen eval] score={eval_result['score']} accept={eval_result['accept']}")
        print(f"[Qwen eval] feedback: {eval_result['feedback']}")

        history.append({
            "iteration": i,
            "target_classes": sorted(target_classes),
            "prob_threshold": prob_threshold,
            "predicted_count": predicted_count,
            "counts_by_type": counts_by_type,
            "score": eval_result["score"],
            "feedback": eval_result["feedback"],
        })

        revised_classes = eval_result.get("revised_target_classes")
        revised_threshold = eval_result.get("revised_prob_threshold")
        if eval_result["accept"] or (not revised_classes and revised_threshold is None):
            break
        if revised_classes:
            target_classes = set(revised_classes)
        if revised_threshold is not None:
            prob_threshold = revised_threshold

    pdf_path = output_dir / args.pdf_name
    save_pdf_report(pdf_path, args.prompt, saved_paths, history)

    print("\n=== Final result ===")
    print(f"Matched count: {predicted_count}")
    print(f"All detected by type: {counts_by_type}")
    print(f"Annotated image: {saved_path}")
    print(f"PDF report: {pdf_path}")
    print(f"History: {json.dumps(history, indent=2)}")


if __name__ == "__main__":
    main()
