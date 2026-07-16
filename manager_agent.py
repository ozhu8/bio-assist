"""
Manager agent: Qwen3-VL-8B-Instruct (local, via Hugging Face transformers)
replaces Claude as the orchestrator across both specialist agents.

Given a task description + image, Qwen:
  1. picks which specialist agent to call -- CountGD (counting) or StarDist
     (nucleus segmentation) -- by reading the task and looking at the image.
  2. for CountGD, turns the task into a short count target (e.g. "count the
     cells" -> "cell"), same role as agentic_countgd.py's interpret_prompt.
  3. runs the retry loop itself, up to --max-iterations times. Each agent is
     scored with its own metric, matching what the standalone scripts already
     use for the cases where ground truth is available:
       - CountGD: Mean Absolute Error (MAE) against a known ground-truth
         count, when one is supplied (e.g. from the BBBC005 manifest, which
         has a ground_truth_count per synthetic image).
       - StarDist: Panoptic Quality (PQ), reusing agentic_stardist.py's own
         compute_panoptic_quality, when a ground-truth instance mask is
         supplied (e.g. from a PanNuke sample -- PanNuke images and their
         ground truth are pulled the same way agentic_stardist.py's
         --pannuke-index path does, via load_pannuke_sample).
     If no ground truth is supplied for the routed agent (a generic image
     with no known answer), there's no metric to compute, so Qwen falls back
     to visually scoring the annotated/outlined image 0-10 instead -- the
     same fallback either standalone script would face with an arbitrary
     --image and no ground truth.
     Either way, when the result isn't accepted, Qwen looks at the image (and
     the metric breakdown, if there is one) and proposes what to try next --
     a revised count target for CountGD, revised prob_thresh/nms_thresh for
     StarDist. This mirrors evaluate_result in agentic_countgd.py/
     agentic_stardist.py, with Qwen driving it instead of Claude.

agentic_countgd.py and agentic_stardist.py are untouched -- this file only
imports their run_countgd/run_stardist/compute_panoptic_quality (and
StarDist's save_instance_outlines/load_pannuke_sample) and drives them with
Qwen instead of Claude.

Setup (needs its own venv -- these are not installed in .venv-countgd/
.venv-stardist):
    pip install torch transformers accelerate qwen-vl-utils pillow
Qwen3-VL is very new; if `AutoModelForImageTextToText` doesn't recognize its
config, install transformers from source instead:
    pip install git+https://github.com/huggingface/transformers

No GPU is required to load the model, but this repo's dev machine has none
detected (no nvidia-smi) -- expect slow (CPU-bound) generation. An 8B model
in bf16 is ~16GB of weights alone; if you do have a small CUDA GPU, loading
in 4-bit via bitsandbytes is the practical way to fit it.

Usage:
    python manager_agent.py --image cells.png --task "count the individual cells"
    python manager_agent.py --image tissue.png --task "segment the individual nuclei"
"""
import argparse
import json
from pathlib import Path

import numpy as np # pyright: ignore[reportMissingImports]
from gradio_client import Client # pyright: ignore[reportMissingImports]
from PIL import Image # pyright: ignore[reportMissingImports]
from stardist.models import StarDist2D # pyright: ignore[reportMissingImports]

from agentic_countgd import COUNTGD_SPACE, run_countgd
from agentic_stardist import (
    PRETRAINED_MODEL,
    best_entry,
    compute_panoptic_quality,
    load_image,
    load_pannuke_sample,
    run_stardist,
    save_instance_outlines,
)

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
ACCEPT_SCORE_THRESHOLD = 7  # Qwen's own 0-10 visual score, used when there's no ground truth to measure against
ACCEPT_PQ_THRESHOLD = 0.5   # StarDist acceptance bar when a ground-truth instance mask is available (PanNuke)
MAE_TOLERANCE_FRACTION = 0.1  # CountGD acceptance bar when a ground-truth count is available (e.g. BBBC005)


def mae_accept_tolerance(ground_truth_count: int) -> float:
    """Accept a CountGD result if its MAE is within this many counts of the ground truth --
    10% of the ground-truth count, with a floor of 1 so small counts aren't impossible to hit."""
    return max(1, round(MAE_TOLERANCE_FRACTION * ground_truth_count))


class QwenVLM:
    """Lazily loads Qwen3-VL and answers single image+text prompts with it."""

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

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt},
            ],
        }]
        chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[chat_text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
        ).to(model.device)

        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
        return processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()

    def ask_json(self, image_path: str, prompt: str, max_new_tokens: int = 512, required_keys: list = None) -> dict:
        """Unlike the Claude calls in agentic_countgd.py/agentic_stardist.py, Qwen has no
        API-enforced JSON schema, so its free-text output can drop a requested key. Callers
        that will subscript the result (e.g. result["score"]) should pass required_keys so a
        malformed response fails here with a clear message instead of a bare KeyError deep in
        the caller."""
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


def select_agent(qwen: QwenVLM, task_description: str, image_path: str) -> str:
    prompt = (
        "You are a manager agent that routes an image-analysis task to one of two "
        "specialist tools:\n"
        "  - countgd: counts individual objects/cells matching a described category. "
        "Use it for tasks about how many of something there are.\n"
        "  - stardist: segments every cell nucleus in the image into instance masks. "
        "Use it for tasks about outlining, segmenting, or delineating individual nuclei/cells.\n\n"
        f"Task: \"{task_description}\"\n\n"
        "Reply with ONLY a JSON object: {\"agent\": \"countgd\" or \"stardist\", \"reason\": \"one sentence\"}"
    )
    result = qwen.ask_json(image_path, prompt)
    agent = result.get("agent", "").strip().lower()
    if agent not in ("countgd", "stardist"):
        raise ValueError(f"Qwen returned an unrecognized agent choice: {result!r}")
    return agent


def interpret_countgd_target(qwen: QwenVLM, task_description: str, image_path: str) -> str:
    prompt = (
        f"The user wants to count objects in this image. Their request: \"{task_description}\"\n\n"
        "Reply with ONLY a short noun phrase (1-3 words) naming the single object type to "
        "count (e.g. 'cell', 'car', 'strawberry'). No punctuation, no explanation, nothing else."
    )
    return qwen.ask(image_path, prompt).strip().strip('."\'')


def evaluate_countgd_visual(
    qwen: QwenVLM, task_description: str, count_target: str, predicted_count: int,
    annotated_image_path: str, history: list,
) -> dict:
    """No ground truth available -- Qwen both scores (0-10) and decides accept/reject by eye."""
    prompt = (
        f"Original user request: \"{task_description}\"\n"
        f"CountGD was asked to count: \"{count_target}\"\n"
        f"Predicted count: {predicted_count}\n"
        f"Prior attempts this session: {json.dumps(history)}\n\n"
        "The attached image shows CountGD's detections as boxes/heatmap. Evaluate: "
        "(1) do the boxes look visually accurate (no obvious double-counts, missed objects, "
        "or false positives)? (2) is the count plausible? (3) does this satisfy the user's "
        "original request?\n"
        f"Score 0-10. If score < {ACCEPT_SCORE_THRESHOLD} and a different/more specific text "
        "prompt would plausibly fix it, set accept=false and give revised_text to retry with. "
        "Otherwise set accept=true and revised_text=null.\n\n"
        "Reply with ONLY a JSON object matching this schema: "
        "{\"accept\": bool, \"score\": int, \"feedback\": str, \"revised_text\": str or null}"
    )
    return qwen.ask_json(annotated_image_path, prompt, required_keys=["accept", "score", "feedback"])


def propose_countgd_revision(
    qwen: QwenVLM, task_description: str, count_target: str, predicted_count: int,
    ground_truth_count: int, mae: float, annotated_image_path: str, history: list,
) -> dict:
    """Ground truth available -- MAE decides accept/reject (see run_countgd_with_feedback);
    Qwen is only asked to propose a better count target, not to judge the result itself."""
    prompt = (
        f"Original user request: \"{task_description}\"\n"
        f"CountGD was asked to count: \"{count_target}\"\n"
        f"Predicted count: {predicted_count}  |  Ground-truth count: {ground_truth_count}  |  "
        f"MAE: {mae} (above the tolerance of {mae_accept_tolerance(ground_truth_count)})\n"
        f"Prior attempts this session: {json.dumps(history)}\n\n"
        "The attached image shows CountGD's detections as boxes/heatmap. The predicted count is "
        "off from the known ground truth by the MAE above. Look at the detections and propose a "
        "different/more specific text prompt that would plausibly reduce the error -- e.g. if the "
        "count is far too low, the target may be too narrow or missing overlapping objects; if "
        "far too high, the target may be matching background clutter or double-counting.\n\n"
        "Reply with ONLY a JSON object matching this schema: {\"revised_text\": str, \"feedback\": str}"
    )
    return qwen.ask_json(annotated_image_path, prompt, required_keys=["feedback"])


def evaluate_stardist_visual(
    qwen: QwenVLM, task_description: str, prob_thresh: float, nms_thresh: float,
    predicted_count: int, outlines_image_path: str, history: list,
) -> dict:
    """No ground truth available -- Qwen both scores (0-10) and decides accept/reject by eye."""
    prompt = (
        f"Original user request: \"{task_description}\"\n"
        f"StarDist ran with prob_thresh={prob_thresh:.3f}, nms_thresh={nms_thresh:.3f}\n"
        f"Detected nuclei: {predicted_count}\n"
        f"Prior attempts this session: {json.dumps(history)}\n\n"
        "The attached image shows the original tissue with each StarDist-detected nucleus "
        "outlined in a distinct color. Evaluate: (1) do the outlines look visually accurate "
        "(no obvious missed nuclei, false positives, or merged/split instances)? (2) is the "
        "nucleus count plausible for what's shown? (3) does this satisfy the user's original "
        "request?\n"
        f"Score 0-10. If score < {ACCEPT_SCORE_THRESHOLD}, propose revised threshold(s): raise "
        "prob_thresh if you see false-positive outlines on background/noise, lower it if real "
        "nuclei look missed; lower nms_thresh if you see duplicate/split outlines around one "
        "nucleus, raise it if adjacent distinct nuclei look merged into one outline. Only set "
        "the threshold(s) that address the problem -- leave the other null. Otherwise set "
        "accept=true and leave both revised fields null.\n\n"
        "Reply with ONLY a JSON object matching this schema: {\"accept\": bool, \"score\": int, "
        "\"feedback\": str, \"revised_prob_thresh\": number or null, \"revised_nms_thresh\": number or null}"
    )
    return qwen.ask_json(outlines_image_path, prompt, required_keys=["accept", "score", "feedback"])


def propose_stardist_revision(
    qwen: QwenVLM, task_description: str, prob_thresh: float, nms_thresh: float,
    pq_result: dict, outlines_image_path: str, history: list,
) -> dict:
    """Ground truth available -- PQ decides accept/reject (see run_stardist_with_feedback);
    Qwen is only asked to propose revised thresholds, not to judge the result itself."""
    prompt = (
        f"Original user request: \"{task_description}\"\n"
        f"StarDist ran with prob_thresh={prob_thresh:.3f}, nms_thresh={nms_thresh:.3f}\n"
        f"Panoptic Quality against ground truth: PQ={pq_result['pq']:.3f} "
        f"(below the {ACCEPT_PQ_THRESHOLD} acceptance bar)\n"
        f"  mean IoU of matched instances: {pq_result['mean_iou']:.3f}\n"
        f"  TP={pq_result['tp']}  FP={pq_result['fp']}  FN={pq_result['fn']}\n"
        f"Prior attempts this session: {json.dumps(history)}\n\n"
        "The attached image shows the original tissue with each StarDist-detected nucleus outlined "
        "in a distinct color. FP = spurious detections with no matching ground-truth nucleus; "
        "FN = ground-truth nuclei StarDist missed; a low mean IoU on matched pairs means boundaries "
        "are poorly aligned or instances are being split/merged. Using both these numbers and the "
        "image, propose revised threshold(s):\n"
        "  - prob_thresh (0-1): raise it if FP is high, lower it if FN is high.\n"
        "  - nms_thresh (0-1): lower it if you see duplicate/split outlines around one nucleus; "
        "raise it if adjacent distinct nuclei look merged into a single outline.\n"
        "Only set the threshold(s) that address the problem -- leave the other null.\n\n"
        "Reply with ONLY a JSON object matching this schema: "
        "{\"revised_prob_thresh\": number or null, \"revised_nms_thresh\": number or null, \"feedback\": str}"
    )
    return qwen.ask_json(outlines_image_path, prompt, required_keys=["feedback"])


def run_countgd_with_feedback(
    qwen: QwenVLM, countgd_client: Client, image_path: str, task_description: str,
    max_iterations: int, output_dir: Path, ground_truth_count: int = None,
) -> dict:
    """If ground_truth_count is given, MAE against it decides accept/reject each iteration
    (see mae_accept_tolerance) and Qwen only proposes a revised count target (propose_countgd_revision).
    Otherwise there's nothing to compute MAE against, so Qwen scores the result visually and decides
    accept/reject itself (evaluate_countgd_visual)."""
    count_target = interpret_countgd_target(qwen, task_description, image_path)
    print(f"[Qwen] counting target: {count_target!r}")

    history = []
    saved_path = None
    predicted_count = None
    for i in range(1, max_iterations + 1):
        print(f"--- Iteration {i}: CountGD counting {count_target!r} ---")
        annotated_path, predicted_count = run_countgd(countgd_client, image_path, count_target)
        saved_path = output_dir / f"countgd_iteration_{i}.png"
        saved_path.write_bytes(Path(annotated_path).read_bytes())
        print(f"[CountGD] count={predicted_count}")

        if ground_truth_count is not None:
            mae = abs(predicted_count - ground_truth_count)
            accept = mae <= mae_accept_tolerance(ground_truth_count)
            if accept:
                feedback, revised_text = "MAE met the acceptance tolerance.", None
            else:
                proposal = propose_countgd_revision(
                    qwen, task_description, count_target, predicted_count,
                    ground_truth_count, mae, str(saved_path), history,
                )
                feedback, revised_text = proposal["feedback"], proposal.get("revised_text")
            print(f"[metric] MAE={mae} accept={accept}")
            history.append({
                "iteration": i, "count_target": count_target, "predicted_count": predicted_count,
                "ground_truth_count": ground_truth_count, "mae": mae, "accept": accept, "feedback": feedback,
            })
        else:
            eval_result = evaluate_countgd_visual(qwen, task_description, count_target, predicted_count, str(saved_path), history)
            accept, revised_text, feedback = eval_result["accept"], eval_result.get("revised_text"), eval_result["feedback"]
            print(f"[Qwen eval] score={eval_result['score']} accept={accept}")
            history.append({
                "iteration": i, "count_target": count_target, "predicted_count": predicted_count,
                "score": eval_result["score"], "accept": accept, "feedback": feedback,
            })

        if accept or not revised_text:
            break
        count_target = revised_text

    return {
        "agent": "countgd", "count_target": count_target, "count": predicted_count,
        "annotated_image": saved_path, "history": history,
    }


def run_stardist_with_feedback(
    qwen: QwenVLM, model: StarDist2D, image_path: str, task_description: str,
    max_iterations: int, output_dir: Path, ground_truth_labels: np.ndarray = None,
) -> dict:
    """If ground_truth_labels is given (a PanNuke-style instance mask), Panoptic Quality against
    it decides accept/reject each iteration (see compute_panoptic_quality/ACCEPT_PQ_THRESHOLD) and
    Qwen only proposes revised thresholds (propose_stardist_revision). Otherwise there's no ground
    truth to compute PQ against, so Qwen scores the result visually and decides accept/reject itself
    (evaluate_stardist_visual)."""
    image = load_image(image_path)
    prob_thresh, nms_thresh = round(model.thresholds.prob, 3), round(model.thresholds.nms, 3)

    history = []
    saved_path = None
    labels = None
    for i in range(1, max_iterations + 1):
        print(f"--- Iteration {i}: StarDist prob_thresh={prob_thresh:.3f}, nms_thresh={nms_thresh:.3f} ---")
        labels, _ = run_stardist(model, image, prob_thresh=prob_thresh, nms_thresh=nms_thresh)
        predicted_count = int(labels.max())
        saved_path = output_dir / f"stardist_iteration_{i}.png"
        save_instance_outlines(image, labels, saved_path)
        print(f"[StarDist] nuclei={predicted_count}")

        if ground_truth_labels is not None:
            pq_result = compute_panoptic_quality(labels, ground_truth_labels)
            accept = pq_result["pq"] >= ACCEPT_PQ_THRESHOLD
            if accept:
                feedback, revised_prob, revised_nms = "PQ met the acceptance threshold.", None, None
            else:
                proposal = propose_stardist_revision(
                    qwen, task_description, prob_thresh, nms_thresh, pq_result, str(saved_path), history
                )
                feedback = proposal["feedback"]
                revised_prob, revised_nms = proposal.get("revised_prob_thresh"), proposal.get("revised_nms_thresh")
            print(f"[metric] PQ={pq_result['pq']:.3f} accept={accept}")
            history.append({
                "iteration": i, "prob_thresh": prob_thresh, "nms_thresh": nms_thresh,
                "predicted_count": predicted_count, "pq": pq_result["pq"], "mean_iou": pq_result["mean_iou"],
                "tp": pq_result["tp"], "fp": pq_result["fp"], "fn": pq_result["fn"],
                "accept": accept, "feedback": feedback,
            })
        else:
            eval_result = evaluate_stardist_visual(
                qwen, task_description, prob_thresh, nms_thresh, predicted_count, str(saved_path), history
            )
            accept = eval_result["accept"]
            revised_prob, revised_nms, feedback = eval_result.get("revised_prob_thresh"), eval_result.get("revised_nms_thresh"), eval_result["feedback"]
            print(f"[Qwen eval] score={eval_result['score']} accept={accept}")
            history.append({
                "iteration": i, "prob_thresh": prob_thresh, "nms_thresh": nms_thresh,
                "predicted_count": predicted_count, "score": eval_result["score"],
                "accept": accept, "feedback": feedback,
            })

        if accept or (revised_prob is None and revised_nms is None):
            break
        if revised_prob is not None:
            prob_thresh = revised_prob
        if revised_nms is not None:
            nms_thresh = revised_nms

    if ground_truth_labels is not None:
        best = best_entry(history)
        if best["iteration"] != history[-1]["iteration"]:
            print(
                f"  search continued past its best result -- reverting to iteration "
                f"{best['iteration']} (PQ={best['pq']:.3f}) instead of the last one tried "
                f"(PQ={history[-1]['pq']:.3f})"
            )
            labels, _ = run_stardist(model, image, prob_thresh=best["prob_thresh"], nms_thresh=best["nms_thresh"])
            saved_path = output_dir / f"stardist_iteration_{best['iteration']}.png"

    return {
        "agent": "stardist", "num_nuclei": int(labels.max()), "labels": labels,
        "outlines_image": saved_path, "history": history,
    }


class ManagerAgent:
    """Routes a task to CountGD or StarDist using Qwen3-VL, and drives Qwen's own
    retry/scoring loop against whichever agent it picked."""

    def __init__(self, model_id: str = MODEL_ID):
        self.qwen = QwenVLM(model_id)
        self._countgd_client = None
        self._stardist_model = None

    @property
    def countgd_client(self) -> Client:
        if self._countgd_client is None:
            self._countgd_client = Client(COUNTGD_SPACE)
        return self._countgd_client

    @property
    def stardist_model(self) -> StarDist2D:
        if self._stardist_model is None:
            self._stardist_model = StarDist2D.from_pretrained(PRETRAINED_MODEL)
        return self._stardist_model

    def run(
        self, task_description: str, image_path: str, max_iterations: int = 3,
        output_dir: str = "./manager_agent_output",
        ground_truth_count: int = None, ground_truth_labels: np.ndarray = None,
    ) -> dict:
        """ground_truth_count (used only if routed to CountGD) and ground_truth_labels (used only
        if routed to StarDist) are both optional -- see run_countgd_with_feedback/
        run_stardist_with_feedback for what happens when the relevant one is left out."""
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        agent = select_agent(self.qwen, task_description, image_path)
        print(f"[Qwen] routed to: {agent}")

        if agent == "countgd":
            return run_countgd_with_feedback(
                self.qwen, self.countgd_client, image_path, task_description, max_iterations, out_dir,
                ground_truth_count=ground_truth_count,
            )
        return run_stardist_with_feedback(
            self.qwen, self.stardist_model, image_path, task_description, max_iterations, out_dir,
            ground_truth_labels=ground_truth_labels,
        )


def main():
    parser = argparse.ArgumentParser(description="Qwen3-VL-managed dispatch to CountGD or StarDist")
    parser.add_argument("--image", default=None, help="Path to the input image (ignored if --pannuke-index is set)")
    parser.add_argument("--task", required=True, help="Task description, e.g. 'count the cells' or 'segment the nuclei'")
    parser.add_argument(
        "--ground-truth-count", type=int, default=None,
        help="Known true count for the image (e.g. from the BBBC005 manifest) -- if set and the "
             "task routes to CountGD, MAE against this decides accept/reject instead of Qwen's visual score",
    )
    parser.add_argument(
        "--ground-truth-labels", default=None,
        help="Path to a .npy ground-truth instance-label mask -- if set and the task routes to "
             "StarDist, Panoptic Quality against this decides accept/reject instead of Qwen's visual score",
    )
    parser.add_argument(
        "--pannuke-index", type=int, default=None,
        help="Instead of --image/--ground-truth-labels, pull one PanNuke sample (image + its real "
             "ground-truth instance mask) at this index and use it directly",
    )
    parser.add_argument("--pannuke-fold", type=int, default=1, choices=[1, 2, 3], help="PanNuke fold to pull from")
    parser.add_argument("--max-iterations", type=int, default=3)
    parser.add_argument("--output-dir", default="./manager_agent_output")
    parser.add_argument("--model-id", default=MODEL_ID, help="Hugging Face repo id for the manager VLM")
    args = parser.parse_args()
    if args.image is None and args.pannuke_index is None:
        parser.error("one of --image or --pannuke-index is required")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_path = args.image
    ground_truth_labels = None
    if args.pannuke_index is not None:
        print(f"Fetching PanNuke fold {args.pannuke_fold} image {args.pannuke_index}...")
        image, ground_truth_labels, tissue = load_pannuke_sample(args.pannuke_fold, args.pannuke_index)
        print(f"tissue={tissue}  ground-truth nuclei={int(ground_truth_labels.max())}")
        image_path = str(output_dir / f"pannuke_fold{args.pannuke_fold}_{args.pannuke_index:02d}.png")
        Image.fromarray(image).save(image_path)
    elif args.ground_truth_labels is not None:
        ground_truth_labels = np.load(args.ground_truth_labels)

    manager = ManagerAgent(model_id=args.model_id)
    result = manager.run(
        args.task, image_path, args.max_iterations, args.output_dir,
        ground_truth_count=args.ground_truth_count, ground_truth_labels=ground_truth_labels,
    )

    print("\n=== Final result ===")
    print(f"Agent used: {result['agent']}")
    if result["agent"] == "countgd":
        print(f"Count target: {result['count_target']!r}")
        print(f"Predicted count: {result['count']}")
        print(f"Annotated image: {result['annotated_image']}")
    else:
        print(f"Detected nuclei: {result['num_nuclei']}")
        print(f"Outlines image: {result['outlines_image']}")
    print(f"History: {json.dumps(result['history'], indent=2)}")


if __name__ == "__main__":
    main()
