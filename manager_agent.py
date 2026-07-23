"""
Manager agent: Qwen3-VL-8B-Instruct (local, via Hugging Face transformers)
replaces Claude as the orchestrator across both specialist agents.

Given a task description + image, Qwen:
  1. picks which specialist agent to call -- CountGD (counting) or StarDist
     (nucleus segmentation) -- by reading the task and looking at the image.
  2. for CountGD, turns the task into a short count target (e.g. "count the
     cells" -> "cell"), same role as agentic_countgd.py's interpret_prompt.
  3. runs the retry loop itself, up to --max-iterations times.

Ground truth is NOT given to the manager. When a ground-truth count (BBBC005)
or instance mask (PanNuke) is supplied, it instead goes to a separate
ExpertReasoner persona -- the *same* loaded Qwen weights, reused under a
different system prompt, playing a domain expert (quantitative biologist /
digital pathologist) who privately holds the true answer plus some "extra
data" the manager never sees (BBBC005 focus/stain metadata parsed from the
image filename; for PanNuke, the tissue type and an outline rendering of the
real ground-truth instance boundaries). Each iteration, the manager gets up
to MAX_EXPERT_TURNS question/answer turns with the expert: it can ask about a
specific detection, cluster, or discrepancy it's unsure about, and the expert
answers in 2-4 sentences of morphological/domain rationale -- instructed to
never reveal the ground-truth number/mask or say accept/reject/correct/wrong
(see EXPERT_PERSONA_COUNTGD / EXPERT_PERSONA_STARDIST). The manager then
decides accept/reject itself from the dialogue transcript plus the image,
and if rejecting, proposes the revision itself (a revised count target for
CountGD, revised prob_thresh/nms_thresh for StarDist) -- mirroring
evaluate_result in agentic_countgd.py/agentic_stardist.py, but the judgment
now comes from Qwen reasoning through a dialogue rather than a hard MAE/PQ
threshold.

The old metric (MAE against the ground-truth count / Panoptic Quality against
the ground-truth mask) is still computed every iteration when ground truth is
available, but purely for the run's own history/logging -- as
"internal_mae"/"internal_pq" plus what the old threshold-based rule would
have decided ("internal_would_accept") -- so the new dialogue-driven
judgments can be compared against the old hard-threshold ones after the
fact. The manager's own prompts never include these numbers.

If no ground truth is supplied for the routed agent (a generic image with no
known answer), there's no expert to consult either, so Qwen falls back to
visually scoring the annotated/outlined image 0-10 itself -- the same
fallback either standalone script would face with an arbitrary --image and
no ground truth.

agentic_countgd.py and agentic_stardist.py are untouched -- this file only
imports their run_countgd/run_stardist/compute_panoptic_quality (and
StarDist's save_instance_outlines/load_pannuke_sample) and drives them with
Qwen instead of Claude.

StarDist's calls run in a spawned subprocess (StardistWorker, below) rather than
being imported directly into this process. Reason: `agentic_stardist.py` imports
`stardist.models`, which imports TensorFlow -- and TensorFlow bundles its own LLVM
(for XLA), which collides with ROCm/Triton's bundled LLVM the moment this process
also runs a GPU torch kernel (`LLVM ERROR: inconsistency in registered CommandLine
options`, a hard process abort). This only surfaces once torch actually touches a
GPU -- see the ROCm section below -- so it stayed invisible while this ran CPU-only.

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
from __future__ import annotations

import argparse
import json
import multiprocessing
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np # pyright: ignore[reportMissingImports]
from gradio_client import Client # pyright: ignore[reportMissingImports]
from PIL import Image # pyright: ignore[reportMissingImports]

from agentic_countgd import COUNTGD_SPACE, run_countgd

# agentic_cellvit is deliberately NOT imported here at module level, even though (unlike
# agentic_stardist) it's safe to import in *this* process -- see CLAUDE.md's empirical
# verification. The reason is StardistWorker: multiprocessing's spawn start method reimports
# this whole module fresh in the spawned subprocess to locate _stardist_worker_* functions, and
# that reimport would re-execute a module-level `import agentic_cellvit` too -- pulling torch
# into the same subprocess that then loads TensorFlow (via agentic_stardist), recreating the
# exact LLVM collision StardistWorker's isolation exists to avoid. (Confirmed the hard way:
# `_stardist_worker_load_pannuke` crashed with the same "CommandLine Error: Option ...
# registered more than once!" once this import was added at module level.) CellvitClient/
# run_cellvit_with_feedback import agentic_cellvit's functions locally instead, the same way
# StarDist's own worker functions import agentic_stardist locally.

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
ACCEPT_PQ_THRESHOLD = 0.5   # old PQ rule, kept only to log internal_would_accept -- see module docstring
MAE_TOLERANCE_FRACTION = 0.1  # old MAE rule, kept only to log internal_would_accept -- see module docstring
MAX_EXPERT_TURNS = 3  # cap on manager<->expert question/answer turns per iteration

# Ground-truth-free sanity check for StarDist: how many instances would survive at a much
# more permissive probability floor (same nms_thresh) vs. how many actually survived
# prob_thresh. A huge gap means detection is suspiciously sparse for this image regardless
# of what the 3-turn expert dialogue happened to probe -- it can't ask about nuclei it never
# had a reason to point at, so a manager relying only on the dialogue transcript is blind to
# this exact failure mode (observed: 10 kept vs a much larger candidate pool on a PanNuke
# image where true recall was 0.045 -- the dialogue's 3 questions all landed on already-
# correct regions and came back "no_issue", so nothing in the transcript flagged it).
STARDIST_CANDIDATE_FLOOR = 0.05
STARDIST_COVERAGE_RATIO_MIN = 0.6  # below this fraction of floor-candidates kept, treat as suspicious
# (raised from 0.4 on 2026-07-21: Kidney/Thyroid/Ovarian all cleared 0.4 while still missing
# 27-44% of real nuclei per internal_recall -- 0.4 wasn't tight enough to keep pushing on those)
STARDIST_COVERAGE_MIN_CANDIDATES = 5  # don't fire the guardrail on images with few candidates anyway

# The model's own tuned default (0.692, from thresholds.json) consistently under-detects on
# PanNuke: backing out true counts from internal_recall across a 2026-07-21 training run showed
# ~20-43% of real nuclei missed at 0.692 across many different tissue types (Colon, Lung, Kidney,
# Thyroid, Bladder, Pancreatic), not just isolated hard images -- so starting every trial at 0.692
# means every retry loop spends its budget clawing back a shortfall baked in from iteration 1.
# Starting lower gives iteration 1 a real shot at adequate coverage instead.
STARDIST_INITIAL_PROB_THRESH = 0.5


def mae_accept_tolerance(ground_truth_count: int) -> float:
    """Accept a CountGD result if its MAE is within this many counts of the ground truth --
    10% of the ground-truth count, with a floor of 1 so small counts aren't impossible to hit."""
    return max(1, round(MAE_TOLERANCE_FRACTION * ground_truth_count))


class QwenVLM:
    """Lazily loads Qwen3-VL and answers single image+text prompts with it.

    Deliberately NOT imported from agentic_cellvit.QwenVLM (which looks like the same class) --
    that one is intentionally trimmed to only the ask/ask_json methods agentic_cellvit.py itself
    needs (see its own docstring), missing ask_images/ask_json_multi/ask_text that ExpertReasoner/
    choose_best_output/train_manager.py all depend on here. Importing agentic_cellvit at this
    module's top level would also pull in torch, which the comment above (module-level import
    section) explains crashes StardistWorker's spawned subprocess -- confirmed the hard way."""

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
        return self.ask_images([image_path], prompt, max_new_tokens=max_new_tokens)

    def ask_images(self, image_paths: list, prompt: str, max_new_tokens: int = 512) -> str:
        """Like ask(), but takes multiple images in one turn -- used by ExpertReasoner to show
        the StarDist expert both the manager's predicted outlines and the private ground-truth
        outlines side by side so it can reason about specific discrepancies asked about."""
        from qwen_vl_utils import process_vision_info # pyright: ignore[reportMissingImports]
        model, processor = self._load()
        assert processor is not None, "Processor failed to load"

        messages = [{
            "role": "user",
            "content": [{"type": "image", "image": p} for p in image_paths] + [{"type": "text", "text": prompt}],
        }]
        chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[chat_text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
        ).to(model.device)

        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
        return processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()

    def ask_json(self, image_path: str, prompt: str, max_new_tokens: int = 512, required_keys: list[str] | None = None) -> dict:
        """Unlike the Claude calls in agentic_countgd.py/agentic_stardist.py, Qwen has no
        API-enforced JSON schema, so its free-text output can drop a requested key. Callers
        that will subscript the result (e.g. result["score"]) should pass required_keys so a
        malformed response fails here with a clear message instead of a bare KeyError deep in
        the caller."""
        raw = self.ask(image_path, prompt, max_new_tokens=max_new_tokens)
        return self._parse_json(raw, required_keys)

    def ask_json_multi(self, image_paths: list, prompt: str, max_new_tokens: int = 512, required_keys: list[str] | None = None) -> dict:
        """Multi-image counterpart to ask_json -- see ask_images for why a single turn can take
        more than one image."""
        raw = self.ask_images(image_paths, prompt, max_new_tokens=max_new_tokens)
        return self._parse_json(raw, required_keys)

    def _parse_json(self, raw: str, required_keys: list[str] | None = None) -> dict:
        start, end = raw.find("{"), raw.rfind("}")
        if start == -1 or end == -1:
            raise ValueError(f"Qwen response did not contain a JSON object: {raw!r}")
        result = json.loads(raw[start:end + 1])
        if required_keys:
            missing = [k for k in required_keys if k not in result]
            if missing:
                raise ValueError(f"Qwen JSON response is missing required key(s) {missing}: {result!r}")
        return result

    def ask_text(self, prompt: str, max_new_tokens: int = 512) -> str:
        """Text-only turn, no image -- for prompts that summarize patterns across multiple
        images/notes rather than looking at any one of them (see train_manager.py)."""
        model, processor = self._load()
        assert processor is not None, "Processor failed to load"
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[chat_text], padding=True, return_tensors="pt").to(model.device)

        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
        return processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()


def select_agent(qwen: QwenVLM, task_description: str, image_path: str) -> str:
    # stardist vs. cellvit is the pair most likely to get routing-confused (both operate on
    # nuclei/tissue images) -- worth spot-checking a few ambiguous prompts (e.g. "count the
    # nuclei in this tissue", with no type named, should land on stardist not cellvit).
    prompt = (
        "You are a manager agent that routes an image-analysis task to one of four "
        "specialist tools:\n"
        "  - countgd: counts individual objects/cells matching a free-text description. Use it "
        "for \"how many of something\" tasks where the object type doesn't need pathology-"
        "specific typing.\n"
        "  - stardist: segments every cell nucleus into instance masks with NO typing of what "
        "kind of nucleus each one is -- a purely generic count/outline of nuclei as "
        "undifferentiated objects. Use it when the task only cares about outlining/segmenting/"
        "delineating/counting nuclei in general, not classifying them.\n"
        "  - cellvit: classifies every nucleus into one of five fixed pathology types "
        "(Neoplastic, Inflammatory, Connective, Dead, Epithelial) and can count/highlight one or "
        "more specific types. Use it whenever the task names or implies a SPECIFIC cell/nucleus "
        "type (e.g. \"how many inflammatory cells\", \"classify the tumor cells\", \"find the "
        "dead cells\", \"break down the nuclei by type\") -- anything requiring pathology-"
        "relevant cell-type identification, not just a generic nucleus count.\n"
        "  - deepgleason: grades prostate tumor severity (Gleason score / ISUP grade group) from "
        "a whole-slide pathology image. Use it for tumor grading/staging tasks (e.g. \"what's the "
        "Gleason score\", \"grade this biopsy\", \"how severe is this tumor\") -- distinct from "
        "cell counting/segmentation/classification, this is a slide-level clinical severity "
        "assessment, not a per-cell/per-nucleus task.\n\n"
        f"Task: \"{task_description}\"\n\n"
        "Reply with ONLY a JSON object: {\"agent\": \"countgd\" or \"stardist\" or \"cellvit\" or "
        "\"deepgleason\", \"reason\": \"one sentence\"}"
    )
    result = qwen.ask_json(image_path, prompt)
    agent = result.get("agent", "").strip().lower()
    if agent not in ("countgd", "stardist", "cellvit", "deepgleason"):
        raise ValueError(f"Qwen returned an unrecognized agent choice: {result!r}")
    return agent


def interpret_countgd_target(qwen: QwenVLM, task_description: str, image_path: str) -> str:
    prompt = (
        f"The user wants to count objects in this image. Their request: \"{task_description}\"\n\n"
        "Reply with ONLY a short noun phrase (1-3 words) naming the single object type to "
        "count (e.g. 'cell', 'car', 'strawberry'). No punctuation, no explanation, nothing else."
    )
    return qwen.ask(image_path, prompt).strip().strip('."\'')


_GROUND_TRUTH_HISTORY_KEYS = {
    "pq", "mean_iou", "tp", "fp", "fn", "internal_would_accept", "internal_recall", "internal_mae",
    "internal_mpq", "internal_f1", "per_class_scores", "internal_isup_mae",
}


def _redact_history_for_manager(history: list) -> list:
    """decide_countgd_from_dialogue/decide_stardist_from_dialogue's docstrings both say ground
    truth is never given to the manager -- but history entries (built in run_*_with_feedback)
    also carry internal_mae/internal_pq/internal_recall/tp/fp/fn purely for our own
    history/logging comparison (see module docstring), and until this fix that whole dict was
    getting serialized straight into the manager's own prompt via json.dumps(history), leaking
    real ground-truth-derived numbers for every prior iteration of the same image. This strips
    exactly those keys, keeping only what the manager could legitimately have (iteration,
    thresholds/count_target, predicted_count, dialogue, accept, feedback)."""
    return [{k: v for k, v in entry.items() if k not in _GROUND_TRUTH_HISTORY_KEYS} for entry in history]


def decide_countgd_from_dialogue(
    qwen: QwenVLM, task_description: str, count_target: str, predicted_count: int,
    dialogue: list, annotated_image_path: str, history: list,
) -> dict:
    """Ground truth is never given to the manager (see module docstring) -- instead the manager
    has just talked to ExpertReasoner (run_expert_dialogue) and decides accept/reject itself from
    that transcript plus the image, proposing a revised count target if rejecting."""
    prompt = (
        f"Original user request: \"{task_description}\"\n"
        f"CountGD was asked to count: \"{count_target}\"\n"
        f"Predicted count: {predicted_count}\n"
        f"Your conversation with the domain expert this iteration:\n{json.dumps(dialogue, default=str)}\n"
        f"Prior attempts this session: {json.dumps(_redact_history_for_manager(history), default=str)}\n"
        "Compare this attempt against those prior ones: did predicted_count and the dialogue's "
        "findings actually improve, or did the last change not help (or make things worse)? "
        "Don't assume a later attempt is better just because it came later -- check whether the "
        "specific concerns raised in past feedback were actually resolved this time.\n\n"
        "Based on the expert's reasoning and what you can see in the image yourself, decide "
        "whether this result satisfies the request. If not, propose a different/more specific "
        "text prompt for CountGD to retry with, informed by what the expert pointed out (e.g. "
        "overlapping cells, faint/out-of-focus cells, background clutter).\n\n"
        "Reply with ONLY a JSON object matching this schema: "
        "{\"accept\": bool, \"feedback\": str (1-2 sentences), \"revised_text\": str or null}"
    )
    return qwen.ask_json(annotated_image_path, prompt, max_new_tokens=768, required_keys=["accept", "feedback"])


def decide_stardist_from_dialogue(
    qwen: QwenVLM, task_description: str, prob_thresh: float, nms_thresh: float,
    predicted_count: int, dialogue: list, outlines_image_path: str, history: list,
) -> dict:
    """Ground truth is never given to the manager (see module docstring) -- instead the manager
    has just talked to ExpertReasoner (run_expert_dialogue) and decides accept/reject itself from
    that transcript plus the image, proposing revised thresholds if rejecting."""
    prompt = (
        f"Original user request: \"{task_description}\"\n"
        f"StarDist ran with prob_thresh={prob_thresh:.3f}, nms_thresh={nms_thresh:.3f}\n"
        f"Detected nuclei: {predicted_count}\n"
        f"Your conversation with the domain expert this iteration:\n{json.dumps(dialogue, default=str)}\n"
        f"Prior attempts this session: {json.dumps(_redact_history_for_manager(history), default=str)}\n"
        "Compare this attempt against those prior ones: did predicted_count and the dialogue's "
        "findings actually improve, or did the last threshold change not help (or make things "
        "worse)? Don't assume a later attempt is better just because it came later -- check "
        "whether the specific concerns raised in past feedback were actually resolved this time.\n\n"
        "First, classify each numbered turn in the conversation above as exactly one of: "
        "missed_object (expert confirmed a real object was missed), false_positive (expert "
        "suggested an outlined/detected region isn't actually a real nucleus), no_issue (expert "
        "confirmed the existing segmentation was correct there), or ambiguous. List these as "
        "turn_analysis.\n\n"
        "Then decide whether this segmentation satisfies the request. Coverage does not need to "
        "be perfect -- a clear majority of real nuclei captured with reasonably accurate outlines "
        "is acceptable, even if the dialogue turned up one isolated missed spot or minor outline "
        "issue. Use your turn_analysis tally as the basis: only reject on coverage grounds if "
        "missed_object verdicts make up the majority of the turns (i.e. most of what the dialogue "
        "surfaced was missed real objects, not just one one-off), or reject on accuracy grounds if "
        "outlines are clearly poor (frequent false positives, merges, or splits). Don't reject "
        "solely because of a single ambiguous or missed spot when everything else checks out.\n\n"
        "If not accepted, base your threshold direction on the turn_analysis tally, not a guess: "
        "if missed_object turns outnumber false_positive turns, LOWER prob_thresh (more permissive "
        "-- catches more candidates); if false_positive turns outnumber missed_object turns, RAISE "
        "prob_thresh (stricter -- drops spurious detections); lower nms_thresh if outlines sound "
        "duplicated/split around one nucleus, raise it if adjacent distinct nuclei sound merged. "
        "You MUST set at least one of revised_prob_thresh/revised_nms_thresh to a number that "
        "actually differs from its current value above -- never leave both null, and never repeat "
        "the current value, when accept is false.\n\n"
        "Reply with ONLY a JSON object matching this schema: {\"turn_analysis\": "
        "[{\"turn\": int, \"verdict\": \"missed_object\"|\"false_positive\"|\"no_issue\"|\"ambiguous\"}], "
        "\"accept\": bool, \"feedback\": str (1-2 sentences), "
        "\"revised_prob_thresh\": number or null, \"revised_nms_thresh\": number or null}"
    )
    return qwen.ask_json(outlines_image_path, prompt, max_new_tokens=900, required_keys=["accept", "feedback"])


def decide_cellvit_from_dialogue(
    qwen: QwenVLM, task_description: str, target_classes: list, prob_threshold: float,
    predicted_count: int, counts_by_type: dict, dialogue: list, annotated_image_path: str, history: list,
) -> dict:
    """Ground truth is never given to the manager (see module docstring) -- instead the manager
    has just talked to ExpertReasoner (run_expert_dialogue) and decides accept/reject itself from
    that transcript plus the image, proposing a revised class selection/threshold if rejecting.
    Unlike StarDist's prob_thresh/nms_thresh (which can genuinely fix missed/merged/split
    detections), target_classes/prob_threshold here can only fix recall/scope issues (catching
    more or fewer already-detected cells) -- there is no tunable lever for outright
    misclassification (the model calling an Epithelial cell Neoplastic). The prompt says this
    explicitly so the manager doesn't hallucinate a fix that doesn't exist."""
    prompt = (
        f"Original user request: \"{task_description}\"\n"
        f"CellViT was asked to highlight: {target_classes} (type_prob >= {prob_threshold})\n"
        f"Matched cell count: {predicted_count}\n"
        f"All detected cells by type: {json.dumps(counts_by_type)}\n"
        f"Your conversation with the domain expert this iteration:\n{json.dumps(dialogue, default=str)}\n"
        f"Prior attempts this session: {json.dumps(_redact_history_for_manager(history), default=str)}\n"
        "Compare this attempt against those prior ones: did the dialogue's findings actually "
        "improve, or did the last class-selection/threshold change not help (or make things "
        "worse)? Don't assume a later attempt is better just because it came later.\n\n"
        "First, classify each numbered turn in the conversation above as exactly one of: "
        "missed_cell (expert confirmed a real cell of a target type was missed), "
        "misclassified (expert suggested a highlighted cell's type looks wrong), no_issue "
        "(expert confirmed the existing classification was correct there), or ambiguous. List "
        "these as turn_analysis.\n\n"
        "Then decide whether this result satisfies the request. Note: target_classes/"
        "prob_threshold can only fix recall/scope problems (catching more or fewer cells "
        "CellViT already detected) -- they CANNOT fix a cell being assigned the wrong type; "
        "there is no adjustable lever for that here. So: if most flagged turns are "
        "missed_cell, propose a lower prob_threshold and/or broader target_classes. If most "
        "flagged turns are misclassified, do NOT propose a class/threshold change to address "
        "it (there is nothing to tune) -- either accept anyway if the overall result still "
        "satisfies the request, or reject with feedback noting this is a classification "
        "limitation, leaving both revised fields null. Don't reject solely because of a single "
        "ambiguous or missed spot when everything else checks out.\n\n"
        "Reply with ONLY a JSON object matching this schema: {\"turn_analysis\": "
        "[{\"turn\": int, \"verdict\": \"missed_cell\"|\"misclassified\"|\"no_issue\"|\"ambiguous\"}], "
        "\"accept\": bool, \"feedback\": str (1-2 sentences), "
        "\"revised_target_classes\": array or null, \"revised_prob_threshold\": number or null}"
    )
    return qwen.ask_json(annotated_image_path, prompt, max_new_tokens=900, required_keys=["accept", "feedback"])


def decide_deepgleason_from_dialogue(
    qwen: QwenVLM, task_description: str, confidence_threshold: float, gleason_result: dict,
    dialogue: list, overlay_image_path: str, history: list,
) -> dict:
    """Ground truth is never given to the manager (see module docstring) -- instead the manager
    has just talked to ExpertReasoner (run_expert_dialogue) and decides accept/reject itself from
    that transcript plus the overlay image, proposing a revised confidence_threshold if
    rejecting. confidence_threshold is DeepGleason's one tunable lever here (see
    DeepGleasonClient/run_deepgleason_with_feedback): raising it excludes more low-confidence
    tiles from Gleason-pattern tallying (fixes spurious/uncertain tiles being counted), lowering
    it includes more borderline tiles (fixes a genuine tumor pattern being excluded as
    Uncertain). Like CellViT's target_classes/prob_threshold, it can only fix which tiles count
    toward the grade -- it cannot fix a confidently-wrong per-tile classification; there is no
    tunable lever for that here, so the prompt says so explicitly."""
    prompt = (
        f"Original user request: \"{task_description}\"\n"
        f"DeepGleason ran with confidence_threshold={confidence_threshold:.2f}\n"
        f"Result: {json.dumps(gleason_result, default=str)}\n"
        f"Your conversation with the domain expert this iteration:\n{json.dumps(dialogue, default=str)}\n"
        f"Prior attempts this session: {json.dumps(_redact_history_for_manager(history), default=str)}\n"
        "Compare this attempt against those prior ones: did the dialogue's findings actually "
        "improve, or did the last threshold change not help (or make things worse)? Don't "
        "assume a later attempt is better just because it came later.\n\n"
        "First, classify each numbered turn in the conversation above as exactly one of: "
        "excluded_pattern (expert confirmed a region genuinely shows a Gleason growth pattern "
        "that's being excluded from the grade as low-confidence/Uncertain), spurious_included "
        "(expert suggested a region counted toward the grade doesn't actually look like a "
        "confident tumor pattern), no_issue (expert confirmed the region's classification and "
        "inclusion/exclusion was correct), or ambiguous. List these as turn_analysis.\n\n"
        "Then decide whether this result satisfies the request. Note: confidence_threshold can "
        "only fix which tiles count toward the grade (excluding uncertain ones, or including "
        "borderline ones) -- it CANNOT fix a tile being confidently misclassified as the wrong "
        "growth pattern; there is no adjustable lever for that here. So: if most flagged turns "
        "are excluded_pattern, propose a LOWER confidence_threshold (includes more borderline "
        "tiles). If most flagged turns are spurious_included, propose a HIGHER "
        "confidence_threshold (excludes more uncertain tiles). Don't reject solely because of a "
        "single ambiguous turn when everything else checks out.\n\n"
        "Reply with ONLY a JSON object matching this schema: {\"turn_analysis\": "
        "[{\"turn\": int, \"verdict\": \"excluded_pattern\"|\"spurious_included\"|\"no_issue\"|\"ambiguous\"}], "
        "\"accept\": bool, \"feedback\": str (1-2 sentences), \"revised_confidence_threshold\": number or null}"
    )
    return qwen.ask_json(overlay_image_path, prompt, max_new_tokens=900, required_keys=["accept", "feedback"])


EXPERT_PERSONA_COUNTGD = (
    "You are a senior quantitative image-analysis scientist who manually verified the true "
    "object count in this image as part of a ground-truth annotation study. You are acting as "
    "a mentor answering a junior colleague's question about a specific automated detection run "
    "-- you are NOT a scorer, and your job is NOT to simply hand them the answer. Never state "
    "the true total count as a number, never say whether the overall result is 'correct', "
    "'accepted', 'right', or 'wrong', never use the words 'accept'/'reject', and do NOT declare "
    "a verdict yourself (do not say things like 'that's two cells' or 'that's a single cell').\n\n"
    "Instead, for the SPECIFIC region/detection asked about, point out the concrete, checkable "
    "visual evidence that IS or ISN'T actually present there -- grounded in the true annotation "
    "you're privately holding for that area -- so your colleague can weigh it and reach their "
    "own conclusion. Good evidence is specific to this exact spot, e.g. 'there's a faint but "
    "continuous dark gap running through the middle of that shape' or 'the brightness along that "
    "edge is uniform with no dip, unlike the clearer separations elsewhere in this image' or "
    "'the intensity there never drops back to background level between the two lobes.' Do not "
    "describe what an annotator 'would' generally look for in the abstract, and do not simply "
    "state your conclusion -- describe the actual evidence present in this specific region, in "
    "2-4 sentences."
)

EXPERT_PERSONA_STARDIST = (
    "You are a senior digital pathologist who manually annotated the ground-truth nucleus "
    "boundaries in this tissue image. You are acting as a mentor answering a junior colleague's "
    "question about a specific automated segmentation run -- you are NOT a scorer, and your job "
    "is NOT to simply hand them the answer. Never state the true total nucleus count as a "
    "number, never say whether the overall result is 'correct', 'accepted', 'right', or "
    "'wrong', never use the words 'accept'/'reject', and do NOT declare a verdict yourself (do "
    "not say things like 'that's one nucleus' or 'those are two nuclei').\n\n"
    "Instead, for the SPECIFIC nucleus/region asked about, point out the concrete, checkable "
    "morphological evidence that IS or ISN'T actually present there -- grounded in the real "
    "instance boundaries shown to you (not your colleague) in the second attached image -- so "
    "your colleague can weigh it and reach their own conclusion. Good evidence is specific to "
    "this exact spot, e.g. 'the nuclear envelope stays continuous all the way around that "
    "outline, with no pinch point' or 'chromatin texture shifts partway across that region, "
    "unlike the uniform texture in the unambiguous nuclei nearby' or 'that outline's edge sits "
    "outside where the staining intensity actually drops off.' Do not describe what a "
    "pathologist 'would' generally look for in the abstract, and do not simply state your "
    "conclusion -- describe the actual evidence present in this specific region, in 2-4 "
    "sentences."
)


EXPERT_PERSONA_CELLVIT = (
    "You are a senior digital pathologist who manually classified every nucleus's cell type "
    "(Neoplastic, Inflammatory, Connective, Dead, or Epithelial) in this tissue image as part of "
    "a ground-truth annotation study. You are acting as a mentor answering a junior colleague's "
    "question about a specific automated classification run -- you are NOT a scorer, and your "
    "job is NOT to simply hand them the answer. Never state the true per-class counts as "
    "numbers, never say whether the overall result is 'correct', 'accepted', 'right', or "
    "'wrong', never use the words 'accept'/'reject', and do NOT declare a verdict yourself (do "
    "not say things like 'that's an inflammatory cell').\n\n"
    "Instead, for the SPECIFIC nucleus/region asked about, point out the concrete, checkable "
    "morphological evidence that IS or ISN'T actually present there -- grounded in the true "
    "classification you're privately holding for that cell -- so your colleague can weigh it "
    "and reach their own conclusion. Good evidence is specific to this exact spot, e.g. 'the "
    "nucleus-to-cytoplasm ratio there is much higher than the regular cells nearby, with visibly "
    "clumped chromatin' or 'that cell's outline is elongated and spindle-shaped, consistent with "
    "stromal tissue rather than a rounded immune cell' or 'the chromatin there looks condensed "
    "and fragmented rather than the smooth, evenly-stained nucleus of a healthy cell.' Do not "
    "describe what a pathologist 'would' generally look for in the abstract, and do not simply "
    "state your conclusion -- describe the actual evidence present in this specific region, in "
    "2-4 sentences."
)

EXPERT_PERSONA_DEEPGLEASON = (
    "You are a senior genitourinary pathologist who manually graded this prostate biopsy's true "
    "Gleason score and ISUP grade group as part of a ground-truth annotation study. You are "
    "acting as a mentor answering a junior colleague's question about a specific automated "
    "grading run -- you are NOT a scorer, and your job is NOT to simply hand them the answer. "
    "Never state the true Gleason score or ISUP grade group as a number, never say whether the "
    "overall result is 'correct', 'accepted', 'right', or 'wrong', never use the words "
    "'accept'/'reject', and do NOT declare a verdict yourself (do not say things like 'that's "
    "Gleason pattern 4').\n\n"
    "Instead, for the SPECIFIC region asked about, point out the concrete, checkable "
    "morphological evidence that IS or ISN'T actually present there -- grounded in the true "
    "grading you're privately holding for that region -- so your colleague can weigh it and "
    "reach their own conclusion. Good evidence is specific to this exact spot, e.g. 'those "
    "glands there have lost their well-formed lumina and are fusing into cribriform sheets, "
    "which is a higher-grade pattern than the discrete round glands elsewhere' or 'that region "
    "still shows individually separate, round-to-oval glands with clear lumina, consistent with "
    "a lower-grade pattern rather than the poorly-formed glands nearby' or 'there's no "
    "infiltrative single-cell or cord-like growth in that area, unlike the clearly infiltrative "
    "pattern in the adjacent focus.' Do not describe what a pathologist 'would' generally look "
    "for in the abstract, and do not simply state your conclusion -- describe the actual "
    "evidence present in this specific region, in 2-4 sentences."
)


def _parse_bbbc005_metadata(image_path: str) -> dict | None:
    """BBBC005 filenames encode focus level and stain channel, e.g.
    SIMCEPImages_A02_C5_F1_s01_w1.TIF -> focus F1 (in-focus), stain w1. Used as the CountGD
    expert's "extra data" -- returns None for filenames that don't match (e.g. an arbitrary
    --image with no BBBC005 naming convention)."""
    match = re.search(r"_F(\d+)_s\d+_w(\d+)", Path(image_path).stem)
    if not match:
        return None
    focus, stain = match.groups()
    return {"focus_level": f"F{focus}", "stain_channel": f"w{stain}"}


# Used by run_countgd_with_feedback/run_stardist_with_feedback/run_cellvit_with_feedback (and
# reused as-is by resolve_escalations.py) whenever there's no real ground truth to build a
# dossier from -- the manager still gets a full expert dialogue either way (see module
# docstring), just with the expert told honestly that it has no private answer to check
# against, so it reasons from what's actually visible in the image rather than fabricating a
# verdict dressed up as a "private fact."
NO_GROUND_TRUTH_DOSSIER = (
    "[PRIVATE] No verified ground truth is available for this case. Reason from general domain "
    "knowledge and what's visible in the image only -- you do not have a private answer to check "
    "against here."
)


def _apply_expert_notes(dossier: str, expert_notes: str) -> str:
    """Prepends persistent, accumulated expert notes (see merge_expert_notes in
    train_manager.py, and resolve_escalations.py's own use of it) to a case's dossier -- so the
    private domain-expert persona reasons consistently across images/escalations within a run,
    informed by what past human-resolved escalations taught it, not just this one case's private
    facts. This is the expert-side counterpart to the manager's running_prompt (checkpoint.json)
    -- same idea, different consumer. Empty expert_notes (the common case: a run that hasn't
    resolved any escalations yet, or a live manager_agent.py CLI call with no checkpoint at all)
    leaves the dossier unchanged."""
    if not expert_notes:
        return dossier
    return f"[PRIVATE -- accumulated notes from prior escalations this run]\n{expert_notes}\n\n{dossier}"


def build_countgd_dossier(ground_truth_count: int, image_path: str) -> str:
    lines = [f"[PRIVATE -- never reveal] Verified true object count: {ground_truth_count}."]
    metadata = _parse_bbbc005_metadata(image_path)
    if metadata:
        lines.append(
            f"[PRIVATE -- never reveal the number, general knowledge OK] Acquisition metadata: "
            f"focus level {metadata['focus_level']}, stain channel {metadata['stain_channel']} "
            "(BBBC005 synthetic dataset)."
        )
    return "\n".join(lines)


def build_stardist_dossier(ground_truth_labels: np.ndarray, tissue: str | None = None) -> str:
    lines = [
        f"[PRIVATE -- never reveal] Verified true nucleus count: {int(ground_truth_labels.max())}. "
        "Their exact boundaries are shown to you (not your colleague) in the second attached image."
    ]
    if tissue:
        lines.append(f"[PRIVATE -- general tissue knowledge OK to reference] Tissue type: {tissue}.")
    return "\n".join(lines)


def build_cellvit_dossier(ground_truth_counts_by_type: dict, tissue: str | None = None) -> str:
    lines = [
        "[PRIVATE -- never reveal] Verified true per-class nucleus counts: "
        + ", ".join(f"{k}={v}" for k, v in ground_truth_counts_by_type.items()) + "."
    ]
    if tissue:
        lines.append(f"[PRIVATE -- general tissue knowledge OK to reference] Tissue type: {tissue}.")
    return "\n".join(lines)


def build_deepgleason_dossier(ground_truth_gleason_score: str | None, ground_truth_isup_grade: int | None) -> str:
    lines = ["[PRIVATE -- never reveal] Verified true grading:"]
    if ground_truth_gleason_score is not None:
        lines.append(f"Gleason score {ground_truth_gleason_score}.")
    if ground_truth_isup_grade is not None:
        lines.append(f"ISUP grade group {ground_truth_isup_grade}.")
    return " ".join(lines)


EXPERT_LEAK_FALLBACK = (
    "I can't get into specifics there -- think about it in terms of the visual/morphological "
    "reasoning rather than any exact number."
)
EXPERT_LEAK_MAX_RETRIES = 2  # after this many re-prompts, EXPERT_LEAK_FALLBACK is returned instead -- see _leaks_forbidden_value


def _spelled_out_forms(n: int) -> list:
    """Likely spelled-out English forms of a non-negative int (hyphenated, spaced, and "and"
    variants for hundreds, e.g. "one hundred and five" / "one hundred five"), so the leak check
    catches "twenty-three" as well as "23". Only covers 0-999 -- realistic counts in this repo
    top out in the hundreds (BBBC005 up to 100, PanNuke nucleus counts a few hundred); values
    outside that range just rely on the digit check in _leaks_forbidden_value."""
    if not (0 <= n < 1000):
        return []
    ones = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
            "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
    tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]

    def under_100(x: int) -> str:
        if x < 20:
            return ones[x]
        t, r = divmod(x, 10)
        return tens[t] + (f"-{ones[r]}" if r else "")

    if n < 100:
        forms = {under_100(n)}
    else:
        hundreds, rest = divmod(n, 100)
        base = f"{ones[hundreds]} hundred"
        forms = {base} if rest == 0 else {f"{base} {under_100(rest)}", f"{base} and {under_100(rest)}"}

    forms |= {f.replace("-", " ") for f in forms}
    return sorted(forms)


class ExpertReasoner:
    """Holds ground truth privately and answers the manager's questions with morphological/
    domain rationale -- never the ground-truth number/mask itself and never an accept/reject
    verdict (see EXPERT_PERSONA_COUNTGD/EXPERT_PERSONA_STARDIST). Reuses the manager's own
    already-loaded QwenVLM under a different persona/system framing rather than loading a
    second 8B model.

    The "never reveal the number" instruction in EXPERT_PERSONA_* is only a prompt -- Qwen is an
    8B model and can ignore it. answer() backs that instruction with a hard code-level check:
    forbidden_values are the exact ground-truth number(s), checked both as digits ("23") and as
    spelled-out English (_spelled_out_forms -- "twenty-three", "twenty three"), and any response
    containing one as a standalone token is never returned to the caller. It's re-prompted up to
    EXPERT_LEAK_MAX_RETRIES times, and if it still leaks, answer() returns the fixed
    EXPERT_LEAK_FALLBACK string instead -- so the manager can never see the true number no matter
    what the model does, at the cost of an occasional false-positive-triggered generic answer
    (e.g. a coincidental digit match unrelated to the ground truth)."""

    def __init__(self, qwen: QwenVLM, persona: str, dossier: str, forbidden_values: list):
        self.qwen = qwen
        self.persona = persona
        self.dossier = dossier
        self._leak_patterns = []
        for value in forbidden_values:
            self._leak_patterns.append(str(value))
            if isinstance(value, int):
                self._leak_patterns.extend(_spelled_out_forms(value))

    def _leaks_forbidden_value(self, text: str) -> bool:
        return any(re.search(rf"\b{re.escape(p)}\b", text, re.IGNORECASE) for p in self._leak_patterns)

    def _build_prompt(self, question: str, dialogue_so_far: list, retry_notice: str = "") -> str:
        return (
            f"{self.persona}\n\n{self.dossier}\n\n"
            f"Conversation so far: {json.dumps(dialogue_so_far, default=str)}\n\n"
            f"Colleague's question: \"{question}\"\n\n"
            f"{retry_notice}"
            "Respond with your reasoning only, in 2-4 sentences. No JSON, no verdict, no "
            "numbers from the private facts above."
        )

    def answer(self, image_paths: list, question: str, dialogue_so_far: list) -> str:
        prompt = self._build_prompt(question, dialogue_so_far)
        response = self.qwen.ask_images(image_paths, prompt, max_new_tokens=256).strip()
        for _ in range(EXPERT_LEAK_MAX_RETRIES):
            if not self._leaks_forbidden_value(response):
                return response
            retry_notice = (
                f"Your previous answer stated a forbidden private number (\"{response}\"). "
                "Answer again, describing the reasoning in words only -- no digits from the "
                "private facts above.\n\n"
            )
            prompt = self._build_prompt(question, dialogue_so_far, retry_notice)
            response = self.qwen.ask_images(image_paths, prompt, max_new_tokens=256).strip()
        if self._leaks_forbidden_value(response):
            print(f"    [expert] leaked the private ground-truth value after {EXPERT_LEAK_MAX_RETRIES} retries -- returning fallback instead")
            return EXPERT_LEAK_FALLBACK
        return response

    def ask_human_question(self, image_paths: list, task_description: str, dialogue_so_far: list) -> str:
        """Counterpart to manager_ask_expert, direction reversed: there the manager (no ground
        truth) asks the expert something; here the expert (holding ground truth privately) is the
        one probing, of a human reviewer, about a specific region/detection it wants their read
        on before it can give useful feedback -- e.g. "does the cluster in the top-left look like
        one nucleus or two to you?" Same leak-check discipline as answer(): the question itself
        must never state the private ground-truth number."""
        prompt = (
            f"{self.persona}\n\n{self.dossier}\n\n"
            f"Task: \"{task_description}\"\n"
            f"This case was escalated to a human reviewer because the automated retry loop never "
            f"reached an acceptable result on its own.\n"
            f"Conversation so far: {json.dumps(dialogue_so_far, default=str)}\n\n"
            "Ask the human ONE focused question about a specific region, detection, or possible "
            "discrepancy you want their visual judgment on -- something that would help you "
            "explain what's wrong once you have their read on it. Do not ask for a count, a "
            "verdict, or state the private ground-truth number/mask above. Reply with only the "
            "question, no preamble."
        )
        response = self.qwen.ask_images(image_paths, prompt, max_new_tokens=150).strip()
        for _ in range(EXPERT_LEAK_MAX_RETRIES):
            if not self._leaks_forbidden_value(response):
                return response
            retry_notice = (
                f"Your previous question stated a forbidden private number (\"{response}\"). "
                "Ask again in words only -- no digits from the private facts above.\n\n"
            )
            response = self.qwen.ask_images(image_paths, prompt + f"\n\n{retry_notice}", max_new_tokens=150).strip()
        if self._leaks_forbidden_value(response):
            print("    [expert] leaked the private ground-truth value while forming a question -- returning fallback instead")
            return "Looking at the final result -- is there a region where the detections look wrong to you?"
        return response

    def summarize_for_manager(self, task_description: str, image_id: str, conversation: list) -> str:
        """After a human reviewer talks through what's wrong with a result the automated loop
        never got accepted, this turns that conversation into transferable tuning guidance for
        the manager -- same leak-check discipline as answer() (the manager must never see the
        ground-truth number/mask, only reasoning), applied here since summarizing a conversation
        could in principle recombine details into the forbidden value even though each individual
        turn already passed the same check."""
        prompt = (
            f"A human reviewer looked at the final {task_description} result for image "
            f"{image_id!r} -- this was escalated because the automated retry loop never reached "
            f"an acceptable result on its own -- and had this conversation with you:\n"
            f"{json.dumps(conversation, default=str)}\n\n"
            "Summarize in 2-4 sentences, for the manager (who will read this as tuning guidance "
            "for future images and never sees the ground truth): what specifically was wrong, and "
            "what adjustment approach would address it. Write it as general, transferable "
            "guidance about image/tissue characteristics and threshold direction, not "
            "image-specific trivia that won't generalize. No numbers from the private facts above."
        )
        response = self.qwen.ask_text(prompt, max_new_tokens=300).strip()
        for _ in range(EXPERT_LEAK_MAX_RETRIES):
            if not self._leaks_forbidden_value(response):
                return response
            response = self.qwen.ask_text(
                prompt + f"\n\nYour previous summary stated a forbidden private number "
                         f"(\"{response}\"). Try again, describing it in words only.",
                max_new_tokens=300,
            ).strip()
        if self._leaks_forbidden_value(response):
            print("    [expert] leaked the private ground-truth value while summarizing for the manager -- returning fallback instead")
            return EXPERT_LEAK_FALLBACK
        return response


def manager_ask_expert(
    qwen: QwenVLM, task_description: str, predicted_summary: str, image_path: str,
    dialogue_so_far: list, turns_left: int,
) -> dict:
    prompt = (
        "You are a manager agent reviewing a specialist model's result, shown below. You do "
        "NOT have the ground-truth answer -- a domain expert does, but they will only explain "
        "their reasoning about specific things you ask, never give you a number or a verdict. "
        "Look at the image and ask ONE focused question about a specific region, detection, or "
        "possible discrepancy you want the expert's judgment on. Consider BOTH kinds of mistake, "
        "not just one: (a) an existing outline might be wrong -- e.g. whether a cluster looks "
        "like one object or several, whether two adjacent outlines should be merged; and (b) the "
        "specialist might have missed something entirely -- scan the image itself (not just the "
        "outlines) for structures that look like they could be the object of interest but have no "
        "outline around them at all, and ask the expert about those specific unmarked spots too. "
        "A result with a few wrong outlines and a result that's missing a large fraction of the "
        "objects are both failures worth catching. Do not ask for the count or a yes/no verdict; "
        "ask about a specific visual thing.\n\n"
        f"Task: \"{task_description}\"\n"
        f"Specialist's result: {predicted_summary}\n"
        f"Conversation with the expert so far: {json.dumps(dialogue_so_far, default=str)}\n"
        f"You have {turns_left} question(s) left.\n\n"
        "Reply with ONLY a JSON object: {\"done\": bool, \"question\": str or null} -- set "
        "done=true and question=null once you've asked enough to form your own judgment."
    )
    return qwen.ask_json(image_path, prompt, required_keys=["done"])


def run_expert_dialogue(
    qwen: QwenVLM, expert: ExpertReasoner, task_description: str, predicted_summary: str,
    manager_image_path: str, expert_image_paths: list, max_turns: int = MAX_EXPERT_TURNS,
) -> list:
    """Drives up to max_turns question/answer turns between the manager (which only sees
    manager_image_path -- this iteration's prediction, no ground truth) and the expert (which
    only sees expert_image_paths -- may include a private ground-truth rendering). The manager
    can stop early by setting done=true; it is never forced to spend all max_turns."""
    dialogue = []
    for turn in range(1, max_turns + 1):
        ask_result = manager_ask_expert(
            qwen, task_description, predicted_summary, manager_image_path, dialogue, max_turns - turn + 1
        )
        if ask_result.get("done") or not ask_result.get("question"):
            break
        question = ask_result["question"]
        answer = expert.answer(expert_image_paths, question, dialogue)
        print(f"    [manager -> expert] {question}")
        print(f"    [expert]  {answer}")
        dialogue.append({"question": question, "answer": answer})
    return dialogue


def run_human_expert_dialogue(
    expert: ExpertReasoner, expert_image_paths: list, task_description: str, get_human_input,
    max_turns: int = MAX_EXPERT_TURNS,
) -> list:
    """Human-driven counterpart to run_expert_dialogue, used to resolve an escalated case
    (write_escalation). Direction reversed from an earlier version of this function: rather than
    dumping a canned summary of the last automated attempt and waiting for the human to volunteer
    feedback into a void, the expert (holding ground truth privately) leads every turn by asking
    the human a specific, case-relevant question (ExpertReasoner.ask_human_question) -- the same
    role the manager plays with the expert during training, just with the human standing in for
    the manager. Ends early if the human answers with an empty response, or after max_turns
    questions. get_human_input is a callable (prompt: str) -> str -- takes a prompt, returns the
    human's typed response -- injected rather than calling input() directly here so this stays
    testable/reusable outside a real terminal."""
    dialogue = []
    for _ in range(max_turns):
        question = expert.ask_human_question(expert_image_paths, task_description, dialogue)
        human_answer = get_human_input(f"[expert asks] {question}\n> ")
        if not human_answer.strip():
            break
        dialogue.append({"question": question, "answer": human_answer})
    return dialogue


def choose_best_output(qwen: QwenVLM, task_description: str, candidates: list) -> dict:
    """Picks which of several attempted outputs to actually return, using the manager's own
    judgment across all of them at once instead of always defaulting to "whichever one happened
    to get accept=True" or "whichever one the last iteration produced." Previously the exhausted-
    without-accept path (run_stardist_with_feedback) fell back to worker.revert_to_best, which
    picks by the hidden ground-truth PQ -- fine for offline comparison logging, but not something
    a real deployment could ever do (no ground truth to pick by). This is that decision made the
    way it would actually have to be made for real: by looking at the candidates themselves.

    candidates: list of {"iteration": int, "image_path": str, "summary": str} (summary should
    already be ground-truth-free -- e.g. "predicted_count=20, feedback: ..."). Always called with
    every attempted iteration, even the accepted one, since an earlier attempt can visually look
    better than a later one that merely happened to be the last one tried (observed: non-monotonic
    quality across iterations is common here, not a rare edge case)."""
    if len(candidates) == 1:
        return {"chosen_iteration": candidates[0]["iteration"], "reasoning": "only one attempt was made"}
    image_paths = [c["image_path"] for c in candidates]
    listing = "\n".join(f"Image {i + 1} = iteration {c['iteration']}: {c['summary']}" for i, c in enumerate(candidates))
    prompt = (
        f"Original task: \"{task_description}\"\n"
        f"You attempted this {len(candidates)} times with different settings; each attached image "
        f"is one attempt's output, in this order:\n{listing}\n\n"
        "Look at all the attached images yourself and pick whichever attempt's output actually "
        "looks best -- don't assume a later attempt is better just because it came later, and "
        "don't rely only on the text summaries above; judge the images directly, the same way you "
        "would if you had no other information about which attempt was 'supposed to be' better.\n\n"
        "Reply with ONLY a JSON object: {\"chosen_iteration\": int, \"reasoning\": str (1-2 sentences)}"
    )
    result = qwen.ask_json_multi(image_paths, prompt, max_new_tokens=300, required_keys=["chosen_iteration", "reasoning"])
    valid_iterations = {c["iteration"] for c in candidates}
    if result["chosen_iteration"] not in valid_iterations:
        print(f"  [choose_best_output] manager picked iteration {result['chosen_iteration']!r}, not one of "
              f"{sorted(valid_iterations)} -- falling back to the last attempt")
        result["chosen_iteration"] = candidates[-1]["iteration"]
    return result


def run_countgd_with_feedback(
    qwen: QwenVLM, countgd_client: Client, image_path: str, task_description: str,
    max_iterations: int, output_dir: Path, ground_truth_count: int | None = None,
    image_id: str | None = None, expert_notes: str = "", escalate: bool = True,
) -> dict:
    """The manager always consults a private ExpertReasoner (never shown ground truth itself --
    see module docstring) via a multi-turn dialogue (run_expert_dialogue) and decides
    accept/reject itself from the transcript (decide_countgd_from_dialogue), whether or not
    ground_truth_count is given. When it isn't, the expert has no private answer either --
    NO_GROUND_TRUTH_DOSSIER tells it to reason from what's actually visible in the image, not
    fabricate a verdict -- so the manager still gets structured multi-turn scrutiny of specific
    regions/detections instead of a single blind self-score. The old MAE rule is still computed
    for history/logging only when real ground truth exists (internal_mae/internal_would_accept
    -- see module docstring), never used for the decision either way. If the loop never reaches
    accept=True, the case is queued for human review (write_escalation) instead of silently
    shipping the last attempt -- requires image_id. escalate=False suppresses that queuing
    entirely regardless of image_id -- used by evaluate_manager.py so a held-out test image that
    never gets accepted doesn't end up in the escalation_queue a human might later resolve,
    which would otherwise let test-set corrections leak back into expert_notes/running_prompt
    through that back door. expert_notes (see _apply_expert_notes) are accumulated guidance from
    previously human-resolved escalations this run, folded into the dossier so the expert's own
    reasoning stays consistent across images too, not just the manager's running_prompt."""
    count_target = interpret_countgd_target(qwen, task_description, image_path)
    print(f"[Qwen] counting target: {count_target!r}")

    dossier = (
        build_countgd_dossier(ground_truth_count, image_path) if ground_truth_count is not None
        else NO_GROUND_TRUTH_DOSSIER
    )
    dossier = _apply_expert_notes(dossier, expert_notes)
    expert = ExpertReasoner(
        qwen, EXPERT_PERSONA_COUNTGD, dossier,
        forbidden_values=[ground_truth_count] if ground_truth_count is not None else [],
    )

    history = []
    saved_path = None
    predicted_count = None
    iteration_outputs = {}  # iteration -> {"predicted_count": int, "path": Path} -- every attempt
    for i in range(1, max_iterations + 1):
        print(f"--- Iteration {i}: CountGD counting {count_target!r} ---")
        annotated_path, predicted_count = run_countgd(countgd_client, image_path, count_target)
        saved_path = output_dir / f"countgd_iteration_{i}.png"
        saved_path.write_bytes(Path(annotated_path).read_bytes())
        print(f"[CountGD] count={predicted_count}")
        iteration_outputs[i] = {"predicted_count": predicted_count, "path": saved_path}

        dialogue = run_expert_dialogue(
            qwen, expert, task_description, f"predicted count = {predicted_count}",
            str(saved_path), [image_path],
        )
        decision = decide_countgd_from_dialogue(
            qwen, task_description, count_target, predicted_count, dialogue, str(saved_path), history,
        )
        accept, revised_text, feedback = decision["accept"], decision.get("revised_text"), decision["feedback"]
        entry = {
            "iteration": i, "count_target": count_target, "predicted_count": predicted_count,
            "dialogue": dialogue, "accept": accept, "feedback": feedback,
        }
        if ground_truth_count is not None:
            internal_mae = abs(predicted_count - ground_truth_count)
            internal_would_accept = internal_mae <= mae_accept_tolerance(ground_truth_count)
            print(f"[manager] accept={accept}  (internal_mae={internal_mae}, old-rule would_accept={internal_would_accept})")
            entry["internal_mae"], entry["internal_would_accept"] = internal_mae, internal_would_accept
        else:
            print(f"[manager] accept={accept}")
        history.append(entry)

        if accept or not revised_text:
            break
        count_target = revised_text

    # Same reasoning as run_stardist_with_feedback: the manager picks which attempt to actually
    # return by looking at all of them, instead of always defaulting to the last one tried.
    chosen_iteration = history[-1]["iteration"]
    if len(iteration_outputs) > 1:
        redacted_by_iteration = {e["iteration"]: e for e in _redact_history_for_manager(history)}
        candidates = [
            {"iteration": i, "image_path": str(o["path"]), "summary": json.dumps(redacted_by_iteration.get(i, {}), default=str)}
            for i, o in sorted(iteration_outputs.items())
        ]
        choice = choose_best_output(qwen, task_description, candidates)
        chosen_iteration = choice["chosen_iteration"]
        print(f"  [choose_best_output] manager picked iteration {chosen_iteration}: {choice['reasoning']}")
        predicted_count = iteration_outputs[chosen_iteration]["predicted_count"]
        saved_path = iteration_outputs[chosen_iteration]["path"]

    if ground_truth_count is not None:
        # Comparison logging only -- never affects predicted_count/saved_path above.
        best_by_mae = min(history, key=lambda e: e["internal_mae"])
        match = "matches" if best_by_mae["iteration"] == chosen_iteration else "differs from"
        print(
            f"  [ground-truth comparison, not used] best-by-MAE iteration is "
            f"{best_by_mae['iteration']} (MAE={best_by_mae['internal_mae']}) -- {match} the "
            f"manager's own choice of iteration {chosen_iteration}"
        )

    # Escalation (and any downstream scoring of "the final result") must key off the
    # attempt the manager actually chose above, not whichever ran last -- those differ
    # whenever choose_best_output reverted to an earlier iteration.
    chosen_entry = next(h for h in history if h["iteration"] == chosen_iteration)
    if not chosen_entry["accept"] and image_id is not None and escalate:
        assert saved_path is not None, "saved_path must not be None if history has entries"
        write_escalation(
            output_dir, image_id, "countgd", task_description, image_path, saved_path, history,
            ground_truth_count,
        )
    return {
        "agent": "countgd", "count_target": count_target, "count": predicted_count,
        "annotated_image": saved_path, "history": history, "chosen_iteration": chosen_iteration,
    }


_worker_model = None  # set inside the spawned StarDist subprocess only -- stays None in this process


def _stardist_worker_init(image_path: str):
    """Runs inside the spawned subprocess. Loads (and caches) the StarDist model and the image."""
    global _worker_model
    from agentic_stardist import PRETRAINED_MODEL, load_image
    from stardist.models import StarDist2D # pyright: ignore[reportMissingImports]
    if _worker_model is None:
        _worker_model = StarDist2D.from_pretrained(PRETRAINED_MODEL)
    image = load_image(image_path)
    assert _worker_model is not None, "StarDist model failed to load"
    return image, STARDIST_INITIAL_PROB_THRESH, round(_worker_model.thresholds.nms, 3)


def _stardist_worker_run(image: np.ndarray, prob_thresh: float, nms_thresh: float,
                          gt_labels: np.ndarray | None, outlines_path: Path) -> dict:
    """Runs inside the spawned subprocess. Assumes _stardist_worker_init already ran in this
    same worker process (ProcessPoolExecutor(max_workers=1) reuses one process for every task)."""
    from agentic_stardist import compute_panoptic_quality, run_stardist, save_instance_outlines
    model = _worker_model
    assert model is not None, "StarDist model not initialized; call _stardist_worker_init first"
    labels, _ = run_stardist(model, image, prob_thresh=prob_thresh, nms_thresh=nms_thresh)
    save_instance_outlines(image, labels, outlines_path)
    pq_result = compute_panoptic_quality(labels, gt_labels) if gt_labels is not None else None
    # See STARDIST_CANDIDATE_FLOOR docstring above -- a second, cheap forward pass at a
    # permissive floor, used only as a ground-truth-free sparse-detection sanity check.
    floor_labels, _ = run_stardist(model, image, prob_thresh=STARDIST_CANDIDATE_FLOOR, nms_thresh=nms_thresh)
    candidate_count = int(floor_labels.max())
    return {"labels": labels, "pq_result": pq_result, "candidate_count": candidate_count}


def _stardist_worker_load_pannuke(fold: int, index: int):
    """Runs inside the spawned subprocess (load_pannuke_sample lives in agentic_stardist.py too)."""
    from agentic_stardist import load_pannuke_sample
    return load_pannuke_sample(fold, index)


def _stardist_worker_load_pannuke_with_classes(fold: int, index: int):
    """Runs inside the spawned subprocess. CellViT counterpart to _stardist_worker_load_pannuke
    (load_pannuke_sample_with_classes lives in agentic_stardist.py too)."""
    from agentic_stardist import load_pannuke_sample_with_classes
    return load_pannuke_sample_with_classes(fold, index)


def _stardist_worker_load_pannuke_diverse(fold: int, n: int, seed: int = 0, split: str = "all"):
    """Runs inside the spawned subprocess. Picks n indices spread across as many distinct
    PanNuke tissue types as possible (agentic_stardist.select_diverse_indices) instead of the
    first n (a single contiguous tissue block), then fetches all of them in one batched
    load_pannuke_samples call rather than n separate from-scratch reads. split ("all"/"train"/
    "test") is select_diverse_indices's own train/test partition -- see its docstring; same idea
    as bbbc005.load_bbbc005_samples's split parameter for CountGD."""
    from agentic_stardist import TISSUE_DIVERSITY_MAX_INDEX, load_pannuke_samples, load_pannuke_types, select_diverse_indices
    all_types = load_pannuke_types(fold)
    selected = select_diverse_indices(all_types, n, max_index=TISSUE_DIVERSITY_MAX_INDEX, seed=seed, split=split)
    images_prefix, gt_labels_prefix, tissue_prefix = load_pannuke_samples(fold, selected[-1] + 1)
    return (
        selected,
        [images_prefix[i] for i in selected],
        [gt_labels_prefix[i] for i in selected],
        [tissue_prefix[i] for i in selected],
    )


def _stardist_worker_load_pannuke_diverse_with_classes(fold: int, n: int, seed: int = 0, split: str = "all"):
    """Runs inside the spawned subprocess. CellViT counterpart to
    _stardist_worker_load_pannuke_diverse: same diverse-tissue index selection, but returns
    per-class ground truth (agentic_stardist.load_pannuke_samples_with_classes) instead of one
    class-agnostic instance mask, since CellViT scores per pathology type. split ("all"/"train"/
    "test") is select_diverse_indices's own train/test partition -- see its docstring; same idea
    as bbbc005.load_bbbc005_samples's split parameter for CountGD."""
    from agentic_stardist import (
        TISSUE_DIVERSITY_MAX_INDEX, load_pannuke_samples_with_classes, load_pannuke_types, select_diverse_indices,
    )
    all_types = load_pannuke_types(fold)
    selected = select_diverse_indices(all_types, n, max_index=TISSUE_DIVERSITY_MAX_INDEX, seed=seed, split=split)
    images_prefix, class_counts_prefix, class_labels_prefix, tissue_prefix = load_pannuke_samples_with_classes(
        fold, selected[-1] + 1
    )
    return (
        selected,
        [images_prefix[i] for i in selected],
        [class_counts_prefix[i] for i in selected],
        [class_labels_prefix[i] for i in selected],
        [tissue_prefix[i] for i in selected],
    )


def _stardist_worker_score_cellvit_predictions(
    predicted_cells: list, gt_class_instance_labels: dict, image_shape: tuple, patch_size: int = 1024,
) -> dict:
    """Runs inside the spawned subprocess (compute_panoptic_quality lives in agentic_stardist.py,
    TF-loaded-process-only -- see the module docstring). predicted_cells is plain data (no
    CellViT-specific objects, so this stays torch-free): a list of {"type_name": str, "contour":
    [(x, y), ...]}, in CellViT's PATCH_SIZE x PATCH_SIZE letterboxed-patch coordinate space (see
    agentic_cellvit.load_patch) -- since PanNuke images (256x256) are smaller than PATCH_SIZE,
    load_patch never downscales them, only centers them on a black canvas, so contours are
    shifted by a fixed offset relative to image_shape and need that offset subtracted back out
    before they line up with the (image_shape-sized) ground-truth arrays. gt_class_instance_labels
    is {class_name: (H, W) instance-label ndarray} from
    agentic_stardist.pannuke_class_instance_labels -- already in compute_panoptic_quality's
    expected standard label-mask format, one array per class. Returns per-class {"pq",
    "mean_iou", "tp", "fp", "fn"} plus the macro-averaged internal_mpq/internal_f1 used as
    CellViT's internal (logged-only, never decision-driving) ground-truth metric."""
    from agentic_stardist import compute_panoptic_quality
    from PIL import ImageDraw

    h, w = image_shape
    off_x, off_y = (patch_size - w) // 2, (patch_size - h) // 2

    cells_by_class: dict = {}
    for cell in predicted_cells:
        cells_by_class.setdefault(cell["type_name"], []).append(cell)

    per_class = {}
    for name, gt_labels in gt_class_instance_labels.items():
        pred_labels = Image.new("I", (w, h), 0)
        draw = ImageDraw.Draw(pred_labels)
        for i, cell in enumerate(cells_by_class.get(name, []), start=1):
            pts = [(x - off_x, y - off_y) for x, y in cell["contour"]]
            if len(pts) >= 3:
                draw.polygon(pts, fill=i)
        pred_labels_arr = np.array(pred_labels, dtype=np.int32)
        per_class[name] = compute_panoptic_quality(pred_labels_arr, gt_labels)

    internal_mpq = sum(v["pq"] for v in per_class.values()) / len(per_class)

    def _f1(v: dict) -> float:
        tp, fp, fn = v["tp"], v["fp"], v["fn"]
        denom = tp + 0.5 * (fp + fn)
        return tp / denom if denom else 1.0

    internal_f1 = sum(_f1(v) for v in per_class.values()) / len(per_class)
    return {"per_class": per_class, "internal_mpq": internal_mpq, "internal_f1": internal_f1}


def _stardist_worker_revert_to_best(image: np.ndarray, history: list, outlines_path: Path) -> dict | None:
    """Runs inside the spawned subprocess. best_entry is pure logic (max(history, key=pq)) but
    lives in agentic_stardist.py, so it still needs to run in here rather than the parent --
    see the module docstring. Returns None if the last iteration tried was already the best."""
    from agentic_stardist import best_entry, run_stardist, save_instance_outlines
    best = best_entry(history)
    if best["iteration"] == history[-1]["iteration"]:
        return None  # type: ignore[return-value]  # dict | None is correct; ignore checker
    model = _worker_model
    assert model is not None, "StarDist model not initialized; call _stardist_worker_init first"
    labels, _ = run_stardist(model, image, prob_thresh=best["prob_thresh"], nms_thresh=best["nms_thresh"])
    save_instance_outlines(image, labels, outlines_path)
    return {"labels": labels, "best_iteration": best["iteration"], "best_pq": best["pq"]}


def _stardist_worker_save_gt_outlines(image: np.ndarray, gt_labels: np.ndarray, outlines_path: Path) -> None:
    """Runs inside the spawned subprocess (save_instance_outlines needs the same TF-avoidance
    treatment as everything else here -- see the module docstring). Renders the ground-truth
    instance mask the same way predictions are rendered, once per run -- this is the private
    image ExpertReasoner (not the manager) looks at."""
    from agentic_stardist import save_instance_outlines
    save_instance_outlines(image, gt_labels, outlines_path)


class StardistWorker:
    """Owns a single persistent spawned subprocess that all StarDist calls are routed through --
    see the module docstring for why StarDist/TensorFlow can't share a process with torch/ROCm.
    `spawn` (not the Linux default `fork`) is required: fork would duplicate this process's
    already-loaded ROCm/Triton state into the child, reintroducing the exact conflict."""

    def __init__(self):
        self._pool = ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn"))

    def init(self, image_path: str):
        return self._pool.submit(_stardist_worker_init, image_path).result()

    def run(self, image: np.ndarray, prob_thresh: float, nms_thresh: float,
            gt_labels: np.ndarray | None, outlines_path: Path) -> dict:
        return self._pool.submit(
            _stardist_worker_run, image, prob_thresh, nms_thresh, gt_labels, outlines_path
        ).result()

    def load_pannuke_sample(self, fold: int, index: int):
        return self._pool.submit(_stardist_worker_load_pannuke, fold, index).result()

    def load_pannuke_sample_with_classes(self, fold: int, index: int):
        return self._pool.submit(_stardist_worker_load_pannuke_with_classes, fold, index).result()

    def load_pannuke_diverse(self, fold: int, n: int, seed: int = 0, split: str = "all"):
        return self._pool.submit(_stardist_worker_load_pannuke_diverse, fold, n, seed, split).result()

    def load_pannuke_diverse_with_classes(self, fold: int, n: int, seed: int = 0, split: str = "all"):
        return self._pool.submit(
            _stardist_worker_load_pannuke_diverse_with_classes, fold, n, seed, split
        ).result()

    def score_cellvit_predictions(self, predicted_cells: list, gt_class_instance_labels: dict, image_shape: tuple):
        return self._pool.submit(
            _stardist_worker_score_cellvit_predictions, predicted_cells, gt_class_instance_labels, image_shape
        ).result()

    def revert_to_best(self, image: np.ndarray, history: list, outlines_path: Path):
        return self._pool.submit(_stardist_worker_revert_to_best, image, history, outlines_path).result()

    def save_gt_outlines(self, image: np.ndarray, gt_labels: np.ndarray, outlines_path: Path):
        return self._pool.submit(_stardist_worker_save_gt_outlines, image, gt_labels, outlines_path).result()

    def shutdown(self):
        self._pool.shutdown(wait=True)


class CellvitClient:
    """Lazily loads CellViT's CellSegmentationInference in this same process -- unlike
    StarDist, empirically verified (see CLAUDE.md) that CellViT's inference path coexists fine
    with the manager's own loaded Qwen/ROCm-torch in one process across repeated GPU calls
    (both models resident, ~20GB combined vs. ~45GB+ still free on this machine), so no
    StardistWorker-style subprocess isolation is needed here."""

    def __init__(self, checkpoint: str, cellvit_repo: str | None = None, gpu: int = 0,
                 magnification: float = 40.0, enforce_amp: bool = False):
        self.checkpoint = checkpoint
        self.cellvit_repo = cellvit_repo
        self.gpu = gpu
        self.magnification = magnification
        self.enforce_amp = enforce_amp
        self._inferer = None
        self._color_dict = None

    def _load(self):
        if self._inferer is None:
            from agentic_cellvit import load_cellvit_module
            CellSegmentationInference, color_dict = load_cellvit_module(self.cellvit_repo)
            self._inferer = CellSegmentationInference(
                model_path=self.checkpoint, gpu=self.gpu, enforce_mixed_precision=self.enforce_amp
            )
            self._color_dict = color_dict
        return self._inferer, self._color_dict

    def run(self, image_path: str, target_classes: set, prob_threshold: float):
        from agentic_cellvit import run_cellvit
        inferer, color_dict = self._load()
        assert color_dict is not None, "color_dict failed to load"
        return run_cellvit(inferer, image_path, self.magnification, target_classes, prob_threshold, color_dict)


class DeepGleasonClient:
    """Unlike CellViT/StarDist, DeepGleason is already maximally isolated by construction --
    agentic_deepgleason.run_deepgleason shells out via subprocess.run to a wholly separate conda
    environment's interpreter (Python 3.11, pinned TensorFlow/AUCMEDI), not even the same Python
    family as this process, so there's no StardistWorker-style LLVM-collision risk to design
    around here.

    The real asymmetry this class exists to handle: run_deepgleason (full WSI tiling + model
    inference over every tile) is expensive and produces a raw per-tile predictions CSV;
    re-aggregating that CSV with a different confidence_threshold (aggregate_gleason) is cheap
    and needs no rerun. run_slide() does the expensive part once per image and caches the
    resulting paths; aggregate() repeats the cheap part every retry iteration."""

    def __init__(self, repo: str | None = None, python: str | None = None, model: str | None = None):
        self.repo = repo
        self.python = python
        self.model = model
        self._predictions_path = None
        self._overlay_path = None
        self._preview_path = None

    def _configure_module(self):
        import agentic_deepgleason as dg
        if self.repo:
            dg.DEEPGLEASON_REPO = Path(self.repo)
            dg.DEEPGLEASON_MODEL = Path(self.repo) / "models" / "model.ConvNeXtBase.hdf5"
        if self.model:
            dg.DEEPGLEASON_MODEL = Path(self.model)
        if self.python:
            dg.DEEPGLEASON_PYTHON = self.python
        return dg

    def run_slide(self, slide_path: str, output_dir: Path) -> None:
        """Runs the expensive DeepGleason subprocess once; caches predictions_path/overlay_path/
        preview_path for aggregate() to reuse across iterations."""
        dg = self._configure_module()
        self._predictions_path, self._overlay_path = dg.run_deepgleason(slide_path, output_dir, generate_overlay=True)
        self._preview_path = output_dir / "gleason_overlay_preview.png"
        dg.render_overlay_preview(self._overlay_path, self._preview_path)

    def aggregate(self, confidence_threshold: float) -> dict:
        assert self._predictions_path is not None, "run_slide must be called before aggregate"
        dg = self._configure_module()
        return dg.aggregate_gleason(self._predictions_path, confidence_threshold=confidence_threshold)

    @property
    def preview_path(self) -> Path:
        assert self._preview_path is not None, "run_slide must be called before preview_path is available"
        return self._preview_path


ESCALATION_QUEUE_DIRNAME = "escalation_queue"


def write_escalation(
    output_dir: Path, image_id: str, agent: str, task_description: str, original_image_path: str,
    final_image_path: Path, history: list, ground_truth, tissue: str | None = None,
) -> None:
    """Queues a case for human review instead of silently accepting whatever the retry loop
    landed on -- written when the automated loop exhausts max_iterations without ever reaching
    accept=True. Deliberately a plain JSON file in a directory, not a real queue/broker: this is
    a single-machine script, and a human resolves these later (resolve_escalations.py), on their
    own time, without blocking the batch run that's still processing other images. ground_truth
    is StarDist's instance-label ndarray or CountGD's int count -- saved alongside (as .npy for
    the array case) purely so the later human<->expert conversation can reconstruct the same
    ExpertReasoner. It's never shown to the human, same boundary as everywhere else in this
    module."""
    queue_dir = output_dir / ESCALATION_QUEUE_DIRNAME
    queue_dir.mkdir(parents=True, exist_ok=True)
    ground_truth_path, ground_truth_value = None, None
    if isinstance(ground_truth, np.ndarray):
        ground_truth_path = queue_dir / f"{image_id}_gt.npy"
        np.save(ground_truth_path, ground_truth)
    elif ground_truth is not None:
        ground_truth_value = ground_truth
    record = {
        "image_id": image_id, "agent": agent, "task_description": task_description,
        "original_image_path": str(original_image_path), "final_image_path": str(final_image_path),
        "history": history, "tissue": tissue,
        "ground_truth_path": str(ground_truth_path) if ground_truth_path else None,
        "ground_truth_value": ground_truth_value, "status": "pending",
    }
    record_path = queue_dir / f"{image_id}.json"
    record_path.write_text(json.dumps(record, indent=2, default=str))
    print(f"  [escalation] never reached an accepted result -- queued for human review at {record_path}")


def run_stardist_with_feedback(
    qwen: QwenVLM, worker: StardistWorker, image_path: str, task_description: str,
    max_iterations: int, output_dir: Path, ground_truth_labels: np.ndarray | None = None, tissue: str | None = None,
    image_id: str | None = None, expert_notes: str = "", escalate: bool = True,
) -> dict:
    """The manager always consults a private ExpertReasoner (never shown ground truth itself --
    see module docstring) via a multi-turn dialogue (run_expert_dialogue) and decides
    accept/reject itself from the transcript (decide_stardist_from_dialogue), whether or not
    ground_truth_labels is given. When it is (a PanNuke-style instance mask), the expert also
    gets a one-time outline rendering of the true instance boundaries and, if known, the PanNuke
    tissue type. When it isn't, the expert has no private answer either -- NO_GROUND_TRUTH_DOSSIER
    tells it to reason from what's actually visible in the image, not fabricate a verdict -- so
    the manager still gets structured multi-turn scrutiny instead of a single blind self-score.
    The old PQ rule is still computed for history/logging only when real ground truth exists
    (internal_pq/internal_would_accept -- see module docstring), never used for the decision
    either way. If the loop never reaches accept=True, the case is queued for human review
    (write_escalation) instead of silently shipping whatever the last/best attempt was --
    requires image_id. escalate=False suppresses that queuing regardless of image_id (see
    run_countgd_with_feedback's docstring for why -- used by evaluate_manager.py). expert_notes
    (see _apply_expert_notes) are accumulated guidance from previously human-resolved
    escalations this run, folded into the dossier so the expert's own reasoning stays consistent
    across images too, not just the manager's running_prompt."""
    image, prob_thresh, nms_thresh = worker.init(image_path)

    gt_outlines_path = output_dir / "stardist_ground_truth.png"
    if ground_truth_labels is not None:
        worker.save_gt_outlines(image, ground_truth_labels, gt_outlines_path)
        dossier = build_stardist_dossier(ground_truth_labels, tissue)
    else:
        dossier = NO_GROUND_TRUTH_DOSSIER
    dossier = _apply_expert_notes(dossier, expert_notes)
    expert = ExpertReasoner(
        qwen, EXPERT_PERSONA_STARDIST, dossier,
        forbidden_values=[int(ground_truth_labels.max())] if ground_truth_labels is not None else [],
    )

    history = []
    saved_path = None
    labels = None
    iteration_outputs = {}  # iteration -> {"labels": ndarray, "path": Path} -- every attempt, not
                             # just the last/accepted one, so choose_best_output can compare all of them
    for i in range(1, max_iterations + 1):
        print(f"--- Iteration {i}: StarDist prob_thresh={prob_thresh:.3f}, nms_thresh={nms_thresh:.3f} ---")
        saved_path = output_dir / f"stardist_iteration_{i}.png"
        result = worker.run(image, prob_thresh, nms_thresh, ground_truth_labels, saved_path)
        labels = result["labels"]
        predicted_count = int(labels.max())
        print(f"[StarDist] nuclei={predicted_count}")
        iteration_outputs[i] = {"labels": labels, "path": saved_path}

        candidate_count = result["candidate_count"]
        coverage_ratio = predicted_count / max(candidate_count, 1)
        guardrail_triggered = (
            candidate_count >= STARDIST_COVERAGE_MIN_CANDIDATES and coverage_ratio < STARDIST_COVERAGE_RATIO_MIN
        )
        if guardrail_triggered:
            print(f"  [guardrail] only {predicted_count}/{candidate_count} candidates kept at "
                  f"prob_thresh={prob_thresh:.3f} (floor={STARDIST_CANDIDATE_FLOOR} finds {candidate_count}) "
                  f"-- overriding accept=False and forcing prob_thresh down regardless of the dialogue")

        expert_image_paths = [str(saved_path), str(gt_outlines_path)] if ground_truth_labels is not None else [str(saved_path)]
        dialogue = run_expert_dialogue(
            qwen, expert, task_description,
            f"detected nuclei = {predicted_count} (prob_thresh={prob_thresh:.3f}, nms_thresh={nms_thresh:.3f})",
            str(saved_path), expert_image_paths,
        )
        decision = decide_stardist_from_dialogue(
            qwen, task_description, prob_thresh, nms_thresh, predicted_count, dialogue, str(saved_path), history,
        )
        accept = decision["accept"]
        revised_prob, revised_nms, feedback = decision.get("revised_prob_thresh"), decision.get("revised_nms_thresh"), decision["feedback"]
        if guardrail_triggered:
            accept = False
            revised_prob = max(STARDIST_CANDIDATE_FLOOR, prob_thresh - (prob_thresh - STARDIST_CANDIDATE_FLOOR) * 0.5)
            feedback = (f"[guardrail override] {predicted_count}/{candidate_count} candidates kept -- "
                         f"detection looks suspiciously sparse regardless of dialogue content. {feedback}")
        entry = {
            "iteration": i, "prob_thresh": prob_thresh, "nms_thresh": nms_thresh,
            "predicted_count": predicted_count, "dialogue": dialogue, "accept": accept, "feedback": feedback,
        }
        pq_result = result["pq_result"]
        if pq_result is not None:
            # pq/mean_iou/tp/fp/fn keep agentic_stardist.py's own field names (unprefixed) --
            # best_entry() below (imported from that untouched module) keys off e["pq"] to pick
            # which attempted iteration to report as final. None of this is shown to the
            # manager or the expert; only internal_would_accept/internal_recall are new (the
            # old threshold rule, and a coverage-only stat -- tp/(tp+fn), how much of the true
            # object set was even found regardless of outline quality -- logged for comparison
            # against the manager's dialogue-driven `accept` above, never used to decide it).
            internal_would_accept = bool(pq_result["pq"] >= ACCEPT_PQ_THRESHOLD)
            tp, fn = pq_result["tp"], pq_result["fn"]
            internal_recall = tp / (tp + fn) if (tp + fn) else 1.0  # coverage only -- see module docstring
            print(f"[manager] accept={accept}  (internal_pq={pq_result['pq']:.3f}, "
                  f"internal_recall={internal_recall:.3f}, old-rule would_accept={internal_would_accept})")
            entry.update({
                "pq": pq_result["pq"], "mean_iou": pq_result["mean_iou"],
                "tp": pq_result["tp"], "fp": pq_result["fp"], "fn": pq_result["fn"],
                "internal_would_accept": internal_would_accept, "internal_recall": internal_recall,
            })
        else:
            print(f"[manager] accept={accept}")
        history.append(entry)

        # A "revision" that repeats the current value changes nothing -- treat it as no
        # revision at all rather than burning an iteration re-running identical thresholds
        # (observed: manager proposed prob_thresh=0.692 when already at 0.692).
        if revised_prob is not None and abs(revised_prob - prob_thresh) < 1e-9:
            revised_prob = None
        if revised_nms is not None and abs(revised_nms - nms_thresh) < 1e-9:
            revised_nms = None

        if accept or (revised_prob is None and revised_nms is None):
            break
        if revised_prob is not None:
            prob_thresh = revised_prob
        if revised_nms is not None:
            nms_thresh = revised_nms

    # Which attempt to actually return is the manager's own call across every attempt made, not
    # just whichever one happened to run last or get accept=True -- see choose_best_output.
    chosen_iteration = history[-1]["iteration"]
    if len(iteration_outputs) > 1:
        redacted_by_iteration = {e["iteration"]: e for e in _redact_history_for_manager(history)}
        candidates = [
            {"iteration": i, "image_path": str(o["path"]), "summary": json.dumps(redacted_by_iteration.get(i, {}), default=str)}
            for i, o in sorted(iteration_outputs.items())
        ]
        choice = choose_best_output(qwen, task_description, candidates)
        chosen_iteration = choice["chosen_iteration"]
        print(f"  [choose_best_output] manager picked iteration {chosen_iteration}: {choice['reasoning']}")
        labels = iteration_outputs[chosen_iteration]["labels"]
        saved_path = iteration_outputs[chosen_iteration]["path"]

    if ground_truth_labels is not None:
        # Comparison logging only -- never affects labels/saved_path above. This used to be what
        # actually got returned (picking by the hidden ground-truth PQ), which only works in this
        # training script; a real deployment has no ground truth to pick by, hence the manager's
        # own choice above being the real selection now.
        revert = worker.revert_to_best(image, history, output_dir / "stardist_best_by_pq.png")
        if revert is not None:
            match = "matches" if revert["best_iteration"] == chosen_iteration else "differs from"
            print(
                f"  [ground-truth comparison, not used] best-by-PQ iteration is "
                f"{revert['best_iteration']} (PQ={revert['best_pq']:.3f}) -- {match} the manager's "
                f"own choice of iteration {chosen_iteration}"
            )

    assert labels is not None, "max_iterations must be at least 1"
    # Same reasoning as run_countgd_with_feedback: escalation must key off the attempt the
    # manager actually chose (chosen_iteration), not whichever ran last -- those differ
    # whenever choose_best_output reverted to an earlier iteration.
    chosen_entry = next(h for h in history if h["iteration"] == chosen_iteration)
    if not chosen_entry["accept"] and image_id is not None and escalate:
        assert saved_path is not None, "saved_path must not be None when writing escalation"
        write_escalation(
            output_dir, image_id, "stardist", task_description, image_path, saved_path, history,
            ground_truth_labels, tissue,
        )
    return {
        "agent": "stardist", "num_nuclei": int(labels.max()), "labels": labels,
        "outlines_image": saved_path, "history": history, "chosen_iteration": chosen_iteration,
    }


def run_cellvit_with_feedback(
    qwen: QwenVLM, cellvit_client: CellvitClient, image_path: str, task_description: str,
    max_iterations: int, output_dir: Path,
    ground_truth_counts_by_type: dict | None = None, ground_truth_class_labels: dict | None = None,
    stardist_worker: StardistWorker | None = None, tissue: str | None = None, image_id: str | None = None,
    expert_notes: str = "", escalate: bool = True,
) -> dict:
    """The manager always consults a private ExpertReasoner (never shown ground truth itself --
    see module docstring) via a multi-turn dialogue (run_expert_dialogue) and decides
    accept/reject itself from the transcript (decide_cellvit_from_dialogue), whether or not
    ground_truth_counts_by_type is given. When it is (PanNuke per-class true instance counts), the
    expert also gets the PanNuke tissue type if known. When it isn't, the expert has no private
    answer either -- NO_GROUND_TRUTH_DOSSIER tells it to reason from what's actually visible in
    the image, not fabricate a verdict -- so the manager still gets structured multi-turn scrutiny
    instead of a single blind self-score. An internal per-class mPQ/F1 score (internal_mpq/
    internal_f1) is additionally computed for history/logging only -- never used for the decision
    -- when ground_truth_class_labels (the raw per-class instance-label arrays needed to actually
    score against, not just count) and stardist_worker are also given. If the loop never reaches
    accept=True, the case is queued for human review (write_escalation) instead of silently
    shipping whatever the last/best attempt was -- requires image_id. escalate=False suppresses
    that queuing regardless of image_id (see run_countgd_with_feedback's docstring for why --
    used by evaluate_manager.py). expert_notes (see _apply_expert_notes) are accumulated guidance
    from previously human-resolved escalations this run, folded into the dossier so the expert's
    own reasoning stays consistent across images too, not just the manager's running_prompt.

    Unlike StarDist/CountGD, scoring against ground truth needs a subprocess round-trip
    (StardistWorker.score_cellvit_predictions) even though CellViT itself runs in-process here --
    compute_panoptic_quality lives in agentic_stardist.py, which this file only ever imports
    inside StarDist's TensorFlow-loaded subprocess (see StardistWorker's own docstring);
    reusing that machinery instead of duplicating PQ-scoring logic a third time."""
    import agentic_cellvit as cellvit_module
    request = cellvit_module.interpret_request(qwen, task_description, image_path)
    target_classes = set(request["target_classes"])
    prob_threshold = request["prob_threshold"]
    print(f"[Qwen] target classes: {sorted(target_classes)}, prob_threshold={prob_threshold:.2f}")

    dossier = (
        build_cellvit_dossier(ground_truth_counts_by_type, tissue) if ground_truth_counts_by_type is not None
        else NO_GROUND_TRUTH_DOSSIER
    )
    dossier = _apply_expert_notes(dossier, expert_notes)
    expert = ExpertReasoner(
        qwen, EXPERT_PERSONA_CELLVIT, dossier,
        forbidden_values=[int(v) for v in ground_truth_counts_by_type.values()]
        if ground_truth_counts_by_type is not None else [],
    )

    history = []
    saved_path = None
    predicted_count = None
    counts_by_type = {}
    iteration_outputs = {}  # iteration -> {"predicted_count": int, "counts_by_type": dict, "path": Path} --
                             # every attempt, not just the last/accepted one, so choose_best_output can compare all
    for i in range(1, max_iterations + 1):
        print(f"--- Iteration {i}: CellViT highlighting {sorted(target_classes)} (p>={prob_threshold:.2f}) ---")
        annotated, matched_cells, counts_by_type, all_cells = cellvit_client.run(image_path, target_classes, prob_threshold)
        predicted_count = len(matched_cells)
        print(f"[CellViT] matched count={predicted_count}, all detected by type={counts_by_type}")

        saved_path = output_dir / f"cellvit_iteration_{i}.png"
        annotated.save(saved_path)
        iteration_outputs[i] = {"predicted_count": predicted_count, "counts_by_type": counts_by_type, "path": saved_path}

        dialogue = run_expert_dialogue(
            qwen, expert, task_description,
            f"matched count = {predicted_count} for {sorted(target_classes)} "
            f"(prob_threshold={prob_threshold:.3f}); all detected cells by type = {counts_by_type}",
            str(saved_path), [image_path],
        )
        decision = decide_cellvit_from_dialogue(
            qwen, task_description, sorted(target_classes), prob_threshold, predicted_count,
            counts_by_type, dialogue, str(saved_path), history,
        )
        accept = decision["accept"]
        revised_classes = decision.get("revised_target_classes")
        revised_threshold = decision.get("revised_prob_threshold")
        feedback = decision["feedback"]

        internal_mpq = internal_f1 = per_class_scores = None
        if ground_truth_class_labels is not None and stardist_worker is not None:
            image_shape = next(iter(ground_truth_class_labels.values())).shape
            score_result = stardist_worker.score_cellvit_predictions(all_cells, ground_truth_class_labels, image_shape)
            internal_mpq, internal_f1 = score_result["internal_mpq"], score_result["internal_f1"]
            per_class_scores = score_result["per_class"]
            print(f"[manager] accept={accept}  (internal_mpq={internal_mpq:.3f}, internal_f1={internal_f1:.3f})")
        else:
            print(f"[manager] accept={accept}")

        history.append({
            "iteration": i, "target_classes": sorted(target_classes), "prob_threshold": prob_threshold,
            "predicted_count": predicted_count, "counts_by_type": counts_by_type,
            "dialogue": dialogue, "accept": accept, "feedback": feedback,
            "internal_mpq": internal_mpq, "internal_f1": internal_f1, "per_class_scores": per_class_scores,
        })

        # A "revision" that repeats the current config changes nothing -- treat it as no
        # revision at all, same reasoning as StarDist's prob_thresh/nms_thresh epsilon check.
        if revised_classes is not None and set(revised_classes) == target_classes:
            revised_classes = None
        if revised_threshold is not None and abs(revised_threshold - prob_threshold) < 1e-9:
            revised_threshold = None

        if accept or (revised_classes is None and revised_threshold is None):
            break
        if revised_classes is not None:
            target_classes = set(revised_classes)
        if revised_threshold is not None:
            prob_threshold = revised_threshold

    # Which attempt to actually return is the manager's own call across every attempt made, not
    # just whichever one happened to run last or get accept=True -- see choose_best_output.
    chosen_iteration = history[-1]["iteration"]
    if len(iteration_outputs) > 1:
        redacted_by_iteration = {e["iteration"]: e for e in _redact_history_for_manager(history)}
        candidates = [
            {"iteration": i, "image_path": str(o["path"]), "summary": json.dumps(redacted_by_iteration.get(i, {}), default=str)}
            for i, o in sorted(iteration_outputs.items())
        ]
        choice = choose_best_output(qwen, task_description, candidates)
        chosen_iteration = choice["chosen_iteration"]
        print(f"  [choose_best_output] manager picked iteration {chosen_iteration}: {choice['reasoning']}")
        predicted_count = iteration_outputs[chosen_iteration]["predicted_count"]
        counts_by_type = iteration_outputs[chosen_iteration]["counts_by_type"]
        saved_path = iteration_outputs[chosen_iteration]["path"]

    chosen_entry = next(h for h in history if h["iteration"] == chosen_iteration)
    if not chosen_entry["accept"] and image_id is not None and escalate:
        assert saved_path is not None, "saved_path must not be None when writing escalation"
        write_escalation(
            output_dir, image_id, "cellvit", task_description, image_path, saved_path, history,
            ground_truth_counts_by_type, tissue,
        )
    return {
        "agent": "cellvit", "target_classes": sorted(target_classes), "prob_threshold": prob_threshold,
        "count": predicted_count, "counts_by_type": counts_by_type, "annotated_image": saved_path,
        "history": history, "chosen_iteration": chosen_iteration,
    }


DEEPGLEASON_INITIAL_CONFIDENCE_THRESHOLD = 0.0  # 0.0 = every tile counts (DeepGleason's own default behavior)


def run_deepgleason_with_feedback(
    qwen: QwenVLM, deepgleason_client: DeepGleasonClient, slide_path: str, task_description: str,
    max_iterations: int, output_dir: Path,
    ground_truth_gleason_score: str | None = None, ground_truth_isup_grade: int | None = None,
    image_id: str | None = None, expert_notes: str = "", escalate: bool = True,
) -> dict:
    """The manager always consults a private ExpertReasoner (never shown ground truth itself --
    see module docstring) via a multi-turn dialogue (run_expert_dialogue) and decides
    accept/reject itself from the transcript (decide_deepgleason_from_dialogue), whether or not
    ground_truth_gleason_score/ground_truth_isup_grade are given. When they aren't, the expert
    has no private answer either -- NO_GROUND_TRUTH_DOSSIER tells it to reason from what's
    actually visible in the overlay image, not fabricate a verdict. expert_notes (see
    _apply_expert_notes) are accumulated guidance from previously human-resolved escalations
    this run, folded into the dossier so the expert's own reasoning stays consistent across
    images too, not just the manager's running_prompt.

    Unlike CountGD/StarDist/CellViT, this never reruns the underlying model on retry: the
    DeepGleason subprocess (expensive -- full WSI tiling + inference) runs exactly once, via
    deepgleason_client.run_slide(); every iteration only re-aggregates the same cached
    predictions CSV with a revised confidence_threshold (cheap -- see DeepGleasonClient).

    Internal metric when ground truth is given: internal_isup_mae = abs(predicted_isup_grade -
    ground_truth_isup_grade), logged only, never used for the decision (same role as
    internal_mae/internal_pq elsewhere). "No tumor found" is treated as ISUP grade 0 for this
    purpose -- an ordinal extension, since no tumor is less severe than any graded one. If the
    loop never reaches accept=True, the case is queued for human review (write_escalation) --
    requires image_id. escalate=False suppresses that queuing regardless of image_id (see
    run_countgd_with_feedback's docstring for why -- used by evaluate_manager.py)."""
    deepgleason_client.run_slide(slide_path, output_dir)
    preview_path = deepgleason_client.preview_path
    confidence_threshold = DEEPGLEASON_INITIAL_CONFIDENCE_THRESHOLD

    dossier = (
        build_deepgleason_dossier(ground_truth_gleason_score, ground_truth_isup_grade)
        if ground_truth_gleason_score is not None or ground_truth_isup_grade is not None
        else NO_GROUND_TRUTH_DOSSIER
    )
    dossier = _apply_expert_notes(dossier, expert_notes)
    forbidden_values = [v for v in (ground_truth_gleason_score, ground_truth_isup_grade) if v is not None]
    expert = ExpertReasoner(qwen, EXPERT_PERSONA_DEEPGLEASON, dossier, forbidden_values=forbidden_values)

    history = []
    for i in range(1, max_iterations + 1):
        print(f"--- Iteration {i}: DeepGleason confidence_threshold={confidence_threshold:.2f} ---")
        gleason_result = deepgleason_client.aggregate(confidence_threshold)
        print(f"[DeepGleason] {gleason_result}")

        dialogue = run_expert_dialogue(
            qwen, expert, task_description,
            f"confidence_threshold={confidence_threshold:.2f}; result={gleason_result}",
            str(preview_path), [str(preview_path)],
        )
        decision = decide_deepgleason_from_dialogue(
            qwen, task_description, confidence_threshold, gleason_result, dialogue, str(preview_path), history,
        )
        accept = decision["accept"]
        revised_threshold = decision.get("revised_confidence_threshold")
        feedback = decision["feedback"]

        entry = {
            "iteration": i, "confidence_threshold": confidence_threshold, "gleason_result": gleason_result,
            "dialogue": dialogue, "accept": accept, "feedback": feedback,
        }
        if ground_truth_isup_grade is not None:
            predicted_grade = gleason_result["isup_grade"] if gleason_result["tumor_found"] else 0
            internal_isup_mae = abs(predicted_grade - ground_truth_isup_grade)
            print(f"[manager] accept={accept}  (internal_isup_mae={internal_isup_mae})")
            entry["internal_isup_mae"] = internal_isup_mae
        else:
            print(f"[manager] accept={accept}")
        history.append(entry)

        if revised_threshold is not None and abs(revised_threshold - confidence_threshold) < 1e-9:
            revised_threshold = None
        if accept or revised_threshold is None:
            break
        confidence_threshold = revised_threshold

    # No choose_best_output here, unlike CountGD/StarDist/CellViT: those rerun their full model
    # each iteration and can produce genuinely different (sometimes non-monotonically better/
    # worse) visual outputs to compare. Here every iteration re-aggregates the exact same cached
    # predictions CSV against the exact same overlay image -- only the numeric gleason_result
    # changes, not anything to visually compare -- so the last iteration's result is simply used.
    if not history[-1]["accept"] and image_id is not None and escalate:
        ground_truth = None
        if ground_truth_gleason_score is not None or ground_truth_isup_grade is not None:
            ground_truth = {"gleason_score": ground_truth_gleason_score, "isup_grade": ground_truth_isup_grade}
        write_escalation(
            output_dir, image_id, "deepgleason", task_description, slide_path, preview_path, history, ground_truth,
        )
    return {
        "agent": "deepgleason", "confidence_threshold": confidence_threshold,
        "gleason_result": history[-1]["gleason_result"], "overlay_preview": preview_path,
        "history": history, "chosen_iteration": history[-1]["iteration"],
    }


class ManagerAgent:
    """Routes a task to CountGD, StarDist, or CellViT using Qwen3-VL, and drives Qwen's own
    retry/scoring loop against whichever agent it picked."""

    def __init__(
        self, model_id: str = MODEL_ID, cellvit_checkpoint: str | None = None,
        cellvit_repo: str | None = None, cellvit_gpu: int = 0, cellvit_magnification: float = 40.0,
        deepgleason_repo: str | None = None, deepgleason_python: str | None = None,
        deepgleason_model: str | None = None,
    ):
        self.qwen = QwenVLM(model_id)
        self._countgd_client = None
        self._stardist_worker = None
        self._cellvit_client = None
        self._deepgleason_client = None
        self.cellvit_checkpoint = cellvit_checkpoint
        self.cellvit_repo = cellvit_repo
        self.cellvit_gpu = cellvit_gpu
        self.cellvit_magnification = cellvit_magnification
        self.deepgleason_repo = deepgleason_repo
        self.deepgleason_python = deepgleason_python
        self.deepgleason_model = deepgleason_model

    @property
    def countgd_client(self) -> Client:
        if self._countgd_client is None:
            self._countgd_client = Client(COUNTGD_SPACE)
        return self._countgd_client

    @property
    def stardist_worker(self) -> StardistWorker:
        if self._stardist_worker is None:
            self._stardist_worker = StardistWorker()
        return self._stardist_worker

    @property
    def cellvit_client(self) -> CellvitClient:
        if self._cellvit_client is None:
            assert self.cellvit_checkpoint, "cellvit_checkpoint must be set to route to CellViT"
            self._cellvit_client = CellvitClient(
                self.cellvit_checkpoint, self.cellvit_repo, gpu=self.cellvit_gpu,
                magnification=self.cellvit_magnification,
            )
        return self._cellvit_client

    @property
    def deepgleason_client(self) -> DeepGleasonClient:
        if self._deepgleason_client is None:
            self._deepgleason_client = DeepGleasonClient(
                self.deepgleason_repo, self.deepgleason_python, self.deepgleason_model,
            )
        return self._deepgleason_client

    def run(
        self, task_description: str, image_path: str, max_iterations: int = 5,
        output_dir: str = "./manager_agent_output",
        ground_truth_count: int | None = None, ground_truth_labels: np.ndarray | None = None,
        ground_truth_counts_by_type: dict | None = None, ground_truth_class_labels: dict | None = None,
        ground_truth_gleason_score: str | None = None, ground_truth_isup_grade: int | None = None,
        tissue: str | None = None, image_id: str | None = None, slide_path: str | None = None,
        expert_notes: str = "", escalate: bool = True,
    ) -> dict:
        """ground_truth_count (CountGD), ground_truth_labels (StarDist),
        ground_truth_counts_by_type/ground_truth_class_labels (CellViT), and
        ground_truth_gleason_score/ground_truth_isup_grade (DeepGleason) are all optional -- see
        run_countgd_with_feedback/run_stardist_with_feedback/run_cellvit_with_feedback/
        run_deepgleason_with_feedback for what happens when the relevant one is left out (the
        manager still gets a full expert dialogue either way, just without a private
        ground-truth answer backing it -- see module docstring). None of them ever reach the
        manager itself -- they're handed to a private ExpertReasoner instead. tissue (StarDist/
        CellViT, both PanNuke) is extra "expert" context, not a ground truth value on its own.
        image_id defaults to the image's filename stem (see main()) if not given -- required for
        escalation (write_escalation) to fire for an unresolved case; without it, an unaccepted
        result is silently returned with no queued follow-up, for any caller, not just
        train_manager.py. expert_notes (see _apply_expert_notes) are accumulated guidance from
        previously human-resolved escalations, typically loaded from checkpoint.json by a caller
        like train_manager.py or resolve_escalations.py -- empty by default (a live one-off CLI
        call with no checkpoint to load from). escalate=False suppresses escalation entirely
        regardless of image_id -- used by evaluate_manager.py so held-out test results never
        enter the escalation_queue (see run_countgd_with_feedback's docstring for why).

        slide_path is DeepGleason-specific: image_path itself must always be something Qwen's
        vision-language model can actually load for routing/select_agent (a normal-sized image),
        never a raw whole-slide TIFF (often gigapixel) -- so for a WSI task, pass a small
        downsampled preview as image_path and the real slide file as slide_path; only the
        deepgleason branch below uses slide_path (falling back to image_path if omitted, for a
        caller that already has a small enough slide)."""
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        if image_id is None:
            image_id = Path(slide_path if slide_path else image_path).stem

        agent = select_agent(self.qwen, task_description, image_path)
        print(f"[Qwen] routed to: {agent}")

        if agent == "countgd":
            return run_countgd_with_feedback(
                self.qwen, self.countgd_client, image_path, task_description, max_iterations, out_dir,
                ground_truth_count=ground_truth_count, image_id=image_id, expert_notes=expert_notes,
                escalate=escalate,
            )
        if agent == "cellvit":
            return run_cellvit_with_feedback(
                self.qwen, self.cellvit_client, image_path, task_description, max_iterations, out_dir,
                ground_truth_counts_by_type=ground_truth_counts_by_type,
                ground_truth_class_labels=ground_truth_class_labels,
                stardist_worker=self.stardist_worker if ground_truth_class_labels is not None else None,
                tissue=tissue, image_id=image_id, expert_notes=expert_notes, escalate=escalate,
            )
        if agent == "deepgleason":
            return run_deepgleason_with_feedback(
                self.qwen, self.deepgleason_client, slide_path or image_path, task_description, max_iterations,
                out_dir, ground_truth_gleason_score=ground_truth_gleason_score,
                ground_truth_isup_grade=ground_truth_isup_grade, image_id=image_id, expert_notes=expert_notes,
                escalate=escalate,
            )
        return run_stardist_with_feedback(
            self.qwen, self.stardist_worker, image_path, task_description, max_iterations, out_dir,
            ground_truth_labels=ground_truth_labels, tissue=tissue, image_id=image_id, expert_notes=expert_notes,
            escalate=escalate,
        )


def main():
    parser = argparse.ArgumentParser(description="Qwen3-VL-managed dispatch to CountGD, StarDist, CellViT, or DeepGleason")
    parser.add_argument("--image", default=None, help="Path to the input image (ignored if --pannuke-index/--slide is set)")
    parser.add_argument(
        "--slide", default=None,
        help="Path to a whole-slide pathology image (OME-TIFF) for DeepGleason tumor grading -- "
             "mutually exclusive with --image/--pannuke-index. A small routing-preview thumbnail is "
             "generated automatically (Qwen's routing step never gets handed the raw, often-gigapixel "
             "slide directly)",
    )
    parser.add_argument("--task", required=True, help="Task description, e.g. 'count the cells' or 'segment the nuclei'")
    parser.add_argument(
        "--ground-truth-count", type=int, default=None,
        help="Known true count for the image (e.g. from the BBBC005 manifest) -- if set and the "
             "task routes to CountGD, this goes to a private ExpertReasoner instead of the manager; "
             "the manager talks to it and decides accept/reject itself (see module docstring), "
             "instead of falling back to Qwen's own visual score",
    )
    parser.add_argument(
        "--ground-truth-labels", default=None,
        help="Path to a .npy ground-truth instance-label mask -- if set and the task routes to "
             "StarDist, this goes to a private ExpertReasoner instead of the manager; the manager "
             "talks to it and decides accept/reject itself (see module docstring), instead of "
             "falling back to Qwen's own visual score",
    )
    parser.add_argument(
        "--ground-truth-counts-by-type", default=None,
        help="Path to a small JSON file ({\"Neoplastic\": 12, ...}) of known true per-class nucleus "
             "counts -- if set and the task routes to CellViT, this goes to a private ExpertReasoner "
             "instead of the manager, same as --ground-truth-count/--ground-truth-labels for the "
             "other two agents. Per-class instance-level internal_mpq/internal_f1 scoring (not just "
             "counts) additionally needs --pannuke-index (the only source of the raw per-class "
             "instance-label arrays that requires) -- passing this flag alone still enables the "
             "expert dialogue, just without that internal instance-level metric.",
    )
    parser.add_argument(
        "--pannuke-index", type=int, default=None,
        help="Instead of --image/--ground-truth-labels/--ground-truth-counts-by-type, pull one "
             "PanNuke sample (image + its real ground-truth instance mask, per-class counts, and "
             "per-class instance-label arrays) at this index and use it directly",
    )
    parser.add_argument("--pannuke-fold", type=int, default=1, choices=[1, 2, 3], help="PanNuke fold to pull from")
    parser.add_argument("--max-iterations", type=int, default=5)
    parser.add_argument("--output-dir", default="./manager_agent_output")
    parser.add_argument("--model-id", default=MODEL_ID, help="Hugging Face repo id for the manager VLM")
    parser.add_argument("--cellvit-checkpoint", default=None, help="Path to a CellViT model checkpoint (.pth) -- required if the task might route to CellViT")
    parser.add_argument("--cellvit-repo", default=None, help="Path to a local clone of TIO-IKIM/CellViT")
    parser.add_argument("--cellvit-gpu", type=int, default=0, help="CUDA/ROCm GPU id for CellViT inference")
    parser.add_argument("--deepgleason-repo", default=None, help="Path to a local clone of frankkramer-lab/DeepGleason (defaults to ~/DeepGleason, see agentic_deepgleason.py)")
    parser.add_argument("--deepgleason-python", default=None, help="Path to the Python interpreter in DeepGleason's own conda env (defaults to ~/.conda/envs/deepgleason/bin/python)")
    parser.add_argument("--deepgleason-model", default=None, help="Path to a DeepGleason model checkpoint (defaults to <repo>/models/model.ConvNeXtBase.hdf5)")
    parser.add_argument(
        "--ground-truth-gleason-score", default=None,
        help="Known true Gleason score (e.g. '3+4') -- if set and the task routes to DeepGleason, "
             "this goes to a private ExpertReasoner instead of the manager, same as the other agents' "
             "--ground-truth-* flags",
    )
    parser.add_argument(
        "--ground-truth-isup-grade", type=int, default=None,
        help="Known true ISUP grade group (1-5) -- same private-ExpertReasoner treatment as "
             "--ground-truth-gleason-score; also backs the internal_isup_mae logging metric",
    )
    args = parser.parse_args()
    if args.image is None and args.pannuke_index is None and args.slide is None:
        parser.error("one of --image, --pannuke-index, or --slide is required")
    if args.slide is not None and args.pannuke_index is not None:
        parser.error("--slide and --pannuke-index are mutually exclusive")
    if args.max_iterations < 1:
        parser.error("--max-iterations must be at least 1")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manager = ManagerAgent(
        model_id=args.model_id, cellvit_checkpoint=args.cellvit_checkpoint,
        cellvit_repo=args.cellvit_repo, cellvit_gpu=args.cellvit_gpu,
        deepgleason_repo=args.deepgleason_repo, deepgleason_python=args.deepgleason_python,
        deepgleason_model=args.deepgleason_model,
    )

    image_path = args.image
    slide_path = None
    ground_truth_labels = None
    ground_truth_counts_by_type = None
    ground_truth_class_labels = None
    tissue = None
    if args.slide is not None:
        import pyvips  # pyright: ignore[reportMissingImports]
        slide_path = args.slide
        image_path = str(output_dir / "slide_routing_preview.png")
        pyvips.Image.thumbnail(slide_path, 512).write_to_file(image_path)
        print(f"Generated routing preview {image_path} from {slide_path}")
    elif args.pannuke_index is not None:
        print(f"Fetching PanNuke fold {args.pannuke_fold} image {args.pannuke_index}...")
        image, ground_truth_labels, tissue = manager.stardist_worker.load_pannuke_sample(
            args.pannuke_fold, args.pannuke_index
        )
        print(f"tissue={tissue}  ground-truth nuclei={int(ground_truth_labels.max())}")
        _, ground_truth_counts_by_type, ground_truth_class_labels, _ = manager.stardist_worker.load_pannuke_sample_with_classes(
            args.pannuke_fold, args.pannuke_index
        )
        print(f"ground-truth per-class counts={ground_truth_counts_by_type}")
        image_path = str(output_dir / f"pannuke_fold{args.pannuke_fold}_{args.pannuke_index:02d}.png")
        Image.fromarray(image).save(image_path)
    else:
        if args.ground_truth_labels is not None:
            ground_truth_labels = np.load(args.ground_truth_labels)
        if args.ground_truth_counts_by_type is not None:
            ground_truth_counts_by_type = json.loads(Path(args.ground_truth_counts_by_type).read_text())

    assert image_path is not None, "one of --image, --pannuke-index, or --slide is required"
    result = manager.run(
        args.task, image_path, args.max_iterations, args.output_dir,
        ground_truth_count=args.ground_truth_count, ground_truth_labels=ground_truth_labels,
        ground_truth_counts_by_type=ground_truth_counts_by_type,
        ground_truth_class_labels=ground_truth_class_labels,
        ground_truth_gleason_score=args.ground_truth_gleason_score,
        ground_truth_isup_grade=args.ground_truth_isup_grade,
        tissue=tissue, slide_path=slide_path,
    )

    print("\n=== Final result ===")
    print(f"Agent used: {result['agent']}")
    if result["agent"] == "countgd":
        print(f"Count target: {result['count_target']!r}")
        print(f"Predicted count: {result['count']}")
        print(f"Annotated image: {result['annotated_image']}")
    elif result["agent"] == "cellvit":
        print(f"Target classes: {result['target_classes']}")
        print(f"Matched count: {result['count']}")
        print(f"All detected by type: {result['counts_by_type']}")
        print(f"Annotated image: {result['annotated_image']}")
    elif result["agent"] == "deepgleason":
        print(f"Confidence threshold: {result['confidence_threshold']:.2f}")
        print(f"Gleason result: {result['gleason_result']}")
        print(f"Overlay preview: {result['overlay_preview']}")
    else:
        print(f"Detected nuclei: {result['num_nuclei']}")
        print(f"Outlines image: {result['outlines_image']}")
    print(f"History: {json.dumps(result['history'], indent=2)}")


if __name__ == "__main__":
    main()
