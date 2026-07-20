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

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
ACCEPT_SCORE_THRESHOLD = 7  # Qwen's own 0-10 visual score, used when there's no ground truth to measure against
ACCEPT_PQ_THRESHOLD = 0.5   # old PQ rule, kept only to log internal_would_accept -- see module docstring
MAE_TOLERANCE_FRACTION = 0.1  # old MAE rule, kept only to log internal_would_accept -- see module docstring
MAX_EXPERT_TURNS = 3  # cap on manager<->expert question/answer turns per iteration


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
        f"Prior attempts this session: {json.dumps(history, default=str)}\n\n"
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
        f"Prior attempts this session: {json.dumps(history, default=str)}\n\n"
        "Based on the expert's reasoning and what you can see in the image yourself, decide "
        "whether this result satisfies the request. If not, propose a different/more specific "
        "text prompt for CountGD to retry with, informed by what the expert pointed out (e.g. "
        "overlapping cells, faint/out-of-focus cells, background clutter).\n\n"
        "Reply with ONLY a JSON object matching this schema: "
        "{\"accept\": bool, \"feedback\": str, \"revised_text\": str or null}"
    )
    return qwen.ask_json(annotated_image_path, prompt, required_keys=["accept", "feedback"])


def evaluate_stardist_visual(
    qwen: QwenVLM, task_description: str, prob_thresh: float, nms_thresh: float,
    predicted_count: int, outlines_image_path: str, history: list,
) -> dict:
    """No ground truth available -- Qwen both scores (0-10) and decides accept/reject by eye."""
    prompt = (
        f"Original user request: \"{task_description}\"\n"
        f"StarDist ran with prob_thresh={prob_thresh:.3f}, nms_thresh={nms_thresh:.3f}\n"
        f"Detected nuclei: {predicted_count}\n"
        f"Prior attempts this session: {json.dumps(history, default=str)}\n\n"
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
        f"Prior attempts this session: {json.dumps(history, default=str)}\n\n"
        "Based on the expert's reasoning and what you can see in the image yourself, decide "
        "whether this segmentation satisfies the request. Weigh both how accurate the existing "
        "outlines are AND how complete the coverage is -- if the dialogue turned up spots that "
        "look like real objects with no outline at all, that's a real failure even if every "
        "existing outline is clean; don't accept just because what WAS detected looks correct. "
        "If not accepted, propose revised threshold(s): raise prob_thresh if the expert's "
        "answers suggest false-positive outlines on background/noise, lower it if real nuclei "
        "sound missed (including the unmarked-spot cases above); lower nms_thresh if outlines "
        "sound duplicated/split around one nucleus, raise it if adjacent distinct nuclei sound "
        "merged. Only set the threshold(s) that address the problem -- leave the other null.\n\n"
        "Reply with ONLY a JSON object matching this schema: {\"accept\": bool, \"feedback\": str, "
        "\"revised_prob_thresh\": number or null, \"revised_nms_thresh\": number or null}"
    )
    return qwen.ask_json(outlines_image_path, prompt, required_keys=["accept", "feedback"])


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


def run_countgd_with_feedback(
    qwen: QwenVLM, countgd_client: Client, image_path: str, task_description: str,
    max_iterations: int, output_dir: Path, ground_truth_count: int | None = None,
) -> dict:
    """If ground_truth_count is given, it goes to a private ExpertReasoner (never to the
    manager) -- each iteration the manager gets a dialogue with it (run_expert_dialogue) and
    then decides accept/reject itself from the transcript (decide_countgd_from_dialogue). The
    old MAE rule is still computed for history/logging only (internal_mae/internal_would_accept
    -- see module docstring), not used for the decision. Otherwise (no ground truth) there's no
    expert to consult, so Qwen scores the result visually and decides accept/reject itself
    (evaluate_countgd_visual)."""
    count_target = interpret_countgd_target(qwen, task_description, image_path)
    print(f"[Qwen] counting target: {count_target!r}")

    expert = None
    if ground_truth_count is not None:
        expert = ExpertReasoner(
            qwen, EXPERT_PERSONA_COUNTGD, build_countgd_dossier(ground_truth_count, image_path),
            forbidden_values=[ground_truth_count],
        )

    history = []
    saved_path = None
    predicted_count = None
    for i in range(1, max_iterations + 1):
        print(f"--- Iteration {i}: CountGD counting {count_target!r} ---")
        annotated_path, predicted_count = run_countgd(countgd_client, image_path, count_target)
        saved_path = output_dir / f"countgd_iteration_{i}.png"
        saved_path.write_bytes(Path(annotated_path).read_bytes())
        print(f"[CountGD] count={predicted_count}")

        if expert is not None:
            dialogue = run_expert_dialogue(
                qwen, expert, task_description, f"predicted count = {predicted_count}",
                str(saved_path), [image_path],
            )
            decision = decide_countgd_from_dialogue(
                qwen, task_description, count_target, predicted_count, dialogue, str(saved_path), history,
            )
            accept, revised_text, feedback = decision["accept"], decision.get("revised_text"), decision["feedback"]
            assert ground_truth_count is not None, "ground_truth_count must not be None if expert is not None"
            internal_mae = abs(predicted_count - ground_truth_count)
            internal_would_accept = internal_mae <= mae_accept_tolerance(ground_truth_count)
            print(f"[manager] accept={accept}  (internal_mae={internal_mae}, old-rule would_accept={internal_would_accept})")
            history.append({
                "iteration": i, "count_target": count_target, "predicted_count": predicted_count,
                "dialogue": dialogue, "accept": accept, "feedback": feedback,
                "internal_mae": internal_mae, "internal_would_accept": internal_would_accept,
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
    return image, round(_worker_model.thresholds.prob, 3), round(_worker_model.thresholds.nms, 3)


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
    return {"labels": labels, "pq_result": pq_result}


def _stardist_worker_load_pannuke(fold: int, index: int):
    """Runs inside the spawned subprocess (load_pannuke_sample lives in agentic_stardist.py too)."""
    from agentic_stardist import load_pannuke_sample
    return load_pannuke_sample(fold, index)


def _stardist_worker_load_pannuke_diverse(fold: int, n: int, seed: int = 0):
    """Runs inside the spawned subprocess. Picks n indices spread across as many distinct
    PanNuke tissue types as possible (agentic_stardist.select_diverse_indices) instead of the
    first n (a single contiguous tissue block), then fetches all of them in one batched
    load_pannuke_samples call rather than n separate from-scratch reads."""
    from agentic_stardist import TISSUE_DIVERSITY_MAX_INDEX, load_pannuke_samples, load_pannuke_types, select_diverse_indices
    all_types = load_pannuke_types(fold)
    selected = select_diverse_indices(all_types, n, max_index=TISSUE_DIVERSITY_MAX_INDEX, seed=seed)
    images_prefix, gt_labels_prefix, tissue_prefix = load_pannuke_samples(fold, selected[-1] + 1)
    return (
        selected,
        [images_prefix[i] for i in selected],
        [gt_labels_prefix[i] for i in selected],
        [tissue_prefix[i] for i in selected],
    )


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

    def load_pannuke_diverse(self, fold: int, n: int, seed: int = 0):
        return self._pool.submit(_stardist_worker_load_pannuke_diverse, fold, n, seed).result()

    def revert_to_best(self, image: np.ndarray, history: list, outlines_path: Path):
        return self._pool.submit(_stardist_worker_revert_to_best, image, history, outlines_path).result()

    def save_gt_outlines(self, image: np.ndarray, gt_labels: np.ndarray, outlines_path: Path):
        return self._pool.submit(_stardist_worker_save_gt_outlines, image, gt_labels, outlines_path).result()

    def shutdown(self):
        self._pool.shutdown(wait=True)


def run_stardist_with_feedback(
    qwen: QwenVLM, worker: StardistWorker, image_path: str, task_description: str,
    max_iterations: int, output_dir: Path, ground_truth_labels: np.ndarray | None = None, tissue: str | None = None,
) -> dict:
    """If ground_truth_labels is given (a PanNuke-style instance mask), it goes to a private
    ExpertReasoner (never to the manager) along with a one-time outline rendering of the true
    instance boundaries and, if known, the PanNuke tissue type. Each iteration the manager gets
    a dialogue with it (run_expert_dialogue) and then decides accept/reject itself from the
    transcript (decide_stardist_from_dialogue). The old PQ rule is still computed for
    history/logging only (internal_pq/internal_would_accept -- see module docstring), not used
    for the decision. Otherwise (no ground truth) there's no expert to consult, so Qwen scores
    the result visually and decides accept/reject itself (evaluate_stardist_visual)."""
    image, prob_thresh, nms_thresh = worker.init(image_path)

    gt_outlines_path = output_dir / "stardist_ground_truth.png"
    expert = None
    if ground_truth_labels is not None:
        worker.save_gt_outlines(image, ground_truth_labels, gt_outlines_path)
        expert = ExpertReasoner(
            qwen, EXPERT_PERSONA_STARDIST, build_stardist_dossier(ground_truth_labels, tissue),
            forbidden_values=[int(ground_truth_labels.max())],
        )

    history = []
    saved_path = None
    labels = None
    for i in range(1, max_iterations + 1):
        print(f"--- Iteration {i}: StarDist prob_thresh={prob_thresh:.3f}, nms_thresh={nms_thresh:.3f} ---")
        saved_path = output_dir / f"stardist_iteration_{i}.png"
        result = worker.run(image, prob_thresh, nms_thresh, ground_truth_labels, saved_path)
        labels = result["labels"]
        predicted_count = int(labels.max())
        print(f"[StarDist] nuclei={predicted_count}")

        if expert is not None:
            dialogue = run_expert_dialogue(
                qwen, expert, task_description,
                f"detected nuclei = {predicted_count} (prob_thresh={prob_thresh:.3f}, nms_thresh={nms_thresh:.3f})",
                str(saved_path), [str(saved_path), str(gt_outlines_path)],
            )
            decision = decide_stardist_from_dialogue(
                qwen, task_description, prob_thresh, nms_thresh, predicted_count, dialogue, str(saved_path), history,
            )
            accept = decision["accept"]
            revised_prob, revised_nms, feedback = decision.get("revised_prob_thresh"), decision.get("revised_nms_thresh"), decision["feedback"]
            pq_result = result["pq_result"]
            internal_would_accept = bool(pq_result["pq"] >= ACCEPT_PQ_THRESHOLD)
            tp, fn = pq_result["tp"], pq_result["fn"]
            internal_recall = tp / (tp + fn) if (tp + fn) else 1.0  # coverage only -- see module docstring
            print(f"[manager] accept={accept}  (internal_pq={pq_result['pq']:.3f}, "
                  f"internal_recall={internal_recall:.3f}, old-rule would_accept={internal_would_accept})")
            history.append({
                # pq/mean_iou/tp/fp/fn keep agentic_stardist.py's own field names (unprefixed) --
                # best_entry() below (imported from that untouched module) keys off e["pq"] to pick
                # which attempted iteration to report as final. None of this is shown to the
                # manager or the expert; only internal_would_accept/internal_recall are new (the
                # old threshold rule, and a coverage-only stat -- tp/(tp+fn), how much of the true
                # object set was even found regardless of outline quality -- logged for comparison
                # against the manager's dialogue-driven `accept` above, never used to decide it).
                "iteration": i, "prob_thresh": prob_thresh, "nms_thresh": nms_thresh,
                "predicted_count": predicted_count, "dialogue": dialogue, "accept": accept, "feedback": feedback,
                "pq": pq_result["pq"], "mean_iou": pq_result["mean_iou"],
                "tp": pq_result["tp"], "fp": pq_result["fp"], "fn": pq_result["fn"],
                "internal_would_accept": internal_would_accept, "internal_recall": internal_recall,
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
        revert = worker.revert_to_best(image, history, output_dir / "stardist_best.png")
        if revert is not None:
            print(
                f"  search continued past its best result -- reverting to iteration "
                f"{revert['best_iteration']} (PQ={revert['best_pq']:.3f}) instead of the last one "
                f"tried (PQ={history[-1]['pq']:.3f})"
            )
            labels = revert["labels"]
            saved_path = output_dir / "stardist_best.png"

    assert labels is not None, "max_iterations must be at least 1"
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
        self._stardist_worker = None

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

    def run(
        self, task_description: str, image_path: str, max_iterations: int = 3,
        output_dir: str = "./manager_agent_output",
        ground_truth_count: int | None = None, ground_truth_labels: np.ndarray | None = None, tissue: str | None = None,
    ) -> dict:
        """ground_truth_count (used only if routed to CountGD) and ground_truth_labels (used only
        if routed to StarDist) are both optional -- see run_countgd_with_feedback/
        run_stardist_with_feedback for what happens when the relevant one is left out. Neither
        ever reaches the manager itself -- see the module docstring: they're handed to a private
        ExpertReasoner instead. tissue (StarDist/PanNuke only) is extra "expert" context, not a
        ground truth value on its own."""
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
            self.qwen, self.stardist_worker, image_path, task_description, max_iterations, out_dir,
            ground_truth_labels=ground_truth_labels, tissue=tissue,
        )


def main():
    parser = argparse.ArgumentParser(description="Qwen3-VL-managed dispatch to CountGD or StarDist")
    parser.add_argument("--image", default=None, help="Path to the input image (ignored if --pannuke-index is set)")
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
    if args.max_iterations < 1:
        parser.error("--max-iterations must be at least 1")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manager = ManagerAgent(model_id=args.model_id)

    image_path = args.image
    ground_truth_labels = None
    tissue = None
    if args.pannuke_index is not None:
        print(f"Fetching PanNuke fold {args.pannuke_fold} image {args.pannuke_index}...")
        image, ground_truth_labels, tissue = manager.stardist_worker.load_pannuke_sample(
            args.pannuke_fold, args.pannuke_index
        )
        print(f"tissue={tissue}  ground-truth nuclei={int(ground_truth_labels.max())}")
        image_path = str(output_dir / f"pannuke_fold{args.pannuke_fold}_{args.pannuke_index:02d}.png")
        Image.fromarray(image).save(image_path)
    elif args.ground_truth_labels is not None:
        ground_truth_labels = np.load(args.ground_truth_labels)

    assert image_path is not None, "one of --image or --pannuke-index is required"
    result = manager.run(
        args.task, image_path, args.max_iterations, args.output_dir,
        ground_truth_count=args.ground_truth_count, ground_truth_labels=ground_truth_labels, tissue=tissue,
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
