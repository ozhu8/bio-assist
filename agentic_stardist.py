"""
Agentic nucleus-segmentation pipeline: Claude orchestrates StarDist.

Unlike CountGD (agentic_countgd.py, a hosted Gradio Space taking a free-text
object name) or CellViT (a local checkpoint-based classifier), StarDist
(https://github.com/stardist/stardist) is a local, pretrained instance
segmentation model. Its `2D_versatile_he` weights ship inside the `stardist`
pip package itself, so no separate checkpoint download is needed. The model
takes no text prompt -- it just segments every nucleus it finds -- so there is
no free-text target for Claude to interpret going in (contrast with
`interpret_prompt` in agentic_countgd.py).

The retry loop (`--pannuke-index` / `--pannuke-loop-n`) needs ground truth to
score against, so it runs on PanNuke image(s) rather than an arbitrary
`--image`. Each iteration is scored with real Panoptic Quality (PQ) against
PanNuke's ground-truth instance mask -- not a Claude-judged 0-10 score -- and
if PQ is below the acceptance bar, something proposes a revised StarDist
threshold to retry with (there's no text prompt to revise, so the retry knob
is `prob_thresh` / `nms_thresh` instead). By default that "something" is
`propose_thresholds`, a free deterministic rule over the TP/FP/FN/mean-IoU
breakdown -- no API key, no cost. Pass `--claude-feedback` to have Claude look
at the outlined image and propose the revision instead (costs a small amount
per call). See `compute_panoptic_quality`, `propose_thresholds`,
`evaluate_result`, and `SCORING_RUBRIC` below for exactly what's computed and
what's judged.

`--image` and `--pannuke-n` remain single forward passes with no ground truth
and no Claude evaluation loop -- an arbitrary user image has no annotated
instance mask to compute PQ against.

`--pannuke-n`/`--pannuke-index` pull images straight from the official
PanNuke fold archive (https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke/
fold_{n}.zip) via HTTP range requests, decompressing only enough of the
(large, DEFLATE-compressed) images.npy/masks.npy streams to cover the
requested image(s) -- without downloading the full ~700MB-per-fold zip.
Requires `fsspec`+`aiohttp` in addition to this script's other deps.

Usage:
    python agentic_stardist.py --image tissue.png
    python agentic_stardist.py --pannuke-n 10 --pannuke-fold 1
    python agentic_stardist.py --pannuke-index 0 --pannuke-fold 1 --prompt "segment the individual nuclei"
    python agentic_stardist.py --pannuke-loop-n 5 --pannuke-fold 1
"""
import argparse
import base64
import json
import mimetypes
import random
import textwrap
import zipfile
from pathlib import Path

import anthropic
import fsspec
import matplotlib.pyplot as plt
import numpy as np
import numpy.lib.format as npy_format
from csbdeep.utils import normalize
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
from stardist.models import StarDist2D

MODEL = "claude-opus-4-8"
PRETRAINED_MODEL = "2D_versatile_he"
MASK_NAME = "stardist_mask.npy"
OVERLAY_NAME = "stardist_overlay.png"
OUTLINES_NAME = "stardist_outlines.png"
PDF_NAME = "stardist_results.pdf"
PANNUKE_FOLD_URL = "https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke/fold_{fold}.zip"
# PanNuke stores each fold as contiguous per-tissue blocks, and images.npy/masks.npy are
# DEFLATE streams that can only be read sequentially from the start -- so the cost of
# including a tissue in a sample is set by that tissue's *earliest* index in the fold.
# In fold 1 (2656 images, 19 tissues), capping at 1500 still covers 17 of the 19 tissues
# (only HeadNeck at 2098 and Liver at 2189 fall outside it) while reading ~56% of the fold
# instead of the ~82% a full 19-tissue spread would need. See select_diverse_indices.
TISSUE_DIVERSITY_MAX_INDEX = 1500


def image_to_content_block(image_path: str) -> dict:
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None:
        mime_type = "image/png"
    data = base64.standard_b64encode(Path(image_path).read_bytes()).decode("utf-8")
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": mime_type, "data": data},
    }


def load_image(image_path: str) -> np.ndarray:
    """Load an RGB image as a numpy array (the H&E model expects 3-channel RGB)."""
    return np.array(Image.open(image_path).convert("RGB"))


def unique_path(path: Path) -> Path:
    """Return `path` unchanged if nothing exists there yet; otherwise append _1, _2, ...
    before the suffix until a free name is found, so a re-run never silently overwrites
    a previous PDF report."""
    if not path.exists():
        return path
    n = 1
    while True:
        candidate = path.with_name(f"{path.stem}_{n}{path.suffix}")
        if not candidate.exists():
            return candidate
        n += 1


def _read_exact(fp, total_bytes: int, chunk_size: int = 4 * 1024 * 1024) -> bytes:
    """Read exactly total_bytes from fp in bounded chunks rather than one giant .read()
    call. A single very large read becomes one huge HTTP range request under fsspec's
    HTTP filesystem, and PanNuke's server tends to drop the connection mid-transfer past
    a few hundred MB; smaller chunks keep each underlying request reliable."""
    chunks = []
    remaining = total_bytes
    while remaining > 0:
        chunk = fp.read(min(chunk_size, remaining))
        if not chunk:
            raise IOError(f"unexpected EOF: read {total_bytes - remaining} of {total_bytes} bytes")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _read_npy_header(fp):
    version = npy_format.read_magic(fp)
    if version == (1, 0):
        return npy_format.read_array_header_1_0(fp)
    return npy_format.read_array_header_2_0(fp)


def load_pannuke_images(fold: int, n: int):
    """Fetch just the first n images (+ tissue labels) from an official PanNuke
    fold archive via HTTP range requests: only enough of the DEFLATE-compressed
    images.npy/types.npy streams is decompressed to cover n images, so this
    never downloads the full ~700MB zip."""
    zf = zipfile.ZipFile(fsspec.open(PANNUKE_FOLD_URL.format(fold=fold)).open())

    with zf.open(f"Fold {fold}/images/fold{fold}/types.npy") as tf:
        _, _, dtype = _read_npy_header(tf)
        types = np.frombuffer(tf.read(n * dtype.itemsize), dtype=dtype)[:n]

    with zf.open(f"Fold {fold}/images/fold{fold}/images.npy") as imf:
        shape, _, dtype = _read_npy_header(imf)
        per_image_bytes = int(np.prod(shape[1:])) * dtype.itemsize
        raw = _read_exact(imf, n * per_image_bytes)
        images = np.frombuffer(raw, dtype=dtype).reshape((n,) + shape[1:])

    return images.astype(np.uint8), [str(t) for t in types]


def load_pannuke_types(fold: int) -> list:
    """Fetch every image's tissue-type label for a fold in one shot -- types.npy is a few
    KB, independent of the multi-GB images.npy/masks.npy streams, so this lets diverse
    sampling see the whole fold's tissue layout without paying for image/mask data."""
    zf = zipfile.ZipFile(fsspec.open(PANNUKE_FOLD_URL.format(fold=fold)).open())
    with zf.open(f"Fold {fold}/images/fold{fold}/types.npy") as tf:
        _, _, dtype = _read_npy_header(tf)
        types = np.frombuffer(tf.read(), dtype=dtype)
    return [str(t) for t in types]


def select_diverse_indices(types: list, n: int, max_index: int = None, seed: int = 0) -> list:
    """Pick n indices spanning as many distinct tissue types as possible instead of the
    first n (which in PanNuke is a single contiguous tissue block). max_index bounds how
    deep into the fold to look (see TISSUE_DIVERSITY_MAX_INDEX) -- a tissue whose earliest
    occurrence falls beyond it is simply unavailable to this sample. Round-robins through
    tissues in shuffled order (seeded for reproducibility), taking one index per tissue per
    pass, so the result isn't dominated by whichever tissue has the biggest block. Returns
    indices sorted ascending (the order they'll need to be read from the fold in anyway)."""
    by_tissue = {}
    limit = len(types) if max_index is None else min(max_index + 1, len(types))
    for i in range(limit):
        by_tissue.setdefault(types[i], []).append(i)

    tissue_order = list(by_tissue.keys())
    random.Random(seed).shuffle(tissue_order)

    selected = []
    pass_num = 0
    while len(selected) < n:
        added_this_pass = False
        for tissue in tissue_order:
            if len(selected) >= n:
                break
            pool = by_tissue[tissue]
            if pass_num < len(pool):
                selected.append(pool[pass_num])
                added_this_pass = True
        if not added_this_pass:
            break
        pass_num += 1

    return sorted(selected)


def pannuke_mask_to_instance_labels(mask: np.ndarray) -> np.ndarray:
    """Dataset-specific loader: convert PanNuke's raw mask format into the standard
    label-mask format compute_panoptic_quality expects (see its docstring). Unions
    PanNuke's 5 per-class instance-ID channels (0=Neoplastic, 1=Inflammatory,
    2=Connective, 3=Dead, 4=Epithelial; channel 5 is background, dropped) into one
    instance-ID label image. Each channel numbers its own instances 1..k independently,
    so IDs are reassigned to stay unique across the combined image.

    A loader for another dataset would live here as its own function, ending in this
    same standard format -- compute_panoptic_quality itself has no PanNuke-specific
    knowledge and doesn't need to change."""
    combined = np.zeros(mask.shape[:2], dtype=np.int32)
    next_id = 1
    for c in range(5):
        channel = mask[..., c]
        for inst_id in np.unique(channel):
            if inst_id == 0:
                continue
            combined[channel == inst_id] = next_id
            next_id += 1
    return combined


def load_pannuke_samples(fold: int, n: int):
    """Fetch the first n images + ground-truth instance masks (+ tissue labels) from an
    official PanNuke fold archive in one pass: extends load_pannuke_images with masks.npy,
    reusing the same partial-DEFLATE-decompression trick so this never downloads the full
    ~700MB zip. Returns (images, gt_labels_list, tissue_labels); gt_labels_list is already
    in the standard label-mask format (see pannuke_mask_to_instance_labels)."""
    zf = zipfile.ZipFile(fsspec.open(PANNUKE_FOLD_URL.format(fold=fold)).open())

    with zf.open(f"Fold {fold}/images/fold{fold}/types.npy") as tf:
        _, _, dtype = _read_npy_header(tf)
        types = np.frombuffer(tf.read(n * dtype.itemsize), dtype=dtype)[:n]

    with zf.open(f"Fold {fold}/images/fold{fold}/images.npy") as imf:
        shape, _, dtype = _read_npy_header(imf)
        per_image_bytes = int(np.prod(shape[1:])) * dtype.itemsize
        raw = _read_exact(imf, n * per_image_bytes)
        images = np.frombuffer(raw, dtype=dtype).reshape((n,) + shape[1:]).astype(np.uint8)

    with zf.open(f"Fold {fold}/masks/fold{fold}/masks.npy") as mf:
        shape, _, dtype = _read_npy_header(mf)
        per_mask_bytes = int(np.prod(shape[1:])) * dtype.itemsize
        raw = _read_exact(mf, n * per_mask_bytes)
        masks = np.frombuffer(raw, dtype=dtype).reshape((n,) + shape[1:])

    gt_labels_list = [pannuke_mask_to_instance_labels(masks[i]) for i in range(n)]
    return images, gt_labels_list, [str(t) for t in types]


def load_pannuke_sample(fold: int, index: int):
    """Fetch a single image + ground-truth instance mask (+ tissue label) at `index`."""
    images, gt_labels_list, tissue_labels = load_pannuke_samples(fold, index + 1)
    return images[-1], gt_labels_list[-1], tissue_labels[-1]


def run_stardist(model: StarDist2D, image: np.ndarray, prob_thresh: float = None, nms_thresh: float = None):
    """One StarDist forward pass: normalize the image, predict instance labels.

    prob_thresh/nms_thresh default to the model's own tuned values (None)."""
    normalized = normalize(image, 1, 99.8, axis=(0, 1))
    labels, details = model.predict_instances(normalized, prob_thresh=prob_thresh, nms_thresh=nms_thresh)
    return labels, details


def compute_panoptic_quality(pred_labels: np.ndarray, gt_labels: np.ndarray, iou_threshold: float = 0.5) -> dict:
    """PQ = (mean IoU of matched instance pairs) x (TP / (TP + 0.5*FP + 0.5*FN)).

    Dataset-agnostic: pred_labels and gt_labels must both be in the standard label-mask
    format -- a 2D int array, same shape, background=0, each instance a unique positive
    ID (1..K). Neither array needs to come from any particular dataset; a loader that
    produces this format (e.g. pannuke_mask_to_instance_labels) is all a new ground-truth
    source needs to plug in here.

    Predicted and ground-truth instances are matched by taking, for each ground-truth
    instance, the overlapping predicted instance with highest IoU. A match only counts
    if that IoU >= iou_threshold; at threshold 0.5 this greedy matching is automatically
    one-to-one (two disjoint instances can't both exceed 50% overlap with the same other
    instance), so no separate assignment/optimization step is needed."""
    pred_ids = [i for i in np.unique(pred_labels) if i != 0]
    gt_ids = [i for i in np.unique(gt_labels) if i != 0]

    matched_ious = []
    matched_pred_ids = set()
    for gt_id in gt_ids:
        gt_mask = gt_labels == gt_id
        best_iou, best_pred_id = 0.0, None
        for pred_id in np.unique(pred_labels[gt_mask]):
            if pred_id == 0:
                continue
            pred_mask = pred_labels == pred_id
            intersection = np.count_nonzero(gt_mask & pred_mask)
            union = np.count_nonzero(gt_mask | pred_mask)
            iou = intersection / union if union else 0.0
            if iou > best_iou:
                best_iou, best_pred_id = iou, pred_id
        if best_pred_id is not None and best_iou >= iou_threshold:
            matched_ious.append(best_iou)
            matched_pred_ids.add(best_pred_id)

    tp = len(matched_ious)
    fp = len(pred_ids) - len(matched_pred_ids)
    fn = len(gt_ids) - tp
    mean_iou = sum(matched_ious) / tp if tp else 0.0
    denom = tp + 0.5 * fp + 0.5 * fn
    pq = mean_iou * (tp / denom) if denom else 1.0

    return {"pq": pq, "mean_iou": mean_iou, "tp": tp, "fp": fp, "fn": fn}


def save_overlay(image: np.ndarray, labels: np.ndarray, overlay_path: Path) -> None:
    from skimage.color import label2rgb
    overlay = label2rgb(labels, image=image, bg_label=0)
    Image.fromarray((overlay * 255).astype(np.uint8)).save(overlay_path)


def save_instance_outlines(image: np.ndarray, labels: np.ndarray, outlines_path: Path) -> None:
    """Draw each nucleus's boundary over the original image, one distinct color per instance ID."""
    import matplotlib as mpl
    from skimage.segmentation import find_boundaries

    num_instances = int(labels.max())
    outlined = image.copy()
    colormap = mpl.colormaps["hsv"].resampled(max(num_instances, 1))

    for instance_id in range(1, num_instances + 1):
        boundary = find_boundaries(labels == instance_id, mode="outer")
        color = (np.array(colormap(instance_id - 1)[:3]) * 255).astype(np.uint8)
        outlined[boundary] = color

    Image.fromarray(outlined).save(outlines_path)


def evaluate_result(
    claude: anthropic.Anthropic,
    user_prompt: str,
    prob_thresh: float,
    nms_thresh: float,
    pq_result: dict,
    outlines_image_path: str,
    history: list,
) -> dict:
    """Only called when PQ is below ACCEPT_PQ_THRESHOLD -- Claude doesn't decide accept/reject
    (that's computed directly from ground truth), it only proposes what to try next."""
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
                        "feedback": {"type": "string"},
                        "revised_prob_thresh": {"type": ["number", "null"]},
                        "revised_nms_thresh": {"type": ["number", "null"]},
                    },
                    "required": ["feedback", "revised_prob_thresh", "revised_nms_thresh"],
                    "additionalProperties": False,
                },
            }
        },
        messages=[{
            "role": "user",
            "content": [
                image_to_content_block(outlines_image_path),
                {"type": "text", "text": (
                    f"Original user request: \"{user_prompt}\"\n"
                    f"StarDist ran with prob_thresh={prob_thresh:.3f}, nms_thresh={nms_thresh:.3f}\n"
                    f"Panoptic Quality against ground truth: PQ={pq_result['pq']:.3f} "
                    f"(below the {ACCEPT_PQ_THRESHOLD} acceptance bar)\n"
                    f"  mean IoU of matched instances: {pq_result['mean_iou']:.3f}\n"
                    f"  TP={pq_result['tp']}  FP={pq_result['fp']}  FN={pq_result['fn']}\n"
                    f"Prior attempts this session: {json.dumps(history)}\n\n"
                    "The attached image shows the original tissue with each StarDist-detected "
                    "nucleus outlined in a distinct color. FP = spurious detections with no "
                    "matching ground-truth nucleus; FN = ground-truth nuclei StarDist missed; "
                    "a low mean IoU on matched pairs means boundaries are poorly aligned or "
                    "instances are being split/merged. Using both these numbers and the image, "
                    "propose revised threshold(s) to reduce the error:\n"
                    "  - prob_thresh (0-1): the minimum detection confidence. Raise it if FP is "
                    "high (background/noise outlined as nuclei); lower it if FN is high (real "
                    "nuclei missed).\n"
                    "  - nms_thresh (0-1): how much overlap is tolerated between candidate "
                    "detections before one is suppressed. Lower it if you see duplicate/split "
                    "outlines around one nucleus; raise it if adjacent distinct nuclei look "
                    "merged into a single outline.\n"
                    "Only set the threshold(s) that address the problem the numbers and image "
                    "point to -- leave the other null to keep its current value."
                )},
            ],
        }],
    )
    text = next(b.text for b in response.content if b.type == "text")
    return json.loads(text)


ACCEPT_PQ_THRESHOLD = 0.5

SCORING_RUBRIC = (
    "Each iteration is scored with Panoptic Quality (PQ) against PanNuke's ground-truth\n"
    "instance mask:\n\n"
    "    PQ = (mean IoU of matched instance pairs) x (TP / (TP + 0.5*FP + 0.5*FN))\n\n"
    "Predicted and ground-truth instances are matched by IoU >= 0.5 (this threshold makes\n"
    "matching automatically one-to-one, since two disjoint instances can't both exceed 50%\n"
    "overlap with the same other instance). TP = matched pairs, FP = predicted instances with\n"
    "no ground-truth match, FN = ground-truth instances with no predicted match.\n\n"
    f"A PQ >= {ACCEPT_PQ_THRESHOLD} accepts the result -- computed directly from the ground\n"
    "truth, not judged by Claude. Below that, either Claude (--claude-feedback) or a free,\n"
    "deterministic rule (the default -- see propose_thresholds) looks at the TP/FP/FN/mean-IoU\n"
    "breakdown and proposes a revised prob_thresh and/or nms_thresh to retry with.\n\n"
    "Loops to good result: the number of iterations run before PQ first reached the accept\n"
    "threshold (or the total number run, if none did) -- a measure of how many retries the\n"
    "agentic loop needed, separate from the accuracy of the final result itself."
)


def loops_to_acceptance(history: list) -> tuple:
    """Return (iteration_count, reached) where iteration_count is the 1-indexed
    iteration that first met ACCEPT_PQ_THRESHOLD, or len(history) if none did."""
    for entry in history:
        if entry["pq"] >= ACCEPT_PQ_THRESHOLD:
            return entry["iteration"], True
    return len(history), False


def best_entry(history: list) -> dict:
    """The highest-PQ iteration seen (ties go to the earliest). The loop keeps exploring
    after a non-accepting iteration, which can regress -- e.g. iteration 3 hits PQ=0.40,
    iteration 5 lands at PQ=0.33 and would otherwise be reported as 'final'. Everything
    that reports a final result uses this instead of history[-1]."""
    return max(history, key=lambda e: e["pq"])


PROB_STEP = 0.05
NMS_STEP = 0.05


def _already_tried(value: float, tried: list, tol: float = 0.001) -> bool:
    """Float-tolerant membership check -- exact equality misses e.g. the model's
    unrounded default (0.6924782541382084) matching a rounded proposal (0.692)."""
    return any(abs(value - t) < tol for t in tried)


def propose_thresholds(prob_thresh: float, nms_thresh: float, pq_result: dict, history: list = None) -> tuple:
    """Free, deterministic alternative to evaluate_result -- no API call, no cost. Reasons
    from the same TP/FP/FN/mean-IoU breakdown Claude would have been shown:
      - FP > FN (more spurious detections than misses): raise prob_thresh.
      - FN > FP (more misses than spurious detections): lower prob_thresh.
      - FP == FN but mean IoU is still low (counts line up, boundaries don't): lower
        nms_thresh -- duplicate/split outlines around one nucleus are the more common
        StarDist failure mode once counts already match.
    Two corrections on top of that base rule, both learned from watching it run:
      - Oscillation: a fixed step can bounce forever between two thresholds that each
        flip which error dominates, so the step is halved whenever the plain step would
        revisit an already-tried value (tolerance-based -- see _already_tried).
      - Dead lever: if the previous iteration's threshold change didn't move TP/FP/FN/
        mean-IoU at all (e.g. nms_thresh had no effect because there were no overlapping
        candidate detections to suppress), switch to the other knob instead of repeating
        a change that provably does nothing on this image.
    Returns (revised_prob_thresh, revised_nms_thresh, feedback) with exactly one of the
    two revised values set (the other None)."""
    history = history or []
    fp, fn, mean_iou = pq_result["fp"], pq_result["fn"], pq_result["mean_iou"]
    tried_prob = [e["prob_thresh"] for e in history]
    tried_nms = [e["nms_thresh"] for e in history]

    use_nms = fp == fn
    dead_lever_note = ""
    if history:
        last = history[-1]
        last_had_no_effect = (
            last["tp"] == pq_result["tp"] and last["fp"] == fp and last["fn"] == fn
            and abs(last["mean_iou"] - mean_iou) < 1e-9
        )
        if last_had_no_effect:
            last_changed_nms = abs(last["nms_thresh"] - nms_thresh) > 1e-9
            if last_changed_nms == use_nms:
                use_nms = not use_nms
                dead_lever_note = " Last change had no effect on the prediction -- switching knobs."

    if not use_nms:
        step = PROB_STEP
        sign = 1 if fp >= fn else -1
        revised_prob = round(min(max(prob_thresh + sign * step, 0.05), 0.95), 3)
        while _already_tried(revised_prob, tried_prob) and step > 0.005:
            step /= 2
            revised_prob = round(min(max(prob_thresh + sign * step, 0.05), 0.95), 3)
        comparison = f"FP={fp} > FN={fn}" if fp > fn else f"FN={fn} > FP={fp}" if fn > fp else f"FP == FN == {fp}"
        direction = "raising" if sign > 0 else "lowering"
        feedback = f"{comparison}: {direction} prob_thresh to {revised_prob}.{dead_lever_note}"
        return revised_prob, None, feedback

    step = NMS_STEP
    revised_nms = round(max(nms_thresh - step, 0.01), 3)
    while _already_tried(revised_nms, tried_nms) and step > 0.005:
        step /= 2
        revised_nms = round(max(nms_thresh - step, 0.01), 3)
    feedback = (
        f"FP == FN == {fp} but mean IoU={mean_iou:.3f}: lowering nms_thresh to "
        f"{revised_nms} to reduce duplicate/split detections.{dead_lever_note}"
    )
    return None, revised_nms, feedback


def run_pq_loop(
    claude,
    model: StarDist2D,
    image: np.ndarray,
    gt_labels: np.ndarray,
    prompt: str,
    max_iterations: int,
    output_dir: Path,
    stem: str,
) -> tuple:
    """Run the prob_thresh/nms_thresh retry loop for one image against its ground truth.
    `claude` is an anthropic.Anthropic instance to use Claude for revision suggestions
    (costs a small amount per call), or None to use the free, deterministic
    propose_thresholds instead (the default -- see main()'s --claude-feedback flag).
    Saves one outlined-instances PNG per iteration under output_dir, named
    f"{stem}_iteration_{{i}}.png". Returns (final_labels, history, saved_paths)."""
    prob_thresh, nms_thresh = round(model.thresholds.prob, 3), round(model.thresholds.nms, 3)
    history = []
    saved_paths = []
    labels = None
    for i in range(1, max_iterations + 1):
        print(f"  [{stem}] iteration {i}: prob_thresh={prob_thresh:.3f}, nms_thresh={nms_thresh:.3f}")
        labels, _ = run_stardist(model, image, prob_thresh=prob_thresh, nms_thresh=nms_thresh)
        predicted_count = int(labels.max())
        pq_result = compute_panoptic_quality(labels, gt_labels)
        accept = pq_result["pq"] >= ACCEPT_PQ_THRESHOLD
        print(
            f"    nuclei={predicted_count}  PQ={pq_result['pq']:.3f} "
            f"(mean_iou={pq_result['mean_iou']:.3f}, TP={pq_result['tp']}, "
            f"FP={pq_result['fp']}, FN={pq_result['fn']})"
        )

        saved_path = output_dir / f"{stem}_iteration_{i}.png"
        save_instance_outlines(image, labels, saved_path)
        saved_paths.append(saved_path)

        revised_prob = revised_nms = None
        if accept:
            feedback = "PQ met the acceptance threshold."
        elif claude is not None:
            eval_result = evaluate_result(
                claude, prompt, prob_thresh, nms_thresh, pq_result, str(saved_path), history
            )
            feedback = eval_result["feedback"]
            revised_prob = eval_result.get("revised_prob_thresh")
            revised_nms = eval_result.get("revised_nms_thresh")
            print(f"    [Claude eval] feedback: {feedback}")
        else:
            revised_prob, revised_nms, feedback = propose_thresholds(prob_thresh, nms_thresh, pq_result, history)
            print(f"    [rule-based] {feedback}")

        history.append({
            "iteration": i,
            "prob_thresh": prob_thresh,
            "nms_thresh": nms_thresh,
            "predicted_count": predicted_count,
            "pq": pq_result["pq"],
            "mean_iou": pq_result["mean_iou"],
            "tp": pq_result["tp"],
            "fp": pq_result["fp"],
            "fn": pq_result["fn"],
            "feedback": feedback,
        })

        if accept or (revised_prob is None and revised_nms is None):
            break
        prob_thresh = revised_prob if revised_prob is not None else prob_thresh
        nms_thresh = revised_nms if revised_nms is not None else nms_thresh

    best = best_entry(history)
    if best["iteration"] != history[-1]["iteration"]:
        print(
            f"  [{stem}] search continued past its best result -- reverting to iteration "
            f"{best['iteration']} (PQ={best['pq']:.3f}) instead of the last one tried "
            f"(PQ={history[-1]['pq']:.3f})"
        )
        labels, _ = run_stardist(model, image, prob_thresh=best["prob_thresh"], nms_thresh=best["nms_thresh"])

    return labels, history, saved_paths


def save_pdf_report(
    pdf_path: Path,
    image_path: str,
    outlines_path: Path,
    mask_shape: tuple,
    num_nuclei: int,
) -> None:
    """Render a single page with the original image next to the outlined instance segmentation."""
    with PdfPages(pdf_path) as pdf:
        fig, (ax_orig, ax_out) = plt.subplots(1, 2, figsize=(11, 6))

        ax_orig.imshow(Image.open(image_path).convert("RGB"))
        ax_orig.axis("off")
        ax_orig.set_title("Original image")

        ax_out.imshow(Image.open(outlines_path))
        ax_out.axis("off")
        ax_out.set_title(f"StarDist instances ({num_nuclei} nuclei)")

        fig.suptitle(
            f"StarDist 2D_versatile_he -- {Path(image_path).name}\nMask shape: {mask_shape}",
            fontsize=13,
            fontweight="bold",
        )
        pdf.savefig(fig)
        plt.close(fig)


def save_loop_pdf_report(
    pdf_path: Path,
    user_prompt: str,
    image_paths: list,
    history: list,
) -> None:
    """Render a methodology page, then one page per iteration (outlined image + PQ breakdown), into a single PDF."""
    with PdfPages(pdf_path) as pdf:
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis("off")
        ax.set_title("Scoring methodology", fontsize=15, fontweight="bold", loc="left")
        ax.text(0, 0.95, SCORING_RUBRIC, va="top", ha="left", fontsize=11, wrap=True)
        pdf.savefig(fig)
        plt.close(fig)

        for entry, image_path in zip(history, image_paths):
            fig, (ax_img, ax_text) = plt.subplots(
                2, 1, figsize=(8.5, 11), gridspec_kw={"height_ratios": [4, 1]}
            )
            ax_img.imshow(Image.open(image_path))
            ax_img.axis("off")
            ax_img.set_title(
                f"Iteration {entry['iteration']}: prob_thresh={entry['prob_thresh']:.3f}, "
                f"nms_thresh={entry['nms_thresh']:.3f}"
            )

            ax_text.axis("off")
            caption = (
                f"Request: {user_prompt}\n"
                f"Detected nuclei: {entry['predicted_count']}\n"
                f"PQ: {entry['pq']:.3f}  (mean IoU: {entry['mean_iou']:.3f}, "
                f"TP={entry['tp']}, FP={entry['fp']}, FN={entry['fn']})\n"
                f"Feedback: {textwrap.fill(entry['feedback'], 100)}"
            )
            ax_text.text(0, 1, caption, va="top", ha="left", fontsize=10, wrap=True)

            pdf.savefig(fig)
            plt.close(fig)

        loops, reached = loops_to_acceptance(history)
        final = best_entry(history)
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis("off")
        ax.set_title("Summary", fontsize=15, fontweight="bold", loc="left")
        loops_line = (
            f"Loops to good result: {loops} of {len(history)} iterations run"
            if reached
            else f"Loops to good result: not reached (all {len(history)} iterations scored "
                 f"below PQ {ACCEPT_PQ_THRESHOLD})"
        )
        summary = (
            f"Request: {user_prompt}\n\n"
            f"{loops_line}\n"
            f"Final prob_thresh: {final['prob_thresh']:.3f}\n"
            f"Final nms_thresh: {final['nms_thresh']:.3f}\n"
            f"Final detected nuclei: {final['predicted_count']}\n"
            f"Final PQ: {final['pq']:.3f} (mean IoU: {final['mean_iou']:.3f}, "
            f"TP={final['tp']}, FP={final['fp']}, FN={final['fn']})"
        )
        ax.text(0, 0.95, summary, va="top", ha="left", fontsize=11, wrap=True)
        pdf.savefig(fig)
        plt.close(fig)


def save_multi_loop_pdf_report(pdf_path: Path, user_prompt: str, results: list) -> None:
    """Methodology page, then one page per image (its final outlined result + per-iteration
    PQ trace), then an aggregate summary page across all images. `results` entries need
    index/tissue/history/image_paths/loops/reached (see run_pq_loop + loops_to_acceptance)."""
    with PdfPages(pdf_path) as pdf:
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis("off")
        ax.set_title("Scoring methodology", fontsize=15, fontweight="bold", loc="left")
        ax.text(0, 0.95, SCORING_RUBRIC, va="top", ha="left", fontsize=11, wrap=True)
        pdf.savefig(fig)
        plt.close(fig)

        for r in results:
            history = r["history"]
            final = best_entry(history)
            fig, (ax_img, ax_text) = plt.subplots(
                2, 1, figsize=(8.5, 11), gridspec_kw={"height_ratios": [4, 1]}
            )
            ax_img.imshow(Image.open(r["image_paths"][final["iteration"] - 1]))
            ax_img.axis("off")
            ax_img.set_title(
                f"Image {r['index']} ({r['tissue']}) -- best iteration {final['iteration']}: "
                f"prob_thresh={final['prob_thresh']:.3f}, nms_thresh={final['nms_thresh']:.3f}"
            )

            ax_text.axis("off")
            pq_trace = " -> ".join(f"{e['pq']:.3f}" for e in history)
            caption = (
                f"Request: {user_prompt}\n"
                f"Detected nuclei: {final['predicted_count']}\n"
                f"PQ per iteration: {pq_trace}\n"
                f"Final PQ: {final['pq']:.3f}  (mean IoU: {final['mean_iou']:.3f}, "
                f"TP={final['tp']}, FP={final['fp']}, FN={final['fn']})\n"
                f"Feedback: {textwrap.fill(final['feedback'], 100)}"
            )
            ax_text.text(0, 1, caption, va="top", ha="left", fontsize=10, wrap=True)

            pdf.savefig(fig)
            plt.close(fig)

        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis("off")
        ax.set_title("Summary", fontsize=15, fontweight="bold", loc="left")
        lines = [
            f"Image {r['index']} ({r['tissue']}): final PQ={best_entry(r['history'])['pq']:.3f}, "
            + (f"loops to good result={r['loops']}" if r["reached"] else f"not reached ({r['loops']} iterations run)")
            for r in results
        ]
        mean_pq = sum(best_entry(r["history"])["pq"] for r in results) / len(results)
        reached_count = sum(1 for r in results if r["reached"])
        summary = (
            f"Request: {user_prompt}\n\n"
            + "\n".join(lines)
            + f"\n\nMean final PQ across {len(results)} images: {mean_pq:.3f}\n"
            + f"Reached PQ >= {ACCEPT_PQ_THRESHOLD} acceptance: {reached_count}/{len(results)} images"
        )
        ax.text(0, 0.95, summary, va="top", ha="left", fontsize=10, wrap=True)
        pdf.savefig(fig)
        plt.close(fig)


def save_batch_pdf_report(pdf_path: Path, results: list) -> None:
    """One page per image (original next to outlined instances), plus a summary page."""
    with PdfPages(pdf_path) as pdf:
        for r in results:
            fig, (ax_orig, ax_out) = plt.subplots(1, 2, figsize=(11, 6))

            ax_orig.imshow(Image.open(r["image_path"]))
            ax_orig.axis("off")
            ax_orig.set_title(f"Original ({r['tissue']})")

            ax_out.imshow(Image.open(r["outlines_path"]))
            ax_out.axis("off")
            ax_out.set_title(f"StarDist instances ({r['num_nuclei']} nuclei)")

            fig.suptitle(
                f"PanNuke image {r['index']} -- {Path(r['image_path']).name}\n"
                f"Mask shape: {r['mask_shape']}",
                fontsize=13,
                fontweight="bold",
            )
            pdf.savefig(fig)
            plt.close(fig)

        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis("off")
        ax.set_title("Summary", fontsize=15, fontweight="bold", loc="left")
        lines = [f"Image {r['index']} ({r['tissue']}): {r['num_nuclei']} nuclei" for r in results]
        total = sum(r["num_nuclei"] for r in results)
        summary = "\n".join(lines) + f"\n\nTotal nuclei detected across {len(results)} images: {total}"
        ax.text(0, 0.95, summary, va="top", ha="left", fontsize=11, wrap=True)
        pdf.savefig(fig)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Run StarDist's 2D_versatile_he model on image(s)")
    parser.add_argument("--image", default=None, help="Path to a single input image (single pass, no ground truth)")
    parser.add_argument(
        "--pannuke-n", type=int, default=None,
        help="Pull this many images straight from an official PanNuke fold (single pass per image)",
    )
    parser.add_argument(
        "--pannuke-index", type=int, default=None,
        help="Run the Claude-guided PQ retry loop against one PanNuke image+ground truth at this index",
    )
    parser.add_argument(
        "--pannuke-loop-n", type=int, default=None,
        help="Run the Claude-guided PQ retry loop independently over the first n PanNuke images (indices 0..n-1)",
    )
    parser.add_argument("--pannuke-fold", type=int, default=1, choices=[1, 2, 3], help="PanNuke fold to pull from")
    parser.add_argument(
        "--prompt", default="segment the individual nuclei",
        help="What the segmentation should satisfy (--pannuke-index/--pannuke-loop-n paths only)",
    )
    parser.add_argument("--max-iterations", type=int, default=5, help="--pannuke-index/--pannuke-loop-n paths only")
    parser.add_argument(
        "--claude-feedback", action="store_true",
        help="Use the Claude API to propose revised thresholds (costs a small amount per call). "
             "Default is the free, deterministic propose_thresholds heuristic -- no API key needed.",
    )
    parser.add_argument(
        "--diverse-tissues", action="store_true",
        help="--pannuke-loop-n only: spread the sample across different tissue types instead of "
             "taking the first n images (PanNuke stores each fold as contiguous per-tissue blocks, "
             "so the plain first-n sample is usually all one tissue). See select_diverse_indices.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Shuffle seed for --diverse-tissues")
    parser.add_argument("--output-dir", default="./stardist_agent_output")
    parser.add_argument("--pdf-name", default=PDF_NAME, help="Filename for the saved PDF report")
    args = parser.parse_args()
    if not args.image and not args.pannuke_n and args.pannuke_index is None and not args.pannuke_loop_n:
        parser.error("one of --image, --pannuke-n, --pannuke-index, or --pannuke-loop-n is required")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = StarDist2D.from_pretrained(PRETRAINED_MODEL)

    if args.pannuke_index is not None:
        claude = anthropic.Anthropic() if args.claude_feedback else None
        print(f"Fetching PanNuke fold {args.pannuke_fold} image {args.pannuke_index}...")
        image, gt_labels, tissue = load_pannuke_sample(args.pannuke_fold, args.pannuke_index)
        print(f"tissue={tissue}  ground-truth nuclei={int(gt_labels.max())}")

        stem = f"pannuke_fold{args.pannuke_fold}_{args.pannuke_index:02d}"
        labels, history, saved_paths = run_pq_loop(
            claude, model, image, gt_labels, args.prompt, args.max_iterations, output_dir, stem
        )

        mask_path = output_dir / MASK_NAME
        np.save(mask_path, labels)

        overlay_path = output_dir / OVERLAY_NAME
        save_overlay(image, labels, overlay_path)

        pdf_path = unique_path(output_dir / args.pdf_name)
        save_loop_pdf_report(pdf_path, args.prompt, saved_paths, history)

        final = best_entry(history)
        print("\n=== Final result ===")
        print(f"Mask shape: {labels.shape}")
        print(f"Detected nuclei: {final['predicted_count']}")
        print(f"Final PQ: {final['pq']:.3f}")
        print(f"Mask saved: {mask_path}")
        print(f"Overlay saved: {overlay_path}")
        print(f"Outlined image: {saved_paths[final['iteration'] - 1]}")
        print(f"PDF report saved: {pdf_path}")
        print(f"History: {json.dumps(history, indent=2)}")
        return

    if args.pannuke_loop_n:
        claude = anthropic.Anthropic() if args.claude_feedback else None
        if args.diverse_tissues:
            print(f"Fetching tissue-type layout for PanNuke fold {args.pannuke_fold}...")
            all_types = load_pannuke_types(args.pannuke_fold)
            selected = select_diverse_indices(
                all_types, args.pannuke_loop_n, max_index=TISSUE_DIVERSITY_MAX_INDEX, seed=args.seed
            )
            n_tissues = len(set(all_types[i] for i in selected))
            print(
                f"Selected {len(selected)} images spanning {n_tissues} tissue types "
                f"(fold indices {selected[0]}-{selected[-1]}, {100 * (selected[-1] + 1) / len(all_types):.0f}% of the fold)"
            )
            images_prefix, gt_labels_prefix, tissue_prefix = load_pannuke_samples(
                args.pannuke_fold, selected[-1] + 1
            )
            image_indices = selected
            images = [images_prefix[i] for i in selected]
            gt_labels_list = [gt_labels_prefix[i] for i in selected]
            tissue_labels = [tissue_prefix[i] for i in selected]
        else:
            print(f"Fetching {args.pannuke_loop_n} images + ground truth from PanNuke fold {args.pannuke_fold}...")
            images, gt_labels_list, tissue_labels = load_pannuke_samples(args.pannuke_fold, args.pannuke_loop_n)
            image_indices = list(range(args.pannuke_loop_n))

        results = []
        for idx, image, gt_labels, tissue in zip(image_indices, images, gt_labels_list, tissue_labels):
            stem = f"pannuke_fold{args.pannuke_fold}_{idx:04d}"
            print(f"\n=== Image {idx} ({tissue}), ground-truth nuclei={int(gt_labels.max())} ===")
            labels, history, saved_paths = run_pq_loop(
                claude, model, image, gt_labels, args.prompt, args.max_iterations, output_dir, stem
            )
            np.save(output_dir / f"{stem}_mask.npy", labels)

            loops, reached = loops_to_acceptance(history)
            print(
                f"[{idx}] tissue={tissue} final PQ={best_entry(history)['pq']:.3f} "
                f"({'reached' if reached else 'did not reach'} PQ>={ACCEPT_PQ_THRESHOLD} "
                f"after {loops} iterations)"
            )
            results.append({
                "index": idx, "tissue": tissue, "history": history,
                "image_paths": saved_paths, "loops": loops, "reached": reached,
            })

        pdf_path = unique_path(output_dir / args.pdf_name)
        save_multi_loop_pdf_report(pdf_path, args.prompt, results)

        mean_pq = sum(best_entry(r["history"])["pq"] for r in results) / len(results)
        reached_count = sum(1 for r in results if r["reached"])
        print("\n=== StarDist PQ-loop batch result ===")
        print(f"Images processed: {len(results)}")
        print(f"Mean final PQ: {mean_pq:.3f}")
        print(f"Reached PQ >= {ACCEPT_PQ_THRESHOLD}: {reached_count}/{len(results)}")
        print(f"PDF report saved: {pdf_path}")
        return

    if args.pannuke_n:
        print(f"Fetching {args.pannuke_n} images from PanNuke fold {args.pannuke_fold}...")
        images, tissue_labels = load_pannuke_images(args.pannuke_fold, args.pannuke_n)

        results = []
        for idx, (image, tissue) in enumerate(zip(images, tissue_labels)):
            stem = f"pannuke_fold{args.pannuke_fold}_{idx:02d}"
            image_path = output_dir / f"{stem}.png"
            Image.fromarray(image).save(image_path)

            labels, details = run_stardist(model, image)
            num_nuclei = int(labels.max())
            np.save(output_dir / f"{stem}_mask.npy", labels)

            outlines_path = output_dir / f"{stem}_outlines.png"
            save_instance_outlines(image, labels, outlines_path)

            print(f"[{idx}] tissue={tissue} nuclei={num_nuclei}")
            results.append({
                "index": idx, "tissue": tissue, "image_path": image_path,
                "outlines_path": outlines_path, "mask_shape": labels.shape, "num_nuclei": num_nuclei,
            })

        pdf_path = unique_path(output_dir / args.pdf_name)
        save_batch_pdf_report(pdf_path, results)

        print("\n=== StarDist batch result ===")
        print(f"Images processed: {len(results)}")
        print(f"Total nuclei detected: {sum(r['num_nuclei'] for r in results)}")
        print(f"PDF report saved: {pdf_path}")
        return

    image = load_image(args.image)
    labels, details = run_stardist(model, image)
    num_nuclei = int(labels.max())

    mask_path = output_dir / MASK_NAME
    np.save(mask_path, labels)

    overlay_path = output_dir / OVERLAY_NAME
    save_overlay(image, labels, overlay_path)

    outlines_path = output_dir / OUTLINES_NAME
    save_instance_outlines(image, labels, outlines_path)

    pdf_path = unique_path(output_dir / args.pdf_name)
    save_pdf_report(pdf_path, args.image, outlines_path, labels.shape, num_nuclei)

    print("=== StarDist result ===")
    print(f"Mask shape: {labels.shape}")
    print(f"Detected nuclei: {num_nuclei}")
    print(f"Mask saved: {mask_path}")
    print(f"Overlay saved: {overlay_path}")
    print(f"Instance outlines saved: {outlines_path}")
    print(f"PDF report saved: {pdf_path}")


if __name__ == "__main__":
    main()
