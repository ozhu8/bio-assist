"""
Agentic nucleus-segmentation pipeline: Claude orchestrates StarDist.

Structural sibling to agentic_countgd.py, but for StarDist2D instead of
CountGD: StarDist has no free-text "count target" to interpret (it always
segments nuclei), so there's nothing for Claude to do before the first run.
Instead Claude's job is entirely on the evaluation side -- score the outlined
result and, if it's not good enough, propose revised prob_thresh/nms_thresh
(raise prob_thresh for false positives, lower it for missed nuclei; lower
nms_thresh for split/duplicate outlines, raise it for merged ones).

When a PanNuke ground-truth instance mask is available (--pannuke-index),
Panoptic Quality against it decides accept/reject instead of Claude's visual
score, and Claude is only asked to propose threshold revisions -- same
ground-truth-vs-visual-fallback split as agentic_countgd.py would face with
an arbitrary --image and no known answer.

manager_agent.py (this folder's Qwen-driven manager) imports from here --
PRETRAINED_MODEL, run_stardist, load_image, load_pannuke_sample,
compute_panoptic_quality, save_instance_outlines, best_entry, plus the
per-class/diverse-selection additions for CellViT and train/test-split
training (load_pannuke_sample_with_classes, load_pannuke_types,
load_pannuke_samples, load_pannuke_samples_with_classes,
select_diverse_indices, TISSUE_DIVERSITY_MAX_INDEX) -- and drives them with
Qwen instead of Claude, the same way it does with agentic_countgd.py's
run_countgd. Everything else in this file (the Claude orchestration, main())
is this script's own standalone use, mirroring agentic_countgd.py.

Usage:
    python agentic_stardist.py --image tissue.png --prompt "segment the individual nuclei"
    python agentic_stardist.py --pannuke-index 0 --pannuke-fold 1 --prompt "segment the individual nuclei"
"""
import argparse
import json
import random
import textwrap
from pathlib import Path

import anthropic
import numpy as np
from matplotlib import colormaps
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
from skimage.segmentation import find_boundaries
from stardist.models import StarDist2D

from agentic_countgd import MODEL, image_to_content_block

PRETRAINED_MODEL = "2D_versatile_he"  # H&E-stained nuclei -- matches PanNuke's stain type
PDF_NAME = "stardist_results.pdf"
ACCEPT_SCORE_THRESHOLD = 7   # Claude's own 0-10 visual score, used when there's no ground truth
ACCEPT_PQ_THRESHOLD = 0.5    # acceptance bar when a ground-truth instance mask is available

_pannuke_folds = {}  # fold -> loaded HF Dataset, cached so a training run doesn't reload it per-image


def load_image(image_path) -> np.ndarray:
    return np.array(Image.open(image_path).convert("RGB"))


def run_stardist(model: StarDist2D, image: np.ndarray, prob_thresh: float, nms_thresh: float):
    """One StarDist2D call: normalized image -> (labels, details). labels is a
    2D int array (0 = background, 1..N = one integer per detected nucleus)."""
    from csbdeep.utils import normalize
    labels, details = model.predict_instances(
        normalize(image), prob_thresh=prob_thresh, nms_thresh=nms_thresh
    )
    return labels, details


def _get_pannuke_fold(fold: int):
    if fold not in _pannuke_folds:
        from datasets import load_dataset
        _pannuke_folds[fold] = load_dataset("RationAI/PanNuke", split=f"fold{fold}")
    return _pannuke_folds[fold]


def load_pannuke_sample(fold: int, index: int):
    """Pull one PanNuke sample: the H&E image, a StarDist-style instance-label
    mask built from PanNuke's per-nucleus binary masks (each nucleus is its
    own binary image in the 'instances' field -- painted here into a single
    2D array with one integer id per nucleus), and the tissue type name."""
    dataset = _get_pannuke_fold(fold)
    row = dataset[index]

    image = np.array(row["image"].convert("RGB"))
    labels = np.zeros(image.shape[:2], dtype=np.int32)
    for instance_id, mask in enumerate(row["instances"], start=1):
        labels[np.array(mask) > 0] = instance_id

    tissue = dataset.features["tissue"].int2str(row["tissue"])
    return image, labels, tissue


def load_pannuke_samples(fold: int, count: int):
    """Batched counterpart to load_pannuke_sample -- loads samples 0..count-1 from a fold in one
    call. Reuses load_pannuke_sample per row rather than duplicating its instance-mask painting
    logic; the actual win here isn't per-row cost but that _get_pannuke_fold's cache means the
    underlying HF dataset itself is only ever loaded once, regardless of how many times this or
    load_pannuke_sample is called afterward. Returns (images, gt_labels, tissues), each a list of
    length count, aligned by index -- manager_agent.py's diverse-index selection then subsets
    these lists by whichever specific indices it actually wants."""
    images, gt_labels, tissues = [], [], []
    for i in range(count):
        image, labels, tissue = load_pannuke_sample(fold, i)
        images.append(image)
        gt_labels.append(labels)
        tissues.append(tissue)
    return images, gt_labels, tissues


def _pannuke_category_names(dataset) -> list:
    """PanNuke's 'categories' field is a Sequence(ClassLabel) aligned index-for-index with
    'instances' -- one class id per nucleus. Confirmed via the RationAI/PanNuke dataset schema:
    ["Neoplastic", "Inflammatory", "Connective", "Dead", "Epithelial"], the same standard
    ordering agentic_cellvit.NUCLEI_CLASSES already uses."""
    return dataset.features["categories"].feature.names


def pannuke_class_counts(row, dataset) -> dict:
    """Per-class true nucleus counts for one PanNuke row -- {class_name: int count} -- the
    ground_truth_counts_by_type shape run_cellvit_with_feedback's dossier/decision functions
    expect."""
    class_names = _pannuke_category_names(dataset)
    counts = {name: 0 for name in class_names}
    for cat_id in row["categories"]:
        counts[class_names[cat_id]] += 1
    return counts


def pannuke_class_instance_labels(row, dataset) -> dict:
    """Per-class instance-label masks for one PanNuke row -- {class_name: (H, W) int32 ndarray},
    one array per class, each with its own independent 1..N instance numbering -- already in
    compute_panoptic_quality's expected standard label-mask format (see
    _stardist_worker_score_cellvit_predictions in manager_agent.py, which scores CellViT's
    per-class predictions against exactly this)."""
    class_names = _pannuke_category_names(dataset)
    image_shape = np.array(row["image"]).shape[:2]
    labels_by_class = {name: np.zeros(image_shape, dtype=np.int32) for name in class_names}
    next_id = {name: 1 for name in class_names}
    for mask, cat_id in zip(row["instances"], row["categories"]):
        name = class_names[cat_id]
        labels_by_class[name][np.array(mask) > 0] = next_id[name]
        next_id[name] += 1
    return labels_by_class


def load_pannuke_sample_with_classes(fold: int, index: int):
    """CellViT counterpart to load_pannuke_sample -- same image/tissue, but per-class ground
    truth (pannuke_class_counts/pannuke_class_instance_labels) instead of one class-agnostic
    instance mask, since CellViT scores per pathology type, not just object count. Returns
    (image, ground_truth_counts_by_type, ground_truth_class_labels, tissue)."""
    dataset = _get_pannuke_fold(fold)
    row = dataset[index]
    image = np.array(row["image"].convert("RGB"))
    counts = pannuke_class_counts(row, dataset)
    labels_by_class = pannuke_class_instance_labels(row, dataset)
    tissue = dataset.features["tissue"].int2str(row["tissue"])
    return image, counts, labels_by_class, tissue


def load_pannuke_samples_with_classes(fold: int, count: int):
    """Batched counterpart to load_pannuke_sample_with_classes, same relationship as
    load_pannuke_samples has to load_pannuke_sample. Returns (images, class_counts, class_labels,
    tissues), each a list of length count, aligned by index."""
    images, class_counts, class_labels, tissues = [], [], [], []
    for i in range(count):
        image, counts, labels_by_class, tissue = load_pannuke_sample_with_classes(fold, i)
        images.append(image)
        class_counts.append(counts)
        class_labels.append(labels_by_class)
        tissues.append(tissue)
    return images, class_counts, class_labels, tissues


def load_pannuke_types(fold: int) -> list:
    """Tissue-type name for every index in a fold, without loading any images -- used by
    select_diverse_indices to search for tissue diversity cheaply before deciding which (few)
    indices to actually load via load_pannuke_samples/load_pannuke_samples_with_classes."""
    dataset = _get_pannuke_fold(fold)
    tissue_feature = dataset.features["tissue"]
    return [tissue_feature.int2str(t) for t in dataset["tissue"]]


# Caps how many of a fold's indices select_diverse_indices will even consider, so the caller's
# downstream batched load (load_pannuke_samples(fold, selected[-1] + 1) -- a prefix load from
# index 0 through the largest selected index) stays bounded regardless of how large the full
# fold is. Diversity across tissue types is what's being optimized for here, not literally every
# index being reachable, so this only needs to comfortably cover a few thousand candidates --
# select_diverse_indices itself clamps to min(max_index, len(all_types)), so this being larger
# than any given fold's actual size is harmless.
TISSUE_DIVERSITY_MAX_INDEX = 2000


def select_diverse_indices(all_types: list, n: int, max_index: int, seed: int = 0, split: str = "all") -> list:
    """Picks up to n indices from all_types (tissue-type name per dataset index, from
    load_pannuke_types), spread across as many distinct tissue types as possible via
    round-robin, instead of the first n (a single contiguous tissue block covering at most one
    or two tissue types). Only considers indices < max_index (see TISSUE_DIVERSITY_MAX_INDEX).

    split ("all"/"train"/"test") partitions candidate indices by parity (even/odd) before
    selection -- same idea as bbbc005.load_bbbc005_samples's split parameter for CountGD --
    guaranteeing train and test never share an index regardless of n on either side.

    Returns indices sorted ascending -- callers rely on this to treat selected[-1] as the
    batch's own upper bound for a prefix load."""
    candidate_indices = list(range(min(max_index, len(all_types))))
    if split == "train":
        candidate_indices = candidate_indices[0::2]
    elif split == "test":
        candidate_indices = candidate_indices[1::2]
    elif split != "all":
        raise ValueError(f"split must be 'all', 'train', or 'test', got {split!r}")

    by_type: dict = {}
    for idx in candidate_indices:
        by_type.setdefault(all_types[idx], []).append(idx)
    rng = random.Random(seed)
    for indices in by_type.values():
        rng.shuffle(indices)

    selected = []
    type_names = sorted(by_type)  # stable order across calls given the same underlying data
    round_num = 0
    while len(selected) < n:
        added_this_round = False
        for name in type_names:
            if round_num < len(by_type[name]):
                selected.append(by_type[name][round_num])
                added_this_round = True
                if len(selected) == n:
                    break
        if not added_this_round:
            break  # every tissue type's candidates within max_index/split are exhausted
        round_num += 1
    return sorted(selected)


def best_entry(history: list) -> dict:
    """Picks the history entry with the highest pq -- pure logic, no I/O, kept here alongside
    compute_panoptic_quality since callers already import PQ-scoring machinery from this module
    rather than duplicating a second "which iteration was best" rule elsewhere."""
    return max(history, key=lambda e: e["pq"])


def compute_panoptic_quality(pred_labels: np.ndarray, gt_labels: np.ndarray, iou_threshold: float = 0.5) -> dict:
    """Standard Panoptic Quality (Kirillov et al., 'Panoptic Segmentation'):
    PQ = (sum of IoU over matched pairs) / (TP + 0.5*FP + 0.5*FN), where a
    match is any (pred, gt) pair with IoU > iou_threshold. At threshold > 0.5,
    a predicted instance can match at most one ground-truth instance (and
    vice versa) as long as neither label set has overlapping instances with
    itself -- true here, since both are single-label segmentations -- so a
    greedy scan finds the same matching an assignment algorithm would."""
    pred_ids = np.unique(pred_labels)
    pred_ids = pred_ids[pred_ids != 0]
    gt_ids = np.unique(gt_labels)
    gt_ids = gt_ids[gt_ids != 0]

    if len(gt_ids) == 0 and len(pred_ids) == 0:
        return {"pq": 1.0, "mean_iou": 1.0, "tp": 0, "fp": 0, "fn": 0}

    max_pred = int(pred_labels.max()) + 1
    max_gt = int(gt_labels.max()) + 1
    joint = pred_labels.astype(np.int64) * max_gt + gt_labels.astype(np.int64)
    counts = np.bincount(joint.ravel(), minlength=max_pred * max_gt).reshape(max_pred, max_gt)
    pred_areas = np.bincount(pred_labels.ravel(), minlength=max_pred)
    gt_areas = np.bincount(gt_labels.ravel(), minlength=max_gt)

    matched_iou_sum = 0.0
    tp = 0
    matched_gt, matched_pred = set(), set()
    for p in pred_ids:
        for g in gt_ids:
            intersection = counts[p, g]
            if intersection == 0:
                continue
            union = pred_areas[p] + gt_areas[g] - intersection
            iou = intersection / union
            if iou > iou_threshold:
                tp += 1
                matched_iou_sum += iou
                matched_pred.add(p)
                matched_gt.add(g)
                break

    fp = len(pred_ids) - len(matched_pred)
    fn = len(gt_ids) - len(matched_gt)
    mean_iou = matched_iou_sum / tp if tp else 0.0
    denom = tp + 0.5 * fp + 0.5 * fn
    pq = matched_iou_sum / denom if denom else 0.0
    # Cast off numpy scalar types (float64/bool) before this leaves the function --
    # comparing a numpy float downstream (e.g. pq_result["pq"] >= ACCEPT_PQ_THRESHOLD)
    # produces numpy.bool, which json.dumps can't serialize, unlike a real Python bool.
    return {"pq": round(float(pq), 4), "mean_iou": round(float(mean_iou), 4), "tp": tp, "fp": fp, "fn": fn}


def save_instance_outlines(image: np.ndarray, labels: np.ndarray, saved_path) -> None:
    """Draw each instance's boundary in a distinct color over the original
    image and save as PNG -- the annotated image Claude/Qwen evaluates each
    iteration, and the final result image shown in the PDF report."""
    height, width = image.shape[:2]
    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
    ax.imshow(image)
    ax.axis("off")

    instance_ids = [i for i in np.unique(labels) if i != 0]
    cmap = colormaps["hsv"].resampled(max(len(instance_ids), 1))
    for idx, instance_id in enumerate(instance_ids):
        ys, xs = np.nonzero(find_boundaries(labels == instance_id, mode="inner"))
        ax.scatter(xs, ys, s=1, color=cmap(idx), marker=".")

    fig.tight_layout(pad=0)
    fig.savefig(saved_path, dpi=100)
    plt.close(fig)


def evaluate_result(
    claude: anthropic.Anthropic, task_description: str, prob_thresh: float, nms_thresh: float,
    predicted_count: int, outlines_image_path: str, history: list,
) -> dict:
    """No ground truth available -- Claude both scores (0-10) and decides accept/reject by eye."""
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
                        "revised_prob_thresh": {"type": ["number", "null"]},
                        "revised_nms_thresh": {"type": ["number", "null"]},
                    },
                    "required": ["accept", "score", "feedback", "revised_prob_thresh", "revised_nms_thresh"],
                    "additionalProperties": False,
                },
            }
        },
        messages=[{
            "role": "user",
            "content": [
                image_to_content_block(outlines_image_path),
                {"type": "text", "text": (
                    f"Original user request: \"{task_description}\"\n"
                    f"StarDist ran with prob_thresh={prob_thresh:.3f}, nms_thresh={nms_thresh:.3f}\n"
                    f"Detected nuclei: {predicted_count}\n"
                    f"Prior attempts this session: {json.dumps(history)}\n\n"
                    "The attached image shows the original tissue with each StarDist-detected "
                    "nucleus outlined in a distinct color. Evaluate: (1) do the outlines look "
                    "visually accurate (no obvious missed nuclei, false positives, or merged/"
                    "split instances)? (2) is the nucleus count plausible for what's shown? "
                    "(3) does this satisfy the user's original request?\n"
                    "Score 0-10. If score < 7, propose revised threshold(s): raise prob_thresh "
                    "if you see false-positive outlines on background/noise, lower it if real "
                    "nuclei look missed; lower nms_thresh if you see duplicate/split outlines "
                    "around one nucleus, raise it if adjacent distinct nuclei look merged into "
                    "one outline. Only set the threshold(s) that address the problem -- leave "
                    "the other null. Otherwise set accept=true and leave both revised fields null."
                )},
            ],
        }],
    )
    text = next(b.text for b in response.content if b.type == "text")
    return json.loads(text)


def propose_threshold_revision(
    claude: anthropic.Anthropic, task_description: str, prob_thresh: float, nms_thresh: float,
    pq_result: dict, outlines_image_path: str, history: list,
) -> dict:
    """Ground truth available -- PQ decides accept/reject (see main()); Claude
    is only asked to propose better thresholds, not to judge the result."""
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
                        "revised_prob_thresh": {"type": ["number", "null"]},
                        "revised_nms_thresh": {"type": ["number", "null"]},
                        "feedback": {"type": "string"},
                    },
                    "required": ["revised_prob_thresh", "revised_nms_thresh", "feedback"],
                    "additionalProperties": False,
                },
            }
        },
        messages=[{
            "role": "user",
            "content": [
                image_to_content_block(outlines_image_path),
                {"type": "text", "text": (
                    f"Original user request: \"{task_description}\"\n"
                    f"StarDist ran with prob_thresh={prob_thresh:.3f}, nms_thresh={nms_thresh:.3f}\n"
                    f"Panoptic Quality against ground truth: PQ={pq_result['pq']:.3f} "
                    f"(below the {ACCEPT_PQ_THRESHOLD} acceptance bar)\n"
                    f"  mean IoU of matched instances: {pq_result['mean_iou']:.3f}\n"
                    f"  TP={pq_result['tp']}  FP={pq_result['fp']}  FN={pq_result['fn']}\n"
                    f"Prior attempts this session: {json.dumps(history)}\n\n"
                    "The attached image shows the original tissue with each StarDist-detected "
                    "nucleus outlined in a distinct color. FP = spurious detections with no "
                    "matching ground-truth nucleus; FN = ground-truth nuclei StarDist missed; a "
                    "low mean IoU on matched pairs means boundaries are poorly aligned or "
                    "instances are being split/merged. Using both these numbers and the image, "
                    "propose revised threshold(s):\n"
                    "  - prob_thresh (0-1): raise it if FP is high, lower it if FN is high.\n"
                    "  - nms_thresh (0-1): lower it if you see duplicate/split outlines around "
                    "one nucleus; raise it if adjacent distinct nuclei look merged into a single "
                    "outline.\n"
                    "Only set the threshold(s) that address the problem -- leave the other null."
                )},
            ],
        }],
    )
    text = next(b.text for b in response.content if b.type == "text")
    return json.loads(text)


SCORING_RUBRIC = (
    "Each iteration is scored 0-10 by evaluating the outlined image against three criteria:\n"
    "  1. Do the outlines look visually accurate (no obvious missed nuclei, false positives,\n"
    "     or merged/split instances)?\n"
    "  2. Is the nucleus count plausible for what's shown?\n"
    "  3. Does the result satisfy the user's original request?\n\n"
    f"A score >= {ACCEPT_SCORE_THRESHOLD} accepts the result. A score below that triggers a\n"
    "retry with revised prob_thresh/nms_thresh, if a threshold change would plausibly fix the\n"
    "issue. When a PanNuke ground-truth mask is available, Panoptic Quality (PQ) against it\n"
    f"decides accept/reject instead (bar: {ACCEPT_PQ_THRESHOLD})."
)


def save_pdf_report(pdf_path: Path, task_description: str, image_paths: list, history: list) -> None:
    """Render a methodology page, then one page per iteration (outlined image + feedback), into a single PDF."""
    with PdfPages(pdf_path) as pdf:
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis("off")
        ax.set_title("Scoring methodology", fontsize=15, fontweight="bold", loc="left")
        ax.text(0, 0.95, SCORING_RUBRIC, va="top", ha="left", fontsize=11, wrap=True)
        pdf.savefig(fig)
        plt.close(fig)

        for entry, image_path in zip(history, image_paths):
            fig, (ax_img, ax_text) = plt.subplots(2, 1, figsize=(8.5, 11), gridspec_kw={"height_ratios": [4, 1]})
            ax_img.imshow(Image.open(image_path))
            ax_img.axis("off")
            ax_img.set_title(
                f"Iteration {entry['iteration']}: prob_thresh={entry['prob_thresh']:.3f}, "
                f"nms_thresh={entry['nms_thresh']:.3f}"
            )

            ax_text.axis("off")
            score_line = f"PQ: {entry['pq']:.3f}" if "pq" in entry else f"Score: {entry['score']}/10"
            caption = (
                f"Request: {task_description}\n"
                f"Detected nuclei: {entry['predicted_count']}\n"
                f"{score_line}\n"
                f"Feedback: {textwrap.fill(entry['feedback'], 100)}"
            )
            ax_text.text(0, 1, caption, va="top", ha="left", fontsize=10, wrap=True)
            pdf.savefig(fig)
            plt.close(fig)

        final = history[-1]
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis("off")
        ax.set_title("Summary", fontsize=15, fontweight="bold", loc="left")
        score_line = f"Final PQ: {final['pq']:.3f}" if "pq" in final else f"Final score: {final['score']}/10"
        summary = (
            f"Request: {task_description}\n\n"
            f"Iterations run: {len(history)}\n"
            f"Final thresholds: prob_thresh={final['prob_thresh']:.3f}, nms_thresh={final['nms_thresh']:.3f}\n"
            f"Final detected nuclei: {final['predicted_count']}\n"
            f"{score_line}"
        )
        ax.text(0, 0.95, summary, va="top", ha="left", fontsize=11, wrap=True)
        pdf.savefig(fig)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Run StarDist with Claude as evaluator/tuner")
    parser.add_argument("--image", default=None, help="Path to the input image (ignored if --pannuke-index is set)")
    parser.add_argument("--prompt", required=True, help="What the user asked for, e.g. 'segment the individual nuclei'")
    parser.add_argument("--pannuke-index", type=int, default=None, help="Pull this PanNuke sample instead of --image; scores against its real ground truth via PQ")
    parser.add_argument("--pannuke-fold", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--max-iterations", type=int, default=3)
    parser.add_argument("--output-dir", default="./stardist_agent_output")
    parser.add_argument("--pdf-name", default=PDF_NAME)
    args = parser.parse_args()
    if args.image is None and args.pannuke_index is None:
        parser.error("one of --image or --pannuke-index is required")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ground_truth_labels = None
    if args.pannuke_index is not None:
        print(f"Fetching PanNuke fold {args.pannuke_fold} image {args.pannuke_index}...")
        raw_image, ground_truth_labels, tissue = load_pannuke_sample(args.pannuke_fold, args.pannuke_index)
        print(f"tissue={tissue}  ground-truth nuclei={int(ground_truth_labels.max())}")
        image_path = output_dir / f"pannuke_fold{args.pannuke_fold}_{args.pannuke_index:02d}.png"
        Image.fromarray(raw_image).save(image_path)
        args.image = str(image_path)

    claude = anthropic.Anthropic()
    model = StarDist2D.from_pretrained(PRETRAINED_MODEL)
    image = load_image(args.image)
    prob_thresh, nms_thresh = round(model.thresholds.prob, 3), round(model.thresholds.nms, 3)

    history = []
    saved_paths = []
    saved_path = None
    labels = None
    for i in range(1, args.max_iterations + 1):
        print(f"\n--- Iteration {i}: prob_thresh={prob_thresh:.3f}, nms_thresh={nms_thresh:.3f} ---")
        labels, _ = run_stardist(model, image, prob_thresh=prob_thresh, nms_thresh=nms_thresh)
        predicted_count = int(labels.max())
        print(f"[StarDist] nuclei={predicted_count}")

        saved_path = output_dir / f"iteration_{i}.png"
        save_instance_outlines(image, labels, saved_path)
        saved_paths.append(saved_path)

        if ground_truth_labels is not None:
            pq_result = compute_panoptic_quality(labels, ground_truth_labels)
            accept = pq_result["pq"] >= ACCEPT_PQ_THRESHOLD
            if accept:
                feedback, revised_prob, revised_nms = "PQ met the acceptance threshold.", None, None
            else:
                proposal = propose_threshold_revision(
                    claude, args.prompt, prob_thresh, nms_thresh, pq_result, str(saved_path), history
                )
                feedback = proposal["feedback"]
                revised_prob, revised_nms = proposal.get("revised_prob_thresh"), proposal.get("revised_nms_thresh")
            print(f"[metric] PQ={pq_result['pq']:.3f} accept={accept}")
            history.append({
                "iteration": i, "prob_thresh": prob_thresh, "nms_thresh": nms_thresh,
                "predicted_count": predicted_count, "pq": pq_result["pq"], "feedback": feedback,
            })
        else:
            eval_result = evaluate_result(claude, args.prompt, prob_thresh, nms_thresh, predicted_count, str(saved_path), history)
            accept = eval_result["accept"]
            revised_prob, revised_nms = eval_result.get("revised_prob_thresh"), eval_result.get("revised_nms_thresh")
            print(f"[Claude eval] score={eval_result['score']} accept={accept}")
            history.append({
                "iteration": i, "prob_thresh": prob_thresh, "nms_thresh": nms_thresh,
                "predicted_count": predicted_count, "score": eval_result["score"], "feedback": eval_result["feedback"],
            })

        if accept or (revised_prob is None and revised_nms is None):
            break
        if revised_prob is not None:
            prob_thresh = revised_prob
        if revised_nms is not None:
            nms_thresh = revised_nms

    pdf_path = output_dir / args.pdf_name
    save_pdf_report(pdf_path, args.prompt, saved_paths, history)

    print("\n=== Final result ===")
    print(f"Detected nuclei: {int(labels.max())}")
    print(f"Outlines image: {saved_path}")
    print(f"PDF report: {pdf_path}")
    print(f"History: {json.dumps(history, indent=2)}")


if __name__ == "__main__":
    main()
