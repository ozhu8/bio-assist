"""
Agentic cell-tracking pipeline: Claude orchestrates btrack.

Unlike CountGD (agentic_countgd.py) and StarDist (agentic_stardist.py), which each
take a single image and produce one result, btrack (https://github.com/quantumjot/btrack)
is a multi-object *tracker* -- it links already-segmented objects across a sequence
of frames into trajectories. So this script needs two stages per run, not one:
per-frame instance segmentation (StarDist, reusing `run_stardist` from
agentic_stardist.py unchanged -- same "only imports existing functions" approach
manager_agent.py uses), then btrack linking those per-frame instances into tracks
across time. There is no free-text target for Claude to interpret going in, same
as StarDist.

The retry loop (`--ctc-dataset`/`--ctc-sequence`) needs ground truth to score
against, so it runs on a Cell Tracking Challenge (http://celltrackingchallenge.net/)
training sequence rather than an arbitrary `--images-dir`. CTC ships each training
sequence with a `*_GT/TRA/` folder of per-frame instance masks where, unlike a
plain segmentation mask, the same track keeps the same label across every frame it
appears in (mitosis gives daughters fresh labels) -- so the ground-truth track ID
for an object is just the pixel value itself, no separate lineage parsing needed.

Each iteration is scored against that ground truth with a simplified tracking-link
accuracy (not the official CTC TRA/AOGM metric -- that's a graph edit distance,
overkill for driving a retry loop) -- see `compute_link_accuracy` and
`SCORING_RUBRIC` below. If it's below the acceptance bar, something proposes a
revised `max_search_radius` to retry with (btrack's gating distance for candidate
links -- the one knob StarDist's prob_thresh/nms_thresh don't have an analogue
for). By default that "something" is `propose_search_radius`, a free deterministic
rule over the switch/broken-link counts -- no API key, no cost. Pass
`--claude-feedback` to have Claude look at the trajectory plot and propose the
revision instead (costs a small amount per call).

Changing max_search_radius only affects the linking stage, not the per-frame
segmentation -- so unlike StarDist's retry loop (which reruns the whole model each
iteration), this one runs StarDist once and reuses the cached per-frame instance
labels across iterations, only rerunning btrack itself.

`--images-dir` remains a single forward pass with no ground truth and no Claude
evaluation loop -- an arbitrary user image sequence has no annotated track IDs to
score against.

CTC training sequences are pulled as a whole zip (a few hundred MB -- an order of
magnitude smaller than a PanNuke fold, so unlike agentic_stardist.py's partial-read
trick for PanNuke, this just downloads and caches the zip, then reads only the
frames + `*_GT/TRA` members it needs out of it).

Usage:
    python agentic_btrack.py --images-dir ./my_timelapse_frames
    python agentic_btrack.py --ctc-dataset Fluo-N2DL-HeLa --ctc-sequence 01
    python agentic_btrack.py --ctc-dataset Fluo-N2DL-HeLa --n-frames 20 --claude-feedback
"""
import argparse
import base64
import io
import json
import mimetypes
import re
import textwrap
import urllib.request
import zipfile
from pathlib import Path

import anthropic
import btrack
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
from stardist.models import StarDist2D

from agentic_stardist import run_stardist

MODEL = "claude-opus-4-8"
STARDIST_PRETRAINED_MODEL = "2D_versatile_fluo"  # CTC sequences are grayscale fluorescence/phase, not H&E
TRACKS_NAME = "btrack_tracks.h5"
TRAJECTORY_NAME = "btrack_trajectories.png"
PDF_NAME = "btrack_results.pdf"
CTC_URL_TEMPLATE = "https://data.celltrackingchallenge.net/training-datasets/{dataset}.zip"


def image_to_content_block(image_path: str) -> dict:
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None:
        mime_type = "image/png"
    data = base64.standard_b64encode(Path(image_path).read_bytes()).decode("utf-8")
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": mime_type, "data": data},
    }


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


def load_image_sequence(images_dir: str) -> tuple:
    """Load every image file in a directory, sorted by name, as a list of 2D arrays."""
    paths = sorted(
        p for p in Path(images_dir).iterdir()
        if p.suffix.lower() in (".tif", ".tiff", ".png", ".jpg", ".jpeg")
    )
    images = [np.array(Image.open(p).convert("L")) if p.suffix.lower() not in (".tif", ".tiff")
              else tifffile.imread(p) for p in paths]
    return images, [p.name for p in paths]


def download_ctc_zip(dataset: str, cache_dir: Path) -> Path:
    """Download a Cell Tracking Challenge training-dataset zip once and cache it --
    at ~100-200MB per dataset this is small enough to fetch whole, unlike PanNuke's
    multi-GB folds (see agentic_stardist.py's partial-read trick for that case)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    zip_path = cache_dir / f"{dataset}.zip"
    if not zip_path.exists():
        url = CTC_URL_TEMPLATE.format(dataset=dataset)
        print(f"Downloading {url} ...")
        urllib.request.urlretrieve(url, zip_path)
    return zip_path


_FRAME_INDEX_RE = re.compile(r"(\d+)\.tif$")


def _frame_index(name: str) -> int:
    return int(_FRAME_INDEX_RE.search(name).group(1))


def load_ctc_sequence(dataset: str, sequence: str, cache_dir: Path, n_frames: int = None) -> tuple:
    """Fetch one CTC training sequence's raw frames and ground-truth track masks.

    Ground truth comes from `{dataset}/{sequence}_GT/TRA/man_track*.tif`: CTC's
    tracking (not segmentation) ground truth, where each frame's pixel labels ARE
    the track IDs directly -- a track keeps the same label across every frame it's
    present in, and mitosis gives daughters fresh labels. This means no lineage
    file needs to be read to build ground truth for the link-accuracy metric (see
    compute_link_accuracy): a parent/daughter pair simply never shares a label
    across consecutive frames, so it's automatically excluded from "GT links"
    without any special-casing.

    Matches raw frames to GT frames by the shared numeric frame index in their
    filenames rather than assuming equal counts -- some CTC sequences have sparser
    GT coverage than raw frames. Returns (images, gt_labels) as equal-length lists
    of 2D arrays, restricted to the first n_frames common indices if given."""
    zip_path = download_ctc_zip(dataset, cache_dir)
    with zipfile.ZipFile(zip_path) as zf:
        img_prefix = f"{dataset}/{sequence}/"
        gt_prefix = f"{dataset}/{sequence}_GT/TRA/man_track"
        img_by_idx = {
            _frame_index(n): n for n in zf.namelist()
            if n.startswith(img_prefix) and n.endswith(".tif")
        }
        gt_by_idx = {
            _frame_index(n): n for n in zf.namelist()
            if n.startswith(gt_prefix) and n.endswith(".tif")
        }
        common = sorted(set(img_by_idx) & set(gt_by_idx))
        if n_frames is not None:
            common = common[:n_frames]

        images = [tifffile.imread(io.BytesIO(zf.read(img_by_idx[i]))) for i in common]
        gt_labels = [
            tifffile.imread(io.BytesIO(zf.read(gt_by_idx[i]))).astype(np.int32) for i in common
        ]
    return images, gt_labels


def segment_sequence(model: StarDist2D, images: list) -> list:
    """Run StarDist independently on each frame -- one pass, no ground truth involved
    at this stage. Each frame's labels are only unique *within* that frame; linking
    them into consistent IDs across frames is btrack's job, not StarDist's."""
    return [run_stardist(model, image)[0] for image in images]


def track_sequence(pred_labels_stack: list, config_path, max_search_radius: float = None) -> tuple:
    """Convert per-frame instance labels into btrack objects and link them into tracks.
    max_search_radius=None keeps whatever the config file (config_path) already sets,
    same "None means use the model's/config's own tuned value" convention as
    agentic_stardist.py's run_stardist.

    Returns (tracks, id_to_frame_label) where id_to_frame_label maps each btrack
    object's assigned ID to the (frame, label) it came from. This works because
    `BayesianTracker.append` assigns `obj.ID = idx + len(self._objects)` in the
    order objects are passed in, and `segmentation_to_objects` builds that list
    frame-by-frame (in ascending regionprops label order within each frame) -- so
    enumerating the pre-append object list gives the exact same IDs append() will
    assign, without needing to re-derive them via centroid matching afterward."""
    stack = np.stack(pred_labels_stack).astype(np.int32)
    objects = btrack.utils.segmentation_to_objects(stack, properties=("area",))
    id_to_frame_label = {i: (int(obj.t), int(obj.label)) for i, obj in enumerate(objects)}

    with btrack.BayesianTracker() as tracker:
        tracker.configure(config_path)
        if max_search_radius is not None:
            tracker.max_search_radius = max_search_radius
        tracker.append(objects)
        height, width = stack.shape[1:]
        tracker.volume = ((0, width), (0, height))
        tracker.track()
        tracker.optimize()
        tracks = tracker.tracks

    return tracks, id_to_frame_label


def build_predicted_track_map(tracks: list, id_to_frame_label: dict) -> dict:
    """Flatten btrack's per-track object references into a (frame, label) -> track_id
    lookup, for comparing against ground truth per object. `track.refs` holds the IDs
    of the objects making up that track; negative refs are dummy objects the tracker
    inserted to bridge a gap (no real (frame, label) to map back to), so they're
    skipped."""
    pred_track_id_map = {}
    for track in tracks:
        for ref in track.refs:
            if ref < 0:
                continue
            frame, label = id_to_frame_label[ref]
            pred_track_id_map[(frame, label)] = track.ID
    return pred_track_id_map


def match_frame_instances(pred_labels: np.ndarray, gt_labels: np.ndarray, iou_threshold: float = 0.5) -> dict:
    """For each ground-truth instance in one frame, find the predicted instance with
    highest IoU, keeping the match only if IoU >= iou_threshold. Same greedy-matching
    logic as agentic_stardist.py's compute_panoptic_quality (at threshold 0.5 it's
    automatically one-to-one), just returning the {gt_id: pred_id} mapping itself
    instead of aggregate counts, since here it feeds into a cross-frame comparison
    rather than a same-frame score."""
    matches = {}
    for gt_id in np.unique(gt_labels):
        if gt_id == 0:
            continue
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
            matches[int(gt_id)] = int(best_pred_id)
    return matches


def compute_link_accuracy(gt_labels_stack: list, pred_labels_stack: list, pred_track_id_map: dict) -> dict:
    """Simplified proxy for tracking accuracy (NOT the official CTC TRA/AOGM metric --
    that's a graph edit distance between the full predicted and ground-truth
    lineage graphs, overkill for driving a retry loop). A "GT link" is a ground-truth
    track ID present in two consecutive frames (t, t+1) -- since GT labels ARE track
    IDs already (see load_ctc_sequence), this needs no lineage file to find. For each
    GT link: match the GT instance to a predicted instance in both frames (IoU-based,
    see match_frame_instances); the link is "kept" if both matched predicted instances
    were assigned the same btrack track ID, "broken" if either side had no predicted
    match at all (a missed detection somewhere), and a "switch" if both matched but got
    different track IDs (btrack failed to link two frames of the same real object).

    Returns {"link_accuracy": kept/total, "num_links": total, "num_kept": ...,
    "num_switches": ..., "num_broken": ...}. link_accuracy of 1.0 means every
    ground-truth object present in two consecutive frames was tracked correctly
    across them."""
    frame_matches = [match_frame_instances(pred, gt) for pred, gt in zip(pred_labels_stack, gt_labels_stack)]

    num_kept = num_switches = num_broken = 0
    for t in range(len(gt_labels_stack) - 1):
        gt_ids_t = set(np.unique(gt_labels_stack[t])) - {0}
        gt_ids_t1 = set(np.unique(gt_labels_stack[t + 1])) - {0}
        for gt_id in gt_ids_t & gt_ids_t1:
            pred_id_t = frame_matches[t].get(gt_id)
            pred_id_t1 = frame_matches[t + 1].get(gt_id)
            if pred_id_t is None or pred_id_t1 is None:
                num_broken += 1
                continue
            track_id_t = pred_track_id_map.get((t, pred_id_t))
            track_id_t1 = pred_track_id_map.get((t + 1, pred_id_t1))
            if track_id_t is not None and track_id_t == track_id_t1:
                num_kept += 1
            else:
                num_switches += 1

    num_links = num_kept + num_switches + num_broken
    link_accuracy = num_kept / num_links if num_links else 1.0
    return {
        "link_accuracy": link_accuracy,
        "num_links": num_links,
        "num_kept": num_kept,
        "num_switches": num_switches,
        "num_broken": num_broken,
    }


def save_trajectories(images: list, tracks: list, output_path: Path) -> None:
    """Plot every track's path as a colored line over the first frame, one distinct
    color per track ID (same "outline every instance in its own color" idea as
    agentic_stardist.py's save_instance_outlines, applied to trajectories instead of
    single-frame boundaries)."""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(images[0], cmap="gray")
    colormap = plt.colormaps["hsv"].resampled(max(len(tracks), 1))
    for i, track in enumerate(tracks):
        ax.plot(track.x, track.y, "-", color=colormap(i), linewidth=1.2)
        ax.plot(track.x[0], track.y[0], "o", color=colormap(i), markersize=3)
    ax.axis("off")
    ax.set_title(f"{len(tracks)} tracks over {len(images)} frames")
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def evaluate_result(
    claude: anthropic.Anthropic,
    user_prompt: str,
    max_search_radius: float,
    link_result: dict,
    trajectory_image_path: str,
    history: list,
) -> dict:
    """Only called when link_accuracy is below ACCEPT_LINK_ACCURACY_THRESHOLD -- Claude
    doesn't decide accept/reject (that's computed directly from ground truth), it only
    proposes what to try next."""
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
                        "revised_max_search_radius": {"type": ["number", "null"]},
                    },
                    "required": ["feedback", "revised_max_search_radius"],
                    "additionalProperties": False,
                },
            }
        },
        messages=[{
            "role": "user",
            "content": [
                image_to_content_block(trajectory_image_path),
                {"type": "text", "text": (
                    f"Original user request: \"{user_prompt}\"\n"
                    f"btrack ran with max_search_radius={max_search_radius:.1f}\n"
                    f"Link accuracy against ground truth: {link_result['link_accuracy']:.3f} "
                    f"(below the {ACCEPT_LINK_ACCURACY_THRESHOLD} acceptance bar)\n"
                    f"  kept={link_result['num_kept']}  switches={link_result['num_switches']}  "
                    f"broken={link_result['num_broken']}  (of {link_result['num_links']} ground-truth links)\n"
                    f"Prior attempts this session: {json.dumps(history)}\n\n"
                    "The attached image shows every predicted track's trajectory over the "
                    "first frame. 'switches' = btrack linked an object correctly in isolation "
                    "each frame but assigned different track IDs across frames (trajectories "
                    "look broken/reassigned mid-path); 'broken' = an object present in "
                    "consecutive ground-truth frames couldn't be matched to any predicted "
                    "instance at all (a missed detection, not a linking failure). Propose a "
                    "revised max_search_radius (the maximum pixel distance btrack will link "
                    "an object across consecutive frames): raise it if switches are driven by "
                    "real, fast-moving objects falling outside the current radius; lower it if "
                    "it's linking to the wrong nearby object instead. Set "
                    "revised_max_search_radius to null only if neither adjustment would help."
                )},
            ],
        }],
    )
    text = next(b.text for b in response.content if b.type == "text")
    return json.loads(text)


ACCEPT_LINK_ACCURACY_THRESHOLD = 0.9

SCORING_RUBRIC = (
    "Each iteration is scored with a simplified tracking-link accuracy against CTC's\n"
    "ground truth (NOT the official CTC TRA/AOGM metric -- that's a graph edit distance,\n"
    "overkill for driving a retry loop):\n\n"
    "    link_accuracy = kept_links / (kept_links + switches + broken_links)\n\n"
    "A 'link' is a ground-truth track present in two consecutive frames. It's 'kept' if\n"
    "btrack assigned the same track ID to the matched predicted instance in both frames,\n"
    "a 'switch' if it matched in both frames but got different track IDs, and 'broken' if\n"
    "the object couldn't be matched to any predicted instance in one of the frames at all\n"
    "(a missed detection, not a linking failure).\n\n"
    f"A link_accuracy >= {ACCEPT_LINK_ACCURACY_THRESHOLD} accepts the result -- computed directly from\n"
    "the ground truth, not judged by Claude. Below that, either Claude (--claude-feedback)\n"
    "or a free, deterministic rule (the default -- see propose_search_radius) looks at the\n"
    "switch/broken breakdown and proposes a revised max_search_radius to retry with.\n\n"
    "Loops to good result: the number of iterations run before link_accuracy first reached\n"
    "the accept threshold (or the total number run, if none did) -- a measure of how many\n"
    "retries the agentic loop needed, separate from the accuracy of the final result itself."
)


def loops_to_acceptance(history: list) -> tuple:
    for entry in history:
        if entry["link_accuracy"] >= ACCEPT_LINK_ACCURACY_THRESHOLD:
            return entry["iteration"], True
    return len(history), False


def best_entry(history: list) -> dict:
    """The highest-link_accuracy iteration seen (ties go to the earliest) -- same
    rationale as agentic_stardist.py's best_entry: the loop can regress after a
    non-accepting iteration, so nothing should just report history[-1] as final."""
    return max(history, key=lambda e: e["link_accuracy"])


RADIUS_STEP = 10.0


def _already_tried(value: float, tried: list, tol: float = 0.5) -> bool:
    return any(abs(value - t) < tol for t in tried)


def propose_search_radius(current_radius: float, link_result: dict, history: list = None) -> tuple:
    """Free, deterministic alternative to evaluate_result -- no API call, no cost.
    max_search_radius is btrack's only easily-revisable knob (contrast with StarDist's
    prob_thresh/nms_thresh pair), so this is a single-lever rule:
      - broken > switches (more missed matches than wrong-ID links): the radius is
        probably gating out real, fast-moving links -- raise it.
      - switches > broken (more wrong-ID links than misses): the radius is probably
        wide enough to catch the wrong nearby object -- lower it.
      - equal: default to raising it (a too-small radius is the more common failure
        mode against default configs tuned on slower-moving reference datasets).
    Same oscillation guard as agentic_stardist.py's propose_thresholds: halve the
    step if the plain step would revisit an already-tried radius.
    Returns (revised_max_search_radius, feedback)."""
    history = history or []
    tried = [e["max_search_radius"] for e in history]
    switches, broken = link_result["num_switches"], link_result["num_broken"]

    sign = 1 if broken >= switches else -1
    step = RADIUS_STEP
    revised = round(max(current_radius + sign * step, 1.0), 1)
    while _already_tried(revised, tried) and step > 1.0:
        step /= 2
        revised = round(max(current_radius + sign * step, 1.0), 1)

    comparison = (
        f"broken={broken} > switches={switches}" if broken > switches
        else f"switches={switches} > broken={broken}" if switches > broken
        else f"broken == switches == {broken}"
    )
    direction = "raising" if sign > 0 else "lowering"
    feedback = f"{comparison}: {direction} max_search_radius to {revised}."
    return revised, feedback


def run_track_loop(
    claude,
    pred_labels_stack: list,
    gt_labels_stack: list,
    images: list,
    config_path,
    starting_radius: float,
    prompt: str,
    max_iterations: int,
    output_dir: Path,
) -> tuple:
    """Run the max_search_radius retry loop for one CTC sequence against its ground
    truth. StarDist runs exactly once before this is called (see segment_sequence) --
    only btrack's linking step reruns each iteration, since max_search_radius has no
    effect on per-frame segmentation. Saves one trajectory PNG per iteration under
    output_dir, named "iteration_{i}.png". Returns (tracks, history, saved_paths)."""
    max_search_radius = starting_radius
    history = []
    saved_paths = []
    tracks = None
    for i in range(1, max_iterations + 1):
        print(f"  iteration {i}: max_search_radius={max_search_radius:.1f}")
        tracks, id_to_frame_label = track_sequence(pred_labels_stack, config_path, max_search_radius)
        pred_track_id_map = build_predicted_track_map(tracks, id_to_frame_label)
        link_result = compute_link_accuracy(gt_labels_stack, pred_labels_stack, pred_track_id_map)
        accept = link_result["link_accuracy"] >= ACCEPT_LINK_ACCURACY_THRESHOLD
        print(
            f"    tracks={len(tracks)}  link_accuracy={link_result['link_accuracy']:.3f} "
            f"(kept={link_result['num_kept']}, switches={link_result['num_switches']}, "
            f"broken={link_result['num_broken']})"
        )

        saved_path = output_dir / f"iteration_{i}.png"
        save_trajectories(images, tracks, saved_path)
        saved_paths.append(saved_path)

        revised_radius = None
        if accept:
            feedback = "Link accuracy met the acceptance threshold."
        elif i == max_iterations:
            feedback = "Reached max iterations without meeting the acceptance threshold."
        elif claude is not None:
            eval_result = evaluate_result(claude, prompt, max_search_radius, link_result, str(saved_path), history)
            feedback = eval_result["feedback"]
            revised_radius = eval_result.get("revised_max_search_radius")
            print(f"    [Claude eval] feedback: {feedback}")
        else:
            revised_radius, feedback = propose_search_radius(max_search_radius, link_result, history)
            print(f"    [rule-based] {feedback}")

        history.append({
            "iteration": i,
            "max_search_radius": max_search_radius,
            "num_tracks": len(tracks),
            "link_accuracy": link_result["link_accuracy"],
            "num_kept": link_result["num_kept"],
            "num_switches": link_result["num_switches"],
            "num_broken": link_result["num_broken"],
            "feedback": feedback,
        })

        if accept or revised_radius is None:
            break
        max_search_radius = revised_radius

    best = best_entry(history)
    if best["iteration"] != history[-1]["iteration"]:
        print(
            f"  search continued past its best result -- reverting to iteration "
            f"{best['iteration']} (link_accuracy={best['link_accuracy']:.3f}) instead of "
            f"the last one tried (link_accuracy={history[-1]['link_accuracy']:.3f})"
        )
        tracks, id_to_frame_label = track_sequence(pred_labels_stack, config_path, best["max_search_radius"])

    return tracks, history, saved_paths


def save_pdf_report(pdf_path: Path, image_names: list, trajectory_path: Path, num_tracks: int) -> None:
    """Render a single page with the first frame next to the trajectory plot (plain
    forward-pass mode, no ground truth -- see agentic_stardist.py's save_pdf_report)."""
    with PdfPages(pdf_path) as pdf:
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis("off")
        ax.imshow(Image.open(trajectory_path))
        fig.suptitle(
            f"btrack -- {len(image_names)} frames, {num_tracks} tracks\nFirst frame: {image_names[0]}",
            fontsize=13,
            fontweight="bold",
        )
        pdf.savefig(fig)
        plt.close(fig)


def save_loop_pdf_report(pdf_path: Path, user_prompt: str, image_paths: list, history: list) -> None:
    """Render a methodology page, then one page per iteration (trajectory plot + link-
    accuracy breakdown), into a single PDF -- same shape as agentic_stardist.py's
    save_loop_pdf_report."""
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
            ax_img.set_title(f"Iteration {entry['iteration']}: max_search_radius={entry['max_search_radius']:.1f}")

            ax_text.axis("off")
            caption = (
                f"Request: {user_prompt}\n"
                f"Tracks found: {entry['num_tracks']}\n"
                f"Link accuracy: {entry['link_accuracy']:.3f}  (kept={entry['num_kept']}, "
                f"switches={entry['num_switches']}, broken={entry['num_broken']})\n"
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
                 f"below link_accuracy {ACCEPT_LINK_ACCURACY_THRESHOLD})"
        )
        summary = (
            f"Request: {user_prompt}\n\n"
            f"{loops_line}\n"
            f"Final max_search_radius: {final['max_search_radius']:.1f}\n"
            f"Final tracks found: {final['num_tracks']}\n"
            f"Final link accuracy: {final['link_accuracy']:.3f} (kept={final['num_kept']}, "
            f"switches={final['num_switches']}, broken={final['num_broken']})"
        )
        ax.text(0, 0.95, summary, va="top", ha="left", fontsize=11, wrap=True)
        pdf.savefig(fig)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Run btrack with Claude as orchestrator/evaluator")
    parser.add_argument("--images-dir", default=None, help="Directory of ordered frame images (single pass, no ground truth)")
    parser.add_argument("--ctc-dataset", default=None, help="CTC training dataset name, e.g. Fluo-N2DL-HeLa")
    parser.add_argument("--ctc-sequence", default="01", choices=["01", "02"], help="Sequence within the CTC dataset")
    parser.add_argument("--ctc-cache-dir", default="./ctc_cache", help="Where to cache downloaded CTC dataset zips")
    parser.add_argument("--n-frames", type=int, default=None, help="Limit to the first n frames (both modes)")
    parser.add_argument(
        "--prompt", default="track individual cells across frames",
        help="What the tracking should satisfy (--ctc-dataset path only)",
    )
    parser.add_argument("--max-iterations", type=int, default=5, help="--ctc-dataset path only")
    parser.add_argument(
        "--claude-feedback", action="store_true",
        help="Use the Claude API to propose a revised max_search_radius (costs a small amount per call). "
             "Default is the free, deterministic propose_search_radius heuristic -- no API key needed.",
    )
    parser.add_argument("--output-dir", default="./btrack_agent_output")
    parser.add_argument("--pdf-name", default=PDF_NAME, help="Filename for the saved PDF report")
    args = parser.parse_args()
    if not args.images_dir and not args.ctc_dataset:
        parser.error("one of --images-dir or --ctc-dataset is required")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stardist_model = StarDist2D.from_pretrained(STARDIST_PRETRAINED_MODEL)
    config_path = btrack.datasets.cell_config()

    if args.ctc_dataset:
        claude = anthropic.Anthropic() if args.claude_feedback else None
        print(f"Fetching CTC {args.ctc_dataset} sequence {args.ctc_sequence}...")
        images, gt_labels_stack = load_ctc_sequence(
            args.ctc_dataset, args.ctc_sequence, Path(args.ctc_cache_dir), args.n_frames
        )
        print(f"{len(images)} frames, {sum(len(np.unique(g)) - 1 for g in gt_labels_stack)} ground-truth instances total")

        print("Running StarDist per frame...")
        pred_labels_stack = segment_sequence(stardist_model, images)

        with btrack.BayesianTracker() as probe:
            probe.configure(config_path)
            starting_radius = float(probe.max_search_radius)

        tracks, history, saved_paths = run_track_loop(
            claude, pred_labels_stack, gt_labels_stack, images, config_path,
            starting_radius, args.prompt, args.max_iterations, output_dir,
        )

        tracks_path = output_dir / TRACKS_NAME
        with btrack.io.HDF5FileHandler(str(tracks_path), "w", obj_type="obj_type_1") as writer:
            writer.write_tracks(tracks)

        pdf_path = unique_path(output_dir / args.pdf_name)
        save_loop_pdf_report(pdf_path, args.prompt, saved_paths, history)

        final = best_entry(history)
        print("\n=== Final result ===")
        print(f"Tracks found: {final['num_tracks']}")
        print(f"Final link accuracy: {final['link_accuracy']:.3f}")
        print(f"Tracks saved: {tracks_path}")
        print(f"Trajectory plot: {saved_paths[final['iteration'] - 1]}")
        print(f"PDF report saved: {pdf_path}")
        print(f"History: {json.dumps(history, indent=2)}")
        return

    images, image_names = load_image_sequence(args.images_dir)
    if args.n_frames is not None:
        images, image_names = images[:args.n_frames], image_names[:args.n_frames]

    print("Running StarDist per frame...")
    pred_labels_stack = segment_sequence(stardist_model, images)

    print("Running btrack...")
    tracks, _ = track_sequence(pred_labels_stack, max_search_radius=None, config_path=config_path)
    num_tracks = len(tracks)

    tracks_path = output_dir / TRACKS_NAME
    with btrack.io.HDF5FileHandler(str(tracks_path), "w", obj_type="obj_type_1") as writer:
        writer.write_tracks(tracks)

    trajectory_path = output_dir / TRAJECTORY_NAME
    save_trajectories(images, tracks, trajectory_path)

    pdf_path = unique_path(output_dir / args.pdf_name)
    save_pdf_report(pdf_path, image_names, trajectory_path, num_tracks)

    print("=== btrack result ===")
    print(f"Frames: {len(images)}")
    print(f"Tracks found: {num_tracks}")
    print(f"Tracks saved: {tracks_path}")
    print(f"Trajectory plot saved: {trajectory_path}")
    print(f"PDF report saved: {pdf_path}")


if __name__ == "__main__":
    main()
