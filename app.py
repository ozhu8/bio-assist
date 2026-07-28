"""
Flask front end for the CountGD tools.

Wraps the existing pipelines in agentic_countgd.py and compare_results.py
(neither of those files is modified) so results can be viewed as an inline
PDF in the browser instead of run from the CLI:

  - /agentic  upload one image + what to count -> routed by
              manager_agent.ManagerAgent (local Qwen3-VL) to CountGD or
              StarDist, retried/evaluated by the same manager -> PDF report
  - /compare  upload a raw image (run through the same ManagerAgent as
              /agentic) alongside an already-finished baseline image + its
              count -> a side-by-side comparison PDF built with plain
              matplotlib (no LLM involved in the drawing/charting itself)

Both routes hand the actual pipeline run off to a background thread and
return a progress page immediately; the browser polls /progress/<run_id>
for status and redirects to /result/<run_id> once it's done. RUNS is an
in-memory job store — fine for a single-process dev server, but progress
state is lost on restart and never evicted.

Run with: python app.py
"""
import mimetypes
import os
import textwrap
import threading
import traceback
import uuid
from pathlib import Path

# Must run before anything (including agentic_countgd/compare_results) imports
# matplotlib.pyplot. On macOS, matplotlib's default interactive backend talks
# to Cocoa/AppKit, which crashes the whole process (uncaught NSException) if
# touched from a background thread — and the pipeline now always runs in one.
# "Agg" is the correct non-interactive backend anyway since we only ever
# write PDFs to disk, never show a window.
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from flask import Flask, flash, jsonify, redirect, render_template, request, send_file, url_for
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
from scipy import ndimage
from skimage.feature import peak_local_max

# Pillow's default decompression-bomb cap (~179 megapixels) exists to stop a malicious tiny
# file from expanding into a huge memory allocation -- but a real whole-slide pathology image
# (the DeepGleason/tumor-grading agent's own input) is legitimately 1+ gigapixels. Even just
# reading its header dimensions (Image.open(path).size, used by prepare_qwen_image below to
# decide whether a pyvips preview is needed) trips this cap before any full decode happens, so
# it has to be disabled to get that far at all. Disabling it is a deliberate tradeoff, not an
# oversight: this app runs privately for a small known set of users, not as a public-facing
# service, and prepare_qwen_image() means the only thing PIL itself ever fully decodes is a
# normal-sized image or a small pyvips-generated preview -- never the raw gigapixel file, which
# pyvips (not Pillow) handles directly -- so this mostly just relaxes a header-only size check.
Image.MAX_IMAGE_PIXELS = None

import agentic_deepgleason
import manager_agent
from agentic_countgd import run_countgd
from compare_results import (
    COLOR_AGENTIC,
    COLOR_BASELINE,
    COLOR_MISSED_BY_AGENTIC,
    COLOR_MISSED_BY_BASELINE,
    GRIDLINE,
    INK_PRIMARY,
    INK_SECONDARY,
    LEGEND_BACKING,
    find_missed_points,
)

# agentic_countgd.image_to_content_block() falls back to image/png when
# mimetypes can't identify the extension (e.g. .webp on Python < 3.11) —
# Claude's vision API then rejects the mismatch between declared and actual
# type. Registering these explicitly makes detection version-independent.
mimetypes.add_type("image/webp", ".webp")
mimetypes.add_type("image/png", ".png")
mimetypes.add_type("image/jpeg", ".jpg")
mimetypes.add_type("image/jpeg", ".jpeg")

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "webapp_data" / "uploads"
OUTPUT_DIR = BASE_DIR / "webapp_data" / "output"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}
MAX_ITERATIONS = 3

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-only-not-secret")

_manager = None

# run_id -> {"status": "running"|"done"|"error"|"cancelled", "progress": 0-100, "stage": str, ...}
RUNS = {}
RUNS_LOCK = threading.Lock()

# run_id -> threading.Event(), set by POST /cancel/<run_id> and checked at
# each pipeline checkpoint (see check_cancelled). Cooperative, not forced —
# a call already in flight to Claude/Qwen/CountGD finishes before the next check.
CANCEL_EVENTS = {}
CANCEL_LOCK = threading.Lock()


class RunCancelled(Exception):
    """Raised at a pipeline checkpoint once /cancel/<run_id> has been hit."""


def get_cancel_event(run_id: str) -> threading.Event:
    with CANCEL_LOCK:
        event = CANCEL_EVENTS.get(run_id)
        if event is None:
            event = threading.Event()
            CANCEL_EVENTS[run_id] = event
        return event


def check_cancelled(cancel_event: threading.Event) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise RunCancelled()


def get_manager() -> manager_agent.ManagerAgent:
    """Single ManagerAgent instance for the whole process -- its .qwen (the loaded Qwen3-VL
    model), .countgd_client (CountGD Gradio Space, with the extended cold-start timeout -- see
    ManagerAgent.countgd_client), .stardist_worker (the spawned StarDist subprocess), .cellvit_client,
    and .deepgleason_client are all lazy singletons of their own underneath, so this is cheap to
    call repeatedly.

    deepgleason_repo/deepgleason_python/deepgleason_model are fine left at their ManagerAgent
    defaults (None) -- DeepGleasonClient falls back to agentic_deepgleason.py's own env-var-based
    defaults (DEEPGLEASON_REPO/DEEPGLEASON_PYTHON/DEEPGLEASON_MODEL) in that case. cellvit_checkpoint
    has no such fallback anywhere (ManagerAgent.cellvit_client asserts it's set) -- there's no
    universal default path for a CellViT-SAM-H checkpoint, so it must come from wherever it was
    actually placed on this machine; set CELLVIT_CHECKPOINT (and CELLVIT_REPO if the CellViT repo
    itself isn't on PYTHONPATH already) before starting the app for CellViT routing to work at
    all -- left unset, requests that route to cellvit will fail with a clear AssertionError
    rather than a silent wrong answer."""
    global _manager
    if _manager is None:
        _manager = manager_agent.ManagerAgent(
            cellvit_checkpoint=os.environ.get("CELLVIT_CHECKPOINT"),
            cellvit_repo=os.environ.get("CELLVIT_REPO"),
        )
    return _manager


def save_upload(file_storage, dest_dir: Path) -> Path:
    ext = Path(file_storage.filename).suffix.lower()
    if ext not in ALLOWED_IMAGE_EXT:
        raise ValueError(f"Unsupported image type: {ext or '(none)'}")
    dest = dest_dir / f"{uuid.uuid4().hex}{ext}"
    file_storage.save(dest)
    return dest


def set_progress(run_id: str, fraction: float, stage: str) -> None:
    """Record a real pipeline checkpoint — not a simulated/animated fraction."""
    with RUNS_LOCK:
        entry = RUNS.get(run_id)
        if entry is None:
            return
        entry["progress"] = max(0, min(100, round(fraction * 100)))
        entry["stage"] = stage


# Comfortably below Pillow's own ~179-megapixel decompression-bomb cap, but high enough that a
# normal microscopy upload (a few megapixels) never triggers this -- only a genuinely huge
# whole-slide image (built for the deepgleason agent) does.
QWEN_PREVIEW_PIXEL_THRESHOLD = 50_000_000


def prepare_qwen_image(image_path: Path, run_dir: Path) -> tuple:
    """Qwen's own image loader (via qwen_vl_utils, backed by Pillow) can't handle a real
    whole-slide image directly -- gigapixel-scale, and Pillow's decompression-bomb cap would
    reject it outright otherwise (see Image.MAX_IMAGE_PIXELS above). manager_agent.py's own CLI
    handles this via its --slide flag: a small pyvips thumbnail for routing/dialogue (image_path)
    plus the real file separately (slide_path, used only if routing lands on deepgleason). A
    browser upload has no equivalent explicit flag to say "this one's huge" -- this checks the
    image's actual pixel dimensions (a fast, header-only read, not a full decode) and generates
    that same kind of preview only when it's actually needed, mirroring the CLI's own pattern.

    Returns (qwen_image_path, slide_path) -- slide_path is None unless the image was actually
    downsampled, in which case it's the original file path (str)."""
    width, height = Image.open(image_path).size
    if width * height <= QWEN_PREVIEW_PIXEL_THRESHOLD:
        return str(image_path), None

    import pyvips  # only needed for this rare, whole-slide-scale path
    preview_path = run_dir / "qwen_routing_preview.png"
    pyvips.Image.thumbnail(str(image_path), 512).write_to_file(str(preview_path))
    return str(preview_path), str(image_path)


def run_baseline_once(manager: manager_agent.ManagerAgent, task_description: str, image_path: str, output_dir) -> dict:
    """One-shot counterpart to ManagerAgent.run() -- no retry loop, no manager<->expert dialogue,
    just routes to whichever agent fits (select_agent) and runs it once. Used by the Compare
    page's "baseline" (model alone, no agentic feedback loop) so it can be compared against the
    full agentic result without the retry/dialogue machinery running several times more work
    than the comparison actually needs. The new manager_agent.py has no equivalent single-shot
    helper of its own (its only exposed entry points are the full retry-loop functions), so this
    is app.py's own thin wrapper around the same underlying one-shot calls those retry loops make
    on their first iteration. deepgleason has no single count to compare, so that agent isn't
    supported here -- raises ValueError instead of returning something the comparison chart/diff
    couldn't meaningfully render anyway."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    agent = manager_agent.select_agent(manager.qwen, task_description, image_path)

    if agent == "countgd":
        count_target = manager_agent.interpret_countgd_target(manager.qwen, task_description, image_path)
        annotated_path, count = run_countgd(manager.countgd_client, image_path, count_target)
        return {"agent": "countgd", "label": count_target, "count": count, "annotated_image": annotated_path}

    if agent == "cellvit":
        from agentic_cellvit import interpret_request as interpret_cellvit_request
        request = interpret_cellvit_request(manager.qwen, task_description, image_path)
        target_classes = set(request["target_classes"])
        annotated, matched_cells, counts_by_type, all_cells = manager.cellvit_client.run(
            image_path, target_classes, request["prob_threshold"]
        )
        annotated_path = output_dir / "cellvit_baseline.png"
        annotated.save(annotated_path)
        return {
            "agent": "cellvit", "label": ", ".join(sorted(target_classes)), "count": len(matched_cells),
            "annotated_image": str(annotated_path),
        }

    if agent == "deepgleason":
        raise ValueError(
            "This image/task routed to DeepGleason (whole-slide tumor grading), which has no "
            "single count to compare -- the Compare page only supports CountGD/StarDist/CellViT. "
            "Try the main Agentic Count page for tumor grading instead."
        )

    image, prob_thresh, nms_thresh = manager.stardist_worker.init(image_path)
    outlines_path = output_dir / "stardist_baseline.png"
    result = manager.stardist_worker.run(image, prob_thresh, nms_thresh, None, outlines_path)
    return {
        "agent": "stardist", "label": "segmentation", "count": int(result["labels"].max()),
        "annotated_image": str(outlines_path), "labels": result["labels"],
    }


def run_agentic_pipeline(
    image_path: Path, prompt: str, run_dir: Path, run_id: str,
    max_iterations: int = MAX_ITERATIONS, progress_cb=None, cancel_event=None,
) -> dict:
    """Routing (select_agent -- one of countgd/stardist/cellvit/deepgleason, decided in a single
    call, no separate task-type pre-check anymore), the retry loop, and the manager<->expert
    dialogue (always present now, even with no ground truth -- see manager_agent.py's module
    docstring) all happen inside manager_agent.ManagerAgent.run(). This function's job is:
    generate a small Qwen-safe preview if the upload is whole-slide-scale (prepare_qwen_image),
    hand run_dir to manager.run() to write its own per-iteration images into, recover a StarDist
    result's label array as a .npy sidecar (for the missed-cell diff -- see
    get_detection_coords), and build whichever agent-specific PDF report fits the result.

    progress_cb(fraction, stage_text), if given, is called at each real pipeline checkpoint —
    fraction is 0-1 progress through THIS function (ManagerAgent.run()'s own 0-1 progress is
    rescaled into 0-0.90, leaving 0.90-1.00 for PDF-building here). cancel_event, if given, is
    checked at the same checkpoints via the report() closure passed down as ManagerAgent.run()'s
    progress_cb — manager_agent.py doesn't know about cancellation itself, it just calls
    whatever callable it's given and lets any exception from it propagate, so report() raising
    RunCancelled unwinds straight back out of manager.run().

    Returns a dict with "agent", "pdf_path", "history", and "summary_rows" (ready for the
    /agentic route's result page) always present. For countgd/stardist/cellvit (but not
    deepgleason, which has no single comparable count -- see the /compare route), it also
    includes "final_count"/"final_image_path"/"count_target"/"counting_model" for the Compare
    page's side-by-side/chart/missed-cell-diff rendering."""
    def report(fraction, stage):
        check_cancelled(cancel_event)
        if progress_cb:
            progress_cb(fraction, stage)

    manager = get_manager()
    qwen_image_path, slide_path = prepare_qwen_image(image_path, run_dir)
    result = manager.run(
        prompt, qwen_image_path, max_iterations, str(run_dir),
        image_id=run_id, slide_path=slide_path,
        progress_cb=lambda frac, stage: report(frac * 0.90, stage),
        # DeepGleason's conda env/repo was never set up on this machine (see CLAUDE.md) -- a
        # real user prompt that happened to route there would otherwise crash deep inside
        # run_deepgleason_with_feedback's subprocess call instead of failing with a clear,
        # actionable message right at routing time.
        disabled_agents={"deepgleason"},
    )

    agent = result["agent"]
    history = result["history"]

    report(0.94, "Building PDF report…")
    pdf_path = run_dir / "results.pdf"

    if agent == "deepgleason":
        save_deepgleason_pdf_report(pdf_path, prompt, result)
        report(1.0, "Done")
        gleason_result = result["gleason_result"]
        return {
            "agent": agent, "pdf_path": pdf_path, "history": history,
            "summary_rows": [
                ("Model used", "deepgleason"),
                ("Result", deepgleason_summary_line(gleason_result)),
                ("Tiles analyzed", gleason_result["total_tiles"]),
                ("Iterations run", len(history)),
            ],
        }

    # ManagerAgent.run() already wrote {run_id}_iteration_N.png straight into run_dir itself
    # (it was handed run_dir as its own output_dir, and image_id=run_id above -- see
    # manager_agent.py's own saved_path = output_dir / f"{image_id or '<agent>'}_iteration_{i}.png")
    # -- reconstruct those same per-iteration paths for save_pdf_report instead of re-saving them.
    saved_paths = [run_dir / f"{run_id}_iteration_{entry['iteration']}.png" for entry in history]

    if agent == "stardist":
        final_image_path = Path(result["outlines_image"])
        final_count = result["num_nuclei"]
        count_target = "segmentation"
        # Save the real per-instance label array as a .npy sidecar next to the final image
        # (same stem) so get_detection_coords() can use real instance centroids for the
        # missed-cell diff on the Compare page, instead of guessing from brightness peaks --
        # manager_agent.py's own StardistWorker only writes the rendered outline PNG.
        np.save(final_image_path.with_suffix(".npy"), result["labels"])
        summary_rows = [
            ("Model used", "stardist"), ("Final count", final_count), ("Iterations run", len(history)),
        ]
    elif agent == "cellvit":
        final_image_path = Path(result["annotated_image"])
        final_count = result["count"]
        count_target = ", ".join(result["target_classes"])
        summary_rows = [
            ("Model used", "cellvit"),
            ("Target classes", count_target),
            ("Final count", final_count),
            ("Counts by type", ", ".join(f"{k}={v}" for k, v in result["counts_by_type"].items())),
            ("Iterations run", len(history)),
        ]
    else:  # countgd
        final_image_path = Path(result["annotated_image"])
        final_count = result["count"]
        count_target = result["count_target"]
        summary_rows = [
            ("Model used", "countgd"),
            ("Count target used", count_target),
            ("Final count", final_count),
            ("Iterations run", len(history)),
        ]

    save_pdf_report(
        pdf_path, prompt, saved_paths, history, counting_model=agent,
        chosen_iteration=result["chosen_iteration"],
    )
    report(1.0, "Done")

    return {
        "agent": agent, "pdf_path": pdf_path, "history": history, "summary_rows": summary_rows,
        "final_count": final_count, "final_image_path": final_image_path,
        "count_target": count_target, "counting_model": agent,
    }


def detect_dot_coords_adaptive(image_path: str, min_distance: int = 6, sigma: float = 1.2, relative_threshold: float = 0.5):
    """Find CountGD's detection-dot centers by brightness, thresholded relative
    to THIS image's own peak brightness rather than a fixed absolute cutoff.

    compare_results.detect_dot_coords uses threshold_abs=40 on a fixed
    yellow-vs-blue color formula. That breaks whenever CountGD renders dots
    at a different size/style between two calls (confirmed on a real run:
    it detected 84 dots on one image and only 4 on another that visually had
    just as many, actual counts 80 and 73) — Gaussian blur dilutes a small
    dot's peak color value below the fixed threshold even though it's still
    clearly the brightest thing in the image. Scaling the threshold to each
    image's own max brightness sidesteps that: 0.45-0.60 all reproduced the
    real counts on the run that motivated this; it only breaks down past ~0.65.
    """
    img = Image.open(image_path).convert("RGB")
    arr = np.array(img).astype(float)
    brightness = ndimage.gaussian_filter(arr.sum(axis=-1), sigma=sigma)
    threshold = brightness.max() * relative_threshold
    return peak_local_max(brightness, min_distance=min_distance, threshold_abs=threshold)


def get_detection_coords(image_path: str) -> np.ndarray:
    """Real per-instance centroids when a StarDist label array was saved
    alongside this image (see run_agentic_pipeline / the /compare route's
    baseline "run" branch — same stem, .npy extension, saved here in app.py
    since manager_agent.py's StardistWorker only writes the rendered outline
    PNG itself), since brightness-peak detection has no idea what StarDist's
    colored outlines mean. Falls back to the adaptive dot-detector for
    CountGD's heatmap/dot-style output, or for any image with no sibling
    .npy (e.g. a "finished result" the user uploaded rather than one this
    app generated)."""
    labels_path = Path(image_path).with_suffix(".npy")
    if labels_path.exists():
        from skimage.measure import regionprops
        labels = np.load(labels_path)
        return np.array([region.centroid for region in regionprops(labels)])
    return detect_dot_coords_adaptive(image_path)


def loops_to_acceptance_by_dialogue(history: list) -> tuple:
    """Same shape as agentic_countgd.loops_to_acceptance -- (iteration_count, reached) -- but
    checks entry["accept"] directly instead of a numeric score threshold. The new
    manager_agent.py's dialogue-driven decide_*_from_dialogue functions never produce a 0-10
    score at all (the manager always consults an ExpertReasoner, even with no ground truth, and
    decides accept/reject from that transcript) -- agentic_countgd.py's own loops_to_acceptance
    (frozen, untouched) assumes a "score" key that no longer exists in any history entry here."""
    for entry in history:
        if entry["accept"]:
            return entry["iteration"], True
    return len(history), False


def _iteration_title(entry: dict, counting_model: str) -> str:
    """set_title() centers and never wraps -- unlike ax.text(..., wrap=True) used elsewhere in
    save_pdf_report, a long label (e.g. CountGD's count_target after a revision) runs off both
    edges of the page instead of wrapping, showing up as text cut off mid-word. Titles are meant
    to be single-line, so shorten instead of wrapping to multiple lines here."""
    if counting_model == "stardist":
        return f"Iteration {entry['iteration']}: segmenting nuclei"
    if counting_model == "cellvit":
        classes = textwrap.shorten(", ".join(entry["target_classes"]), width=50, placeholder="…")
        return f"Iteration {entry['iteration']}: highlighting {classes}"
    target_display = textwrap.shorten(entry["count_target"], width=60, placeholder="…")
    return f"Iteration {entry['iteration']}: counting {target_display!r}"


def _iteration_count_line(entry: dict, counting_model: str) -> str:
    if counting_model == "stardist":
        return f"Detected nuclei: {entry['predicted_count']}"
    if counting_model == "cellvit":
        # Cell types get their own clearly-labeled line (not buried in a parenthetical) since
        # that's the actual answer to "what cell types are here" -- and Neoplastic gets called
        # out explicitly as the tumor-cell count, since that's what a "is there a tumor" request
        # is really asking CellViT to answer, even though CellViT itself only classifies
        # individual cells in a tile -- it doesn't make a whole-slide diagnosis the way
        # DeepGleason does. Phrased as a plain count, not a "TUMOR: YES/NO" verdict, so this
        # doesn't overstate what a per-cell classifier can actually determine.
        by_type = ", ".join(f"{k}={v}" for k, v in entry["counts_by_type"].items())
        neoplastic_count = entry["counts_by_type"].get("Neoplastic", 0)
        return (
            f"Matched count: {entry['predicted_count']} (target classes: {', '.join(entry['target_classes'])})\n"
            f"Cell types found: {by_type}\n"
            f"Neoplastic (tumor-type) cells detected: {neoplastic_count}"
        )
    return f"Predicted count: {entry['predicted_count']}"


def save_pdf_report(
    pdf_path: Path, user_prompt: str, image_paths: list, history: list,
    counting_model: str = "countgd", chosen_iteration: int | None = None,
) -> None:
    """One page per iteration (image + caption + the manager<->expert dialogue, which is always
    present now -- see manager_agent.py's module docstring) plus a summary page. counting_model
    is "countgd"/"stardist"/"cellvit" -- deepgleason has its own save_deepgleason_pdf_report
    instead, since it has no per-iteration image at all (every iteration re-aggregates the same
    cached tile predictions, not a fresh model run -- see run_deepgleason_with_feedback).

    chosen_iteration, if given, is manager_agent.py's own choose_best_output() choice of which
    attempt to actually report as the final result -- NOT always the last one tried (the manager
    can revert to an earlier iteration it judges better after looking at all of them; see
    choose_best_output's docstring). Defaults to the last iteration in history if not given, same
    as before choose_best_output existed. Without this, the summary page's "Final" line could
    show a different count than what the website itself displays (result["count"]/
    result["annotated_image"], which already reflect the chosen iteration) -- exactly the
    discrepancy a user would notice comparing the two."""
    if chosen_iteration is None:
        chosen_iteration = history[-1]["iteration"]

    with PdfPages(pdf_path) as pdf:

        for entry, image_path in zip(history, image_paths):
            fig, (ax_img, ax_text) = plt.subplots(2, 1, figsize=(8.5, 11), gridspec_kw={"height_ratios": [4, 1]})
            ax_img.imshow(Image.open(image_path))
            ax_img.axis("off")
            title = _iteration_title(entry, counting_model)
            if entry["iteration"] == chosen_iteration:
                title += "  [chosen as final result]"
            ax_img.set_title(title)

            ax_text.axis("off")
            caption = (
                f"Request: {user_prompt}\n"
                f"{_iteration_count_line(entry, counting_model)}\n"
                f"Accepted: {entry['accept']}\n"
                f"Feedback: {textwrap.fill(entry['feedback'], 100)}"
            )
            ax_text.text(0, 1, caption, va="top", ha="left", fontsize=10, wrap=True)
            pdf.savefig(fig)
            plt.close(fig)

            # Manager<->expert dialogue (manager_agent.py's run_expert_dialogue) gets its own
            # page rather than sharing the cramped caption strip above -- up to 3 Q&A turns of
            # 2-4 sentences each would overflow that fixed-height area.
            dialogue = entry.get("dialogue") or []
            if dialogue:
                fig, ax = plt.subplots(figsize=(8.5, 11))
                ax.axis("off")
                ax.set_title(
                    f"Iteration {entry['iteration']}: conversation with domain expert",
                    fontsize=13, loc="left",
                )
                lines = []
                for qa_i, qa in enumerate(dialogue, 1):
                    lines.append(f"Q{qa_i}: {textwrap.fill(qa['question'], 95)}")
                    lines.append(f"A{qa_i}: {textwrap.fill(qa['answer'], 95)}")
                    lines.append("")
                ax.text(0, 0.95, "\n".join(lines), va="top", ha="left", fontsize=10, wrap=True)
                pdf.savefig(fig)
                plt.close(fig)

        loops, reached = loops_to_acceptance_by_dialogue(history)
        final = next(h for h in history if h["iteration"] == chosen_iteration)
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis("off")
        ax.set_title("Summary", fontsize=15, fontweight="bold", loc="left")
        loops_line = (
            f"Loops to an accepted result: {loops} of {len(history)} iterations run"
            if reached
            else f"Loops to an accepted result: not reached (all {len(history)} iterations rejected)"
        )
        chosen_line = (
            f"Iteration {chosen_iteration} chosen as final result\n"
            if chosen_iteration != history[-1]["iteration"] else ""
        )
        summary = (
            f"Request: {user_prompt}\n\n"
            f"{loops_line}\n"
            f"{chosen_line}"
            f"Final: {_iteration_count_line(final, counting_model)}\n"
            f"Final accepted: {final['accept']}"
        )
        ax.text(0, 0.95, summary, va="top", ha="left", fontsize=11, wrap=True)
        pdf.savefig(fig)
        plt.close(fig)


def _panel_caption(label: str, count: int, counting_model: str) -> str:
    """StarDist has no text prompt/count-target concept -- it always segments
    every nucleus -- so labeling its result "text: '...' -> count = N" like a
    CountGD result would misrepresent what actually happened."""
    if counting_model == "stardist":
        return f"segmented {count} nuclei"
    text = textwrap.fill(f"text: {label!r}", 40)
    return f"{text}  →  count = {count}"


def side_by_side_page(
    pdf: PdfPages, baseline_image: str, baseline_label: str, baseline_count: int, baseline_counting_model: str,
    agentic_image: str, agentic_label: str, agentic_count: int, agentic_counting_model: str,
) -> None:
    """Same rendering as compare_results.side_by_side_page, but with generic
    "model"/"agent" wording instead of hardcoding "CountGD alone" and
    "Claude + CountGD" — this app treats the counting model and the brain
    behind the agent as swappable, so naming one pair specifically here would
    be misleading regardless of which brain/model a given run actually used.
    Each side's caption is also model-aware (see _panel_caption) since the
    two sides can now each independently be CountGD or StarDist."""
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(11, 6.5))

    ax_left.imshow(Image.open(baseline_image))
    ax_left.axis("off")
    ax_left.set_title(f"Model alone\n{_panel_caption(baseline_label, baseline_count, baseline_counting_model)}", fontsize=10)

    ax_right.imshow(Image.open(agentic_image))
    ax_right.axis("off")
    ax_right.set_title(
        f"Agentic (agent + model)\n{_panel_caption(agentic_label, agentic_count, agentic_counting_model)}", fontsize=10
    )

    fig.suptitle("Model alone vs. agentic result", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    pdf.savefig(fig)
    plt.close(fig)


def _bar_label(prefix: str, label: str, counting_model: str) -> str:
    """Same reasoning as _panel_caption -- a count-target phrase is never
    actually used by StarDist, so showing one under a StarDist bar would
    misrepresent what ran."""
    if counting_model == "stardist":
        return f"{prefix}\n(segmentation)"
    return f"{prefix}\n({textwrap.shorten(label, 30, placeholder='…')!r})"


def comparison_chart_page(
    pdf: PdfPages, baseline_label: str, baseline_count: int, baseline_counting_model: str,
    agentic_label: str, agentic_count: int, agentic_counting_model: str,
) -> None:
    """Same rendering as compare_results.comparison_chart_page, but with
    generic "Model alone" wording instead of hardcoding "CountGD alone", and
    model-aware bar labels since each side can now independently be CountGD
    or StarDist."""
    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    labels = [
        _bar_label("Model alone", baseline_label, baseline_counting_model),
        _bar_label("Agentic", agentic_label, agentic_counting_model),
    ]
    counts = [baseline_count, agentic_count]
    colors = [COLOR_BASELINE, COLOR_AGENTIC]

    bars = ax.bar(labels, counts, color=colors, width=0.5)
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height(),
            str(count), ha="center", va="bottom", fontsize=12, color=INK_PRIMARY,
        )

    ax.set_ylabel("Predicted count", color=INK_SECONDARY)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRIDLINE)
    ax.tick_params(colors=INK_SECONDARY)
    ax.set_title("Count comparison", fontsize=13, fontweight="bold", color=INK_PRIMARY)

    delta = agentic_count - baseline_count
    if baseline_count:
        pct = 100 * delta / baseline_count
        delta_text = f"Δ = {delta:+d} ({pct:+.0f}%)"
    else:
        delta_text = f"Δ = {delta:+d} (baseline found none)"
    fig.text(0.5, 0.01, delta_text, ha="center", fontsize=11, color=INK_SECONDARY)

    fig.tight_layout(rect=(0, 0.04, 1, 1))
    pdf.savefig(fig)
    plt.close(fig)


def missed_cells_page(pdf: PdfPages, baseline_image: str, agentic_image: str, match_radius: float = 12, box_size: int = 22) -> tuple:
    """Same rendering as compare_results.missed_cells_page (box every missed
    cell on both images at matching coordinates), but using get_detection_coords
    so a StarDist side compares on its own real instance centroids instead of
    brightness-peak guessing."""
    baseline_coords = get_detection_coords(baseline_image)
    agentic_coords = get_detection_coords(agentic_image)

    missed_by_baseline = find_missed_points(agentic_coords, baseline_coords, match_radius)
    missed_by_agentic = find_missed_points(baseline_coords, agentic_coords, match_radius)

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(11, 6.5))
    half = box_size / 2

    for ax, image_path, title in (
        (ax_left, baseline_image, f"Model alone (missed {len(missed_by_baseline)})"),
        (ax_right, agentic_image, f"Agentic (missed {len(missed_by_agentic)})"),
    ):
        ax.imshow(Image.open(image_path))
        ax.axis("off")
        ax.set_title(title, fontsize=11)
        for row, col in missed_by_baseline:
            ax.add_patch(plt.Rectangle(
                (col - half, row - half), box_size, box_size,
                edgecolor=COLOR_MISSED_BY_BASELINE, facecolor="none", linewidth=1.5,
            ))
        for row, col in missed_by_agentic:
            ax.add_patch(plt.Rectangle(
                (col - half, row - half), box_size, box_size,
                edgecolor=COLOR_MISSED_BY_AGENTIC, facecolor="none", linewidth=1.5,
            ))

    fig.suptitle("Missed-cell diff (adaptive dot-detection heuristic)", fontsize=14, fontweight="bold")
    legend_handles = [
        plt.Line2D([0], [0], color=COLOR_MISSED_BY_BASELINE, lw=1.5,
                   label="found by agentic, missed by model alone"),
        plt.Line2D([0], [0], color=COLOR_MISSED_BY_AGENTIC, lw=1.5,
                   label="found by model alone, missed by agentic"),
    ]
    legend = fig.legend(handles=legend_handles, loc="lower center", ncol=1, fontsize=9, frameon=True)
    legend.get_frame().set_facecolor(LEGEND_BACKING)
    legend.get_frame().set_edgecolor("none")
    for text in legend.get_texts():
        text.set_color("#ffffff")
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    pdf.savefig(fig)
    plt.close(fig)

    return missed_by_baseline, missed_by_agentic


def deepgleason_summary_line(gleason_result: dict) -> str:
    if gleason_result["tumor_found"]:
        return (
            f"Tumor/cancer was found, and it appears to be Gleason score {gleason_result['gleason_score']} "
            f"(primary pattern {gleason_result['primary_pattern']}, secondary pattern "
            f"{gleason_result['secondary_pattern']}), which is grade group {gleason_result['isup_grade']} "
            f"on the ISUP/Gleason scale."
        )
    return "No tumor/cancer was found in this slide."


def save_deepgleason_pdf_report(pdf_path: Path, prompt: str, result: dict) -> None:
    """DeepGleason's retry loop (run_deepgleason_with_feedback) never reruns the underlying
    model -- every iteration just re-aggregates the same cached tile predictions with a
    different confidence_threshold (see manager_agent.py's module docstring) -- so there's one
    shared overlay preview image, not a fresh one per iteration like CountGD/StarDist/CellViT.
    Renders an overview page (shared preview + final Gleason score/ISUP grade + tile-class bar
    chart), then one text-only page per iteration (confidence_threshold/result/accept/feedback)
    plus its own manager<->expert dialogue page -- the dialogue is always present now, same as
    every other agent (see run_expert_dialogue)."""
    history = result["history"]
    final_gleason = result["gleason_result"]
    preview_path = result["overlay_preview"]
    loops, reached = loops_to_acceptance_by_dialogue(history)

    with PdfPages(pdf_path) as pdf:
        fig, (ax_img, ax_text) = plt.subplots(2, 1, figsize=(8.5, 11), gridspec_kw={"height_ratios": [3, 2]})
        ax_img.imshow(Image.open(preview_path))
        ax_img.axis("off")
        ax_img.set_title("DeepGleason tumor detection result")
        ax_text.axis("off")
        loops_line = (
            f"Loops to an accepted result: {loops} of {len(history)} iterations run"
            if reached
            else f"Loops to an accepted result: not reached (all {len(history)} iterations rejected)"
        )
        text = (
            f"Request: {prompt}\n\n"
            f"{deepgleason_summary_line(final_gleason)}\n\n"
            f"{loops_line}\n"
            f"Tiles analyzed: {final_gleason['total_tiles']}\n"
            f"Final confidence_threshold: {result['confidence_threshold']:.2f}"
        )
        ax_text.text(0, 1, text, va="top", ha="left", fontsize=11, wrap=True)
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8.5, 5.5))
        classes = agentic_deepgleason.TILE_CLASSES
        counts = [final_gleason["tile_counts"].get(c, 0) for c in classes]
        bar_labels = [agentic_deepgleason.CLASS_LABELS[c] for c in classes]
        colors = [
            "#9aa0a6" if c in ("A_S", "A_D") else COLOR_BASELINE if c == "R" else COLOR_AGENTIC
            for c in classes
        ]
        ax.bar(bar_labels, counts, color=colors)
        ax.set_ylabel("Tile count", color=INK_SECONDARY)
        ax.set_title("Tile classification breakdown", fontsize=13, fontweight="bold", color=INK_PRIMARY)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="x", rotation=30, colors=INK_SECONDARY)
        ax.tick_params(axis="y", colors=INK_SECONDARY)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        for entry in history:
            fig, ax = plt.subplots(figsize=(8.5, 11))
            ax.axis("off")
            ax.set_title(f"Iteration {entry['iteration']}: confidence_threshold={entry['confidence_threshold']:.2f}")
            text = (
                f"Result: {deepgleason_summary_line(entry['gleason_result'])}\n\n"
                f"Accepted: {entry['accept']}\n"
                f"Feedback: {textwrap.fill(entry['feedback'], 100)}"
            )
            ax.text(0, 0.95, text, va="top", ha="left", fontsize=11, wrap=True)
            pdf.savefig(fig)
            plt.close(fig)

            dialogue = entry.get("dialogue") or []
            if dialogue:
                fig, ax = plt.subplots(figsize=(8.5, 11))
                ax.axis("off")
                ax.set_title(
                    f"Iteration {entry['iteration']}: conversation with domain expert",
                    fontsize=13, loc="left",
                )
                lines = []
                for qa_i, qa in enumerate(dialogue, 1):
                    lines.append(f"Q{qa_i}: {textwrap.fill(qa['question'], 95)}")
                    lines.append(f"A{qa_i}: {textwrap.fill(qa['answer'], 95)}")
                    lines.append("")
                ax.text(0, 0.95, "\n".join(lines), va="top", ha="left", fontsize=10, wrap=True)
                pdf.savefig(fig)
                plt.close(fig)

    return {"result": result, "summary_line": summary_line, "pdf_path": pdf_path}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/agentic", methods=["GET", "POST"])
def agentic():
    if request.method == "GET":
        return render_template("agentic.html")

    image = request.files.get("image")
    prompt = (request.form.get("prompt") or "").strip()
    if not image or not image.filename:
        flash("Please choose an image.")
        return redirect(url_for("agentic"))
    if not prompt:
        flash("Please describe the task.")
        return redirect(url_for("agentic"))

    run_id = uuid.uuid4().hex
    run_dir = OUTPUT_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    try:
        image_path = save_upload(image, UPLOAD_DIR)
    except ValueError as exc:
        flash(str(exc))
        return redirect(url_for("agentic"))

    RUNS[run_id] = {"status": "running", "progress": 0, "stage": "Starting…"}
    cancel_event = get_cancel_event(run_id)

    def worker():
        try:
            # Routing (which of countgd/stardist/cellvit/deepgleason fits this task) now
            # happens in a single select_agent() call inside run_agentic_pipeline's own
            # manager.run() -- no separate task-type pre-check needed anymore.
            result = run_agentic_pipeline(
                image_path, prompt, run_dir, run_id,
                progress_cb=lambda frac, stage: set_progress(run_id, frac, stage),
                cancel_event=cancel_event,
            )
            with RUNS_LOCK:
                RUNS[run_id] = {
                    "status": "done",
                    "progress": 100,
                    "stage": "Done",
                    "title": f"{result['agent'].capitalize()} result",
                    "pdf_filename": result["pdf_path"].name,
                    "summary": [("Prompt", prompt)] + result["summary_rows"],
                }
        except RunCancelled:
            with RUNS_LOCK:
                RUNS[run_id] = {"status": "cancelled", "progress": 0, "stage": "Stopped by request."}
        except Exception as exc:
            # str(exc) alone (the only thing previously shown/stored) often isn't enough to find
            # the actual failing line -- e.g. a bare "[Errno 2] No such file or directory: ..."
            # gives the missing path but not which of several path-construction sites produced
            # it. Print the full traceback to the server's own stdout (never to the website
            # itself -- a raw traceback isn't something to show an end user) so it's available
            # in the terminal/log the next time this fires.
            print(f"\n!!! run {run_id} failed:")
            traceback.print_exc()
            with RUNS_LOCK:
                RUNS[run_id] = {"status": "error", "progress": 0, "stage": str(exc)}
        finally:
            with CANCEL_LOCK:
                CANCEL_EVENTS.pop(run_id, None)

    threading.Thread(target=worker, daemon=True).start()

    return render_template(
        "progress.html", title="Running agentic", run_id=run_id, back_url=url_for("agentic"),
    )


@app.route("/compare", methods=["GET", "POST"])
def compare():
    if request.method == "GET":
        return render_template("compare.html")

    baseline_label_input = (request.form.get("baseline_label") or "").strip()

    # Agentic side: "run" calls Claude + CountGD here; "finished" is an
    # already-produced agentic result you just want compared as-is.
    agentic_source_mode = request.form.get("agentic_source_mode") or "run"
    if agentic_source_mode not in ("run", "finished"):
        agentic_source_mode = "run"

    # Baseline side: "finished" = user already ran CountGD elsewhere and
    # uploads the result image + count. "run" = we call CountGD's own
    # /count_main endpoint here — the exact same call the public Space's UI
    # makes — so the result matches what you'd get running
    # https://huggingface.co/spaces/nikigoli/countgd by hand with that same
    # image and text, with no Claude involved.
    baseline_mode = request.form.get("baseline_mode") or "run"
    if baseline_mode not in ("finished", "run"):
        baseline_mode = "run"

    raw_image = None
    prompt = None
    agentic_finished_image = None
    agentic_finished_count = None
    agentic_finished_label_input = None

    if agentic_source_mode == "run":
        raw_image = request.files.get("raw_image")
        prompt = (request.form.get("prompt") or "").strip()
        if not raw_image or not raw_image.filename:
            flash("Please choose the image to run the agent on.")
            return redirect(url_for("compare"))
        if not prompt:
            flash("Please describe the task.")
            return redirect(url_for("compare"))
    else:
        agentic_finished_image = request.files.get("agentic_finished_image")
        # Optional — estimated from the image if left blank, same as the
        # baseline count below.
        agentic_finished_count = request.form.get("agentic_finished_count", type=int)
        agentic_finished_label_input = (request.form.get("agentic_finished_label") or "").strip()
        if not agentic_finished_image or not agentic_finished_image.filename:
            flash("Please choose the finished agentic result image.")
            return redirect(url_for("compare"))

    baseline_image = None
    baseline_count = None
    baseline_raw_image = None
    baseline_text = None

    if baseline_mode == "finished":
        baseline_image = request.files.get("baseline_image")
        # Optional — if left blank, the worker estimates it directly from
        # the image using the same adaptive dot-detector as the missed-cells
        # diff, rather than blocking submission on a required number.
        baseline_count = request.form.get("baseline_count", type=int)
        if not baseline_image or not baseline_image.filename:
            flash("Please choose the finished baseline image.")
            return redirect(url_for("compare"))
        # compare_results.py's "label" IS the text prompt used, by design
        # (see its own --baseline-label help: "Text prompt used for the
        # baseline run") — for an already-finished result we don't know
        # that text, so the generic default is the best we can show.
        baseline_label = baseline_label_input or "model alone"
    else:
        baseline_raw_image = request.files.get("baseline_raw_image")
        baseline_text = (request.form.get("baseline_text") or "").strip()
        if not baseline_raw_image or not baseline_raw_image.filename:
            flash("Please choose an image to run the model on.")
            return redirect(url_for("compare"))
        if not baseline_text:
            flash("Please describe the task.")
            return redirect(url_for("compare"))
        # Here we DO know the real text — default the PDF's label to it
        # instead of the generic "model alone" placeholder, unless the
        # user explicitly typed a different display label.
        baseline_label = baseline_label_input or baseline_text

    run_id = uuid.uuid4().hex
    run_dir = OUTPUT_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    try:
        if agentic_source_mode == "run":
            raw_image_path = save_upload(raw_image, UPLOAD_DIR)
        else:
            agentic_finished_image_path = save_upload(agentic_finished_image, UPLOAD_DIR)

        if baseline_mode == "finished":
            baseline_image_path = save_upload(baseline_image, UPLOAD_DIR)
        else:
            baseline_raw_image_path = save_upload(baseline_raw_image, UPLOAD_DIR)
    except ValueError as exc:
        flash(str(exc))
        return redirect(url_for("compare"))

    RUNS[run_id] = {"status": "running", "progress": 0, "stage": "Starting…"}
    cancel_event = get_cancel_event(run_id)

    def worker():
        try:
            # --- Baseline side: 0.00 - 0.15 ---
            if baseline_mode == "run":
                check_cancelled(cancel_event)
                set_progress(run_id, 0.02, "Deciding which model fits this task…")
                set_progress(run_id, 0.03, f"Running the model on “{baseline_text}”…")
                baseline_once = run_baseline_once(
                    get_manager(), baseline_text, str(baseline_raw_image_path), run_dir
                )
                baseline_counting_model = baseline_once["agent"]
                resolved_baseline_count = baseline_once["count"]
                baseline_suffix = Path(baseline_once["annotated_image"]).suffix or ".png"
                resolved_baseline_image_path = run_dir / f"baseline{baseline_suffix}"
                resolved_baseline_image_path.write_bytes(Path(baseline_once["annotated_image"]).read_bytes())
                # Save StarDist's real per-instance label array as a .npy sidecar (same stem) so
                # the missed-cell diff can use real instance centroids, same fix as
                # run_agentic_pipeline's own -- run_baseline_once's StardistWorker.run() only
                # writes the rendered outline PNG itself.
                if baseline_once.get("labels") is not None:
                    np.save(resolved_baseline_image_path.with_suffix(".npy"), baseline_once["labels"])
                if baseline_counting_model == "stardist":
                    set_progress(run_id, 0.15, f"StarDist segmented {resolved_baseline_count} nuclei")
                else:
                    set_progress(run_id, 0.15, f"Model found {resolved_baseline_count}")
                baseline_count_estimated = False
            else:
                baseline_counting_model = "unknown"  # a finished result was uploaded -- we don't know what produced it
                resolved_baseline_image_path = baseline_image_path
                baseline_count_estimated = baseline_count is None
                if baseline_count_estimated:
                    check_cancelled(cancel_event)
                    set_progress(run_id, 0.03, "No baseline count given — estimating it from the image…")
                    resolved_baseline_count = len(detect_dot_coords_adaptive(str(resolved_baseline_image_path)))
                    set_progress(run_id, 0.15, f"Estimated {resolved_baseline_count} from the image")
                else:
                    resolved_baseline_count = baseline_count

            # --- Agentic side: 0.15 - 0.80 ---
            if agentic_source_mode == "run":
                agentic_result = run_agentic_pipeline(
                    raw_image_path, prompt, run_dir, run_id,
                    # Scale the agentic pipeline's own 0-1 progress into the
                    # remaining span before PDF-building starts at 80%.
                    progress_cb=lambda frac, stage: set_progress(run_id, 0.15 + frac * 0.65, stage),
                    cancel_event=cancel_event,
                )
                if agentic_result["agent"] == "deepgleason":
                    # run_agentic_pipeline doesn't populate final_image_path/count_target/
                    # final_count for deepgleason (no single count to compare -- see its own
                    # docstring), same reasoning as run_baseline_once's own deepgleason guard.
                    raise ValueError(
                        "This image/task routed to DeepGleason (whole-slide tumor grading), "
                        "which has no single count to compare -- the Compare page only supports "
                        "CountGD/StarDist/CellViT. Try the main Agentic Count page instead."
                    )
                agentic_count_estimated = False
            else:
                check_cancelled(cancel_event)
                agentic_count_estimated = agentic_finished_count is None
                if agentic_count_estimated:
                    set_progress(run_id, 0.3, "No agentic count given — estimating it from the image…")
                    resolved_agentic_count = len(detect_dot_coords_adaptive(str(agentic_finished_image_path)))
                else:
                    resolved_agentic_count = agentic_finished_count
                set_progress(run_id, 0.80, "Agentic side ready")
                agentic_result = {
                    "final_image_path": agentic_finished_image_path,
                    "count_target": agentic_finished_label_input or "finished result",
                    "final_count": resolved_agentic_count,
                    "counting_model": "unknown",  # a finished result was uploaded -- we don't know what produced it
                }

            check_cancelled(cancel_event)
            set_progress(run_id, 0.85, "Building side-by-side comparison…")
            pdf_path = run_dir / "comparison.pdf"
            with PdfPages(pdf_path) as pdf:
                side_by_side_page(
                    pdf,
                    str(resolved_baseline_image_path), baseline_label, resolved_baseline_count, baseline_counting_model,
                    str(agentic_result["final_image_path"]), agentic_result["count_target"],
                    agentic_result["final_count"], agentic_result["counting_model"],
                )
                check_cancelled(cancel_event)
                set_progress(run_id, 0.90, "Charting the count comparison…")
                comparison_chart_page(
                    pdf,
                    baseline_label, resolved_baseline_count, baseline_counting_model,
                    agentic_result["count_target"], agentic_result["final_count"], agentic_result["counting_model"],
                )
                check_cancelled(cancel_event)
                set_progress(run_id, 0.95, "Diffing missed detections…")
                missed_by_baseline, missed_by_agentic = missed_cells_page(
                    pdf, str(resolved_baseline_image_path), str(agentic_result["final_image_path"]),
                )

            set_progress(run_id, 1.0, "Done")
            summary = []
            if agentic_source_mode == "run":
                summary.append(("Prompt", prompt))
                summary.append(("Model used (agentic)", agentic_result["counting_model"]))
            else:
                summary.append(("Model used (agentic)", "n/a — finished result provided, not run here"))
            summary.append(("Baseline label", baseline_label))
            if baseline_mode == "run":
                summary.append(("Model used (baseline)", baseline_counting_model))
                summary.append(("Task (baseline)", baseline_text))
            else:
                summary.append(("Model used (baseline)", "n/a — finished result provided, not run here"))

            baseline_count_display = (
                f"{resolved_baseline_count} (estimated from image — no count was entered)"
                if baseline_count_estimated else resolved_baseline_count
            )
            agentic_count_display = (
                f"{agentic_result['final_count']} (estimated from image — no count was entered)"
                if agentic_count_estimated else agentic_result["final_count"]
            )
            summary.append(("Baseline count", baseline_count_display))
            # count_target is meaningless for StarDist (it always segments every
            # nucleus, ignoring any target phrase) -- drop the row instead of
            # showing a noun phrase that was never actually used for anything.
            if agentic_result["counting_model"] != "stardist":
                summary.append(("Agentic count target", agentic_result["count_target"]))
            summary += [
                ("Agentic count", agentic_count_display),
                ("Delta", agentic_result["final_count"] - resolved_baseline_count),
                ("Missed by baseline", len(missed_by_baseline)),
                ("Missed by agentic", len(missed_by_agentic)),
            ]
            with RUNS_LOCK:
                RUNS[run_id] = {
                    "status": "done",
                    "progress": 100,
                    "stage": "Done",
                    "title": "Comparison result",
                    "pdf_filename": pdf_path.name,
                    "summary": summary,
                }
        except RunCancelled:
            with RUNS_LOCK:
                RUNS[run_id] = {"status": "cancelled", "progress": 0, "stage": "Stopped by request."}
        except Exception as exc:
            # str(exc) alone (the only thing previously shown/stored) often isn't enough to find
            # the actual failing line -- e.g. a bare "[Errno 2] No such file or directory: ..."
            # gives the missing path but not which of several path-construction sites produced
            # it. Print the full traceback to the server's own stdout (never to the website
            # itself -- a raw traceback isn't something to show an end user) so it's available
            # in the terminal/log the next time this fires.
            print(f"\n!!! run {run_id} failed:")
            traceback.print_exc()
            with RUNS_LOCK:
                RUNS[run_id] = {"status": "error", "progress": 0, "stage": str(exc)}
        finally:
            with CANCEL_LOCK:
                CANCEL_EVENTS.pop(run_id, None)

    threading.Thread(target=worker, daemon=True).start()

    return render_template(
        "progress.html", title="Building comparison", run_id=run_id, back_url=url_for("compare"),
    )


@app.route("/progress/<run_id>")
def progress_status(run_id):
    """Polled by progress.html — returns the background thread's latest checkpoint."""
    with RUNS_LOCK:
        info = RUNS.get(run_id)
        if info is None:
            return jsonify({"status": "error", "progress": 0, "stage": "Unknown run"}), 404
        payload = {
            "status": info["status"],
            "progress": info.get("progress", 0),
            "stage": info.get("stage", ""),
        }
        if info["status"] == "done":
            payload["redirect"] = url_for("show_result", run_id=run_id)
    return jsonify(payload)


@app.route("/cancel/<run_id>", methods=["POST"])
def cancel_run(run_id):
    """Requests a stop at the next pipeline checkpoint — not immediate if a
    Claude/CountGD call is already in flight, that call still completes."""
    print(f"[cancel] /cancel/{run_id} hit — request came from the Stop button, not spawned internally")
    with RUNS_LOCK:
        info = RUNS.get(run_id)
        if info is None:
            return jsonify({"status": "error", "stage": "Unknown run"}), 404
        if info["status"] != "running":
            return jsonify({"status": info["status"], "stage": info.get("stage", "")})
        info["stage"] = "Stopping…"
    get_cancel_event(run_id).set()
    return jsonify({"status": "running", "stage": "Stopping…"})


@app.route("/result/<run_id>")
def show_result(run_id):
    with RUNS_LOCK:
        info = RUNS.get(run_id)
    if not info or info.get("status") != "done":
        return "Result not found or not ready yet.", 404
    return render_template(
        "result.html",
        title=info["title"],
        pdf_url=url_for("view_pdf", run_id=run_id, filename=info["pdf_filename"]),
        summary=info["summary"],
    )


@app.route("/pdf/<run_id>/<filename>")
def view_pdf(run_id, filename):
    """Serve a generated PDF for inline viewing in the browser (no forced download)."""
    pdf_path = OUTPUT_DIR / Path(run_id).name / Path(filename).name
    if not pdf_path.is_file():
        return "Not found", 404
    response = send_file(pdf_path, mimetype="application/pdf", as_attachment=False)
    response.headers["Content-Disposition"] = f'inline; filename="{pdf_path.name}"'
    return response


if __name__ == "__main__":
    # threaded=True is required now — the background pipeline thread and the
    # browser's /progress polling both need to be served concurrently, not
    # queued behind each other.
    # host="0.0.0.0" makes this reachable from other devices on the same
    # network, not just this machine. debug=False because Werkzeug's
    # interactive debugger allows remote code execution if it's reachable
    # by anyone other than you.
    app.run(host="0.0.0.0", port=5050, debug=False, threaded=True)
