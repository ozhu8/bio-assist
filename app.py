"""
Flask front end for the CountGD tools.

Wraps the existing pipelines in agentic_countgd.py and compare_results.py
(neither of those files is modified) so results can be viewed as an inline
PDF in the browser instead of run from the CLI:

  - /agentic  upload one image + what to count -> Claude+CountGD PDF report
  - /compare  upload a raw image (run through Claude+CountGD) alongside an
              already-finished CountGD-alone image + its count -> a
              side-by-side comparison PDF

Both routes hand the actual pipeline run off to a background thread and
return a progress page immediately; the browser polls /progress/<run_id>
for status and redirects to /result/<run_id> once it's done. RUNS is an
in-memory job store — fine for a single-process dev server, but progress
state is lost on restart and never evicted.

Run with: python app.py
"""
import mimetypes 
import os
import threading
import uuid
from pathlib import Path

# Must run before anything (including agentic_countgd/compare_results) imports
# matplotlib.pyplot. On macOS, matplotlib's default interactive backend talks
# to Cocoa/AppKit, which crashes the whole process (uncaught NSException) if
# touched from a background thread — and the pipeline now always runs in one.
# "Agg" is the correct non-interactive backend anyway since we only ever
# write PDFs to disk, never show a window.
import matplotlib # pyright: ignore[reportMissingModuleSource]
matplotlib.use("Agg")

import anthropic # pyright: ignore[reportMissingImports]
import matplotlib.pyplot as plt # pyright: ignore[reportMissingModuleSource]
import numpy as np # pyright: ignore[reportMissingImports]
from flask import Flask, flash, jsonify, redirect, render_template, request, send_file, url_for # pyright: ignore[reportMissingImports]
from gradio_client import Client # pyright: ignore[reportMissingImports]
from matplotlib.backends.backend_pdf import PdfPages # pyright: ignore[reportMissingModuleSource]
from PIL import Image # pyright: ignore[reportMissingImports]
from scipy import ndimage # pyright: ignore[reportMissingImports]
from skimage.feature import peak_local_max # pyright: ignore[reportMissingModuleSource]

from agentic_countgd import (
    COUNTGD_SPACE,
    evaluate_result,
    interpret_prompt,
    run_countgd,
    save_pdf_report,
)
from compare_results import (
    COLOR_MISSED_BY_AGENTIC,
    COLOR_MISSED_BY_BASELINE,
    LEGEND_BACKING,
    comparison_chart_page,
    find_missed_points,
    side_by_side_page,
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

ALLOWED_IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp"}
MAX_ITERATIONS = 3

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-only-not-secret")

_claude = None
_countgd = None

# run_id -> {"status": "running"|"done"|"error"|"cancelled", "progress": 0-100, "stage": str, ...}
RUNS = {}
RUNS_LOCK = threading.Lock()

# run_id -> threading.Event(), set by POST /cancel/<run_id> and checked at
# each pipeline checkpoint (see check_cancelled). Cooperative, not forced —
# a call already in flight to Claude/CountGD finishes before the next check.
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


def get_claude() -> anthropic.Anthropic:
    global _claude
    if _claude is None:
        # Higher than the SDK default (2) so transient 529 overloads during a
        # multi-iteration run don't need a full manual retry from the browser.
        _claude = anthropic.Anthropic(max_retries=5)
    return _claude


def get_countgd() -> Client:
    global _countgd
    if _countgd is None:
        _countgd = Client(COUNTGD_SPACE)
    return _countgd


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


def run_agentic_pipeline(
    image_path: Path, prompt: str, run_dir: Path,
    max_iterations: int = MAX_ITERATIONS, progress_cb=None, cancel_event=None,
) -> dict:
    """Same loop as agentic_countgd.py's main(), calling its exported functions
    directly, so this app gets a plain return value without touching that file.

    progress_cb(fraction, stage_text), if given, is called at each real
    pipeline checkpoint — fraction is 0-1 progress through THIS function
    (interpret_prompt -> up to max_iterations CountGD+eval rounds -> PDF).
    cancel_event, if given, is checked at the same checkpoints — raises
    RunCancelled between steps rather than mid-API-call."""
    def report(fraction, stage):
        check_cancelled(cancel_event)
        if progress_cb:
            progress_cb(fraction, stage)

    claude = get_claude()
    countgd = get_countgd()

    report(0.03, "Reading your image and figuring out what to count…")
    count_target = interpret_prompt(claude, prompt, str(image_path))
    report(0.10, f"Starting with count target: “{count_target}”")

    history = []
    saved_paths = []
    saved_path = None
    predicted_count = None
    for i in range(1, max_iterations + 1):
        iter_start = 0.10 + (i - 1) / max_iterations * 0.80
        iter_mid = 0.10 + (i - 1 + 0.5) / max_iterations * 0.80
        iter_end = 0.10 + i / max_iterations * 0.80

        report(iter_start, f"Iteration {i}: running CountGD on “{count_target}”…")
        annotated_path, predicted_count = run_countgd(countgd, str(image_path), count_target)

        # Preserve CountGD's actual output format (e.g. .webp) instead of
        # hardcoding .png — a mismatched extension makes evaluate_result's
        # declared image media type wrong, which the API rejects.
        annotated_suffix = Path(annotated_path).suffix or ".png"
        saved_path = run_dir / f"iteration_{i}{annotated_suffix}"
        saved_path.write_bytes(Path(annotated_path).read_bytes())
        saved_paths.append(saved_path)

        report(iter_mid, f"Iteration {i}: CountGD found {predicted_count} — Claude is evaluating…")
        eval_result = evaluate_result(
            claude, prompt, count_target, predicted_count, str(saved_path), history
        )
        history.append({
            "iteration": i,
            "count_target": count_target,
            "predicted_count": predicted_count,
            "score": eval_result["score"],
            "feedback": eval_result["feedback"],
        })
        report(iter_end, f"Iteration {i} scored {eval_result['score']}/10")

        if eval_result["accept"] or not eval_result.get("revised_text"):
            break
        count_target = eval_result["revised_text"]

    report(0.94, "Building PDF report…")
    pdf_path = run_dir / "countgd_results.pdf"
    save_pdf_report(pdf_path, prompt, saved_paths, history)
    report(1.0, "Done")

    return {
        "final_count": predicted_count,
        "final_image_path": saved_path,
        "count_target": count_target,
        "pdf_path": pdf_path,
        "history": history,
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


def missed_cells_page(pdf: PdfPages, baseline_image: str, agentic_image: str, match_radius: float = 12, box_size: int = 22) -> tuple:
    """Same rendering as compare_results.missed_cells_page (box every missed
    cell on both images at matching coordinates), but with the adaptive
    detector above instead of that module's fixed-threshold one."""
    baseline_coords = detect_dot_coords_adaptive(baseline_image)
    agentic_coords = detect_dot_coords_adaptive(agentic_image)

    missed_by_baseline = find_missed_points(agentic_coords, baseline_coords, match_radius)
    missed_by_agentic = find_missed_points(baseline_coords, agentic_coords, match_radius)

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(11, 6.5))
    half = box_size / 2

    for ax, image_path, title in (
        (ax_left, baseline_image, f"CountGD alone (missed {len(missed_by_baseline)})"),
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
                   label="found by agentic, missed by CountGD alone"),
        plt.Line2D([0], [0], color=COLOR_MISSED_BY_AGENTIC, lw=1.5,
                   label="found by CountGD alone, missed by agentic"),
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
            result = run_agentic_pipeline(
                image_path, prompt, run_dir,
                progress_cb=lambda frac, stage: set_progress(run_id, frac, stage),
                cancel_event=cancel_event,
            )
            with RUNS_LOCK:
                RUNS[run_id] = {
                    "status": "done",
                    "progress": 100,
                    "stage": "Done",
                    "title": "Agentic count result",
                    "pdf_filename": result["pdf_path"].name,
                    "summary": [
                        ("Prompt", prompt),
                        ("Count target CountGD used", result["count_target"]),
                        ("Final count", result["final_count"]),
                        ("Iterations run", len(result["history"])),
                    ],
                }
        except RunCancelled:
            with RUNS_LOCK:
                RUNS[run_id] = {"status": "cancelled", "progress": 0, "stage": "Stopped by request."}
        except Exception as exc:
            with RUNS_LOCK:
                RUNS[run_id] = {"status": "error", "progress": 0, "stage": str(exc)}
        finally:
            with CANCEL_LOCK:
                CANCEL_EVENTS.pop(run_id, None)

    threading.Thread(target=worker, daemon=True).start()

    return render_template(
        "progress.html", title="Running agentic count", run_id=run_id, back_url=url_for("agentic"),
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
            flash("Please choose the image to run Claude + CountGD on.")
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
            flash("Please choose the finished CountGD baseline image.")
            return redirect(url_for("compare"))
        # compare_results.py's "label" IS the text prompt used, by design
        # (see its own --baseline-label help: "Text prompt used for the
        # baseline run") — for an already-finished result we don't know
        # that text, so the generic default is the best we can show.
        baseline_label = baseline_label_input or "CountGD alone"
    else:
        baseline_raw_image = request.files.get("baseline_raw_image")
        baseline_text = (request.form.get("baseline_text") or "").strip()
        if not baseline_raw_image or not baseline_raw_image.filename:
            flash("Please choose an image to run CountGD alone on.")
            return redirect(url_for("compare"))
        if not baseline_text:
            flash("Please enter the text CountGD should count.")
            return redirect(url_for("compare"))
        # Here we DO know the real text — default the PDF's label to it
        # instead of the generic "CountGD alone" placeholder, unless the
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
                set_progress(run_id, 0.03, f"Running CountGD alone on “{baseline_text}”…")
                baseline_annotated_path, resolved_baseline_count = run_countgd(
                    get_countgd(), str(baseline_raw_image_path), baseline_text
                )
                baseline_suffix = Path(baseline_annotated_path).suffix or ".png"
                resolved_baseline_image_path = run_dir / f"baseline{baseline_suffix}"
                resolved_baseline_image_path.write_bytes(Path(baseline_annotated_path).read_bytes())
                set_progress(run_id, 0.15, f"CountGD alone found {resolved_baseline_count}")
                baseline_count_estimated = False
            else:
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
                    raw_image_path, prompt, run_dir,
                    # Scale the agentic pipeline's own 0-1 progress into the
                    # remaining span before PDF-building starts at 80%.
                    progress_cb=lambda frac, stage: set_progress(run_id, 0.15 + frac * 0.65, stage),
                    cancel_event=cancel_event,
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
                }

            check_cancelled(cancel_event)
            set_progress(run_id, 0.85, "Building side-by-side comparison…")
            pdf_path = run_dir / "comparison.pdf"
            with PdfPages(pdf_path) as pdf:
                side_by_side_page(
                    pdf,
                    str(resolved_baseline_image_path), baseline_label, resolved_baseline_count,
                    str(agentic_result["final_image_path"]), agentic_result["count_target"],
                    agentic_result["final_count"],
                )
                check_cancelled(cancel_event)
                set_progress(run_id, 0.90, "Charting the count comparison…")
                comparison_chart_page(
                    pdf,
                    baseline_label, resolved_baseline_count,
                    agentic_result["count_target"], agentic_result["final_count"],
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
            summary.append(("Baseline label", baseline_label))
            if baseline_mode == "run":
                summary.append(("Text sent to CountGD (baseline)", baseline_text))

            baseline_count_display = (
                f"{resolved_baseline_count} (estimated from image — no count was entered)"
                if baseline_count_estimated else resolved_baseline_count
            )
            agentic_count_display = (
                f"{agentic_result['final_count']} (estimated from image — no count was entered)"
                if agentic_count_estimated else agentic_result["final_count"]
            )
            summary += [
                ("Baseline count", baseline_count_display),
                ("Agentic count target", agentic_result["count_target"]),
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
