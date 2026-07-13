"""
Compare a plain CountGD-alone result against an agentic (Claude + CountGD)
result on the same image, and render a side-by-side PDF report highlighting
the differences.

Both results must already exist (annotated image + predicted count) — this
script does not call CountGD or Claude itself, it just compares.

Usage:
    python compare_results.py \\
        --baseline-image countgd_agent_output_cells3/iteration_1.png --baseline-count 0 --baseline-label "cell" \\
        --agentic-image countgd_agent_output_cells3/iteration_3.png --agentic-count 148 --agentic-label "bacteria" \\
        --pdf-name comparison.pdf
"""
import argparse
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt # pyright: ignore[reportMissingModuleSource]
import numpy as np # pyright: ignore[reportMissingImports]
from matplotlib.backends.backend_pdf import PdfPages # pyright: ignore[reportMissingModuleSource]
from PIL import Image # pyright: ignore[reportMissingImports]
from scipy import ndimage # pyright: ignore[reportMissingImports]
from scipy.spatial import cKDTree # pyright: ignore[reportMissingImports]
from skimage.feature import peak_local_max # pyright: ignore[reportMissingModuleSource]

PDF_NAME = "comparison_results.pdf"

# Categorical slots from the shared palette, assigned in fixed order.
COLOR_BASELINE = "#2a78d6"  # slot 1: blue
COLOR_AGENTIC = "#1baf7a"   # slot 2: aqua
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRIDLINE = "#e1e0d9"


def detect_dot_coords(image_path: str, min_distance: int = 6, threshold_abs: float = 40, sigma: float = 1.2):
    """Find CountGD's detection-dot centers by their yellow/green core against the blue overlay.

    Returns an (N, 2) array of (row, col) pixel coordinates. This is a heuristic
    tuned for CountGD's dot-style visualization — it won't work on bounding-box
    style output.
    """
    img = Image.open(image_path).convert("RGB")
    arr = np.array(img).astype(float)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    yellowness = ndimage.gaussian_filter((r + g) - 2 * b, sigma=sigma)
    return peak_local_max(yellowness, min_distance=min_distance, threshold_abs=threshold_abs)


def find_missed_points(reference_coords, other_coords, match_radius: float = 12):
    """Points in reference_coords with no match in other_coords within match_radius pixels."""
    if len(reference_coords) == 0:
        return np.empty((0, 2))
    if len(other_coords) == 0:
        return reference_coords
    tree = cKDTree(other_coords)
    dist, _ = tree.query(reference_coords, k=1)
    return reference_coords[dist > match_radius]


def side_by_side_page(
    pdf: PdfPages,
    baseline_image: str,
    baseline_label: str,
    baseline_count: int,
    agentic_image: str,
    agentic_label: str,
    agentic_count: int,
) -> None:
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(11, 6.5))

    baseline_text = textwrap.fill(f"text: {baseline_label!r}", 40)
    agentic_text = textwrap.fill(f"text: {agentic_label!r}", 40)

    ax_left.imshow(Image.open(baseline_image))
    ax_left.axis("off")
    ax_left.set_title(f"CountGD alone\n{baseline_text}  →  count = {baseline_count}", fontsize=10)

    ax_right.imshow(Image.open(agentic_image))
    ax_right.axis("off")
    ax_right.set_title(f"Agentic (Claude + CountGD)\n{agentic_text}  →  count = {agentic_count}", fontsize=10)

    fig.suptitle("CountGD alone vs. agentic result", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    pdf.savefig(fig)
    plt.close(fig)


def comparison_chart_page(
    pdf: PdfPages,
    baseline_label: str,
    baseline_count: int,
    agentic_label: str,
    agentic_count: int,
) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    labels = [
        f"CountGD alone\n({textwrap.shorten(baseline_label, 30, placeholder='…')!r})",
        f"Agentic\n({textwrap.shorten(agentic_label, 30, placeholder='…')!r})",
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


COLOR_MISSED_BY_BASELINE = "#e34948"  # red — agentic found it, baseline missed it
COLOR_MISSED_BY_AGENTIC = "#ffffff"   # white — baseline found it, agentic missed it
LEGEND_BACKING = "#52514e"            # dark neutral gray so the white swatch is clearly visible on the page


def missed_cells_page(
    pdf: PdfPages,
    baseline_image: str,
    agentic_image: str,
    match_radius: float = 12,
    box_size: int = 22,
) -> tuple:
    """Box every missed cell on BOTH images, at the same coordinates, so a box on
    the left (no dot under it) can be compared directly to the same box on the
    right (a dot under it), and vice versa.

    Detection is a heuristic (finds CountGD's dot markers by color), so it only
    applies to dot-style CountGD output, not bounding-box style. Returns
    (missed_by_baseline, missed_by_agentic) coordinate arrays for reporting.
    """
    baseline_coords = detect_dot_coords(baseline_image)
    agentic_coords = detect_dot_coords(agentic_image)

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

    fig.suptitle("Missed-cell diff (dot-detection heuristic)", fontsize=14, fontweight="bold")
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


def main():
    parser = argparse.ArgumentParser(description="Compare CountGD-alone vs. agentic results")
    parser.add_argument("--baseline-image", required=True, help="Annotated image from the plain CountGD run")
    parser.add_argument("--baseline-count", required=True, type=int)
    parser.add_argument("--baseline-label", default="CountGD alone", help="Text prompt used for the baseline run")
    parser.add_argument("--agentic-image", required=True, help="Annotated image from the agentic run")
    parser.add_argument("--agentic-count", required=True, type=int)
    parser.add_argument("--agentic-label", default="agentic", help="Text prompt/target used for the agentic run")
    parser.add_argument("--output-dir", default="./comparison_output")
    parser.add_argument("--pdf-name", default=PDF_NAME, help="Filename for the saved PDF report")
    parser.add_argument(
        "--no-missed-cells", action="store_true",
        help="Skip the missed-cell diff page (only works on CountGD's dot-style output)",
    )
    parser.add_argument(
        "--match-radius", type=float, default=12,
        help="Pixel radius within which two detections count as the same cell",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / args.pdf_name

    with PdfPages(pdf_path) as pdf:
        side_by_side_page(
            pdf,
            args.baseline_image, args.baseline_label, args.baseline_count,
            args.agentic_image, args.agentic_label, args.agentic_count,
        )
        comparison_chart_page(
            pdf,
            args.baseline_label, args.baseline_count,
            args.agentic_label, args.agentic_count,
        )
        if not args.no_missed_cells:
            missed_by_baseline, missed_by_agentic = missed_cells_page(
                pdf, args.baseline_image, args.agentic_image, match_radius=args.match_radius,
            )
            print(f"Missed by CountGD alone: {len(missed_by_baseline)}")
            print(f"Missed by agentic: {len(missed_by_agentic)}")

    print(f"Comparison report: {pdf_path}")


if __name__ == "__main__":
    main()
