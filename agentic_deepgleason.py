"""
Tumor/Gleason-grading pipeline: runs DeepGleason (frankkramer-lab/DeepGleason)
tile classification on a whole-slide prostate pathology image, then
aggregates the per-tile predictions into a standard clinical Gleason score
and ISUP grade group. DeepGleason itself only classifies 1024x1024 tiles
into 6 classes (confirmed against its own code/main.py) -- it has no
aggregation step of its own, so the primary/secondary pattern -> Gleason
score -> ISUP grade group logic here implements the standard 2014 ISUP
consensus grading rules on top of its raw output.

Setup (not run by this file -- do this once on whatever machine runs it):
    git clone https://github.com/frankkramer-lab/DeepGleason.git
    cd DeepGleason && git lfs pull   # plain `git clone` leaves LFS pointer
                                      # files, not the real ~100MB+ weights
    pip install -r requirements.txt  # includes AUCMEDI, TensorFlow
    sudo apt install libvips-dev     # pyvips (a requirements.txt dep) needs
                                      # the system library, not just the pip package
Point DEEPGLEASON_REPO at wherever you cloned it (env var, or edit below).
DeepGleason's own dependencies (TensorFlow/AUCMEDI/etc, pinned to versions
that need Python 3.11, not whatever this Flask app itself runs under) live
in a separate conda environment -- point DEEPGLEASON_PYTHON at that
environment's interpreter, not a bare "python" off the calling process's
own PATH, or this will try to run DeepGleason in the wrong environment.

Usage:
    python agentic_deepgleason.py --slide biopsy.ome.tiff
"""
import argparse
import os
import subprocess
from pathlib import Path

import pandas as pd

DEEPGLEASON_REPO = Path(os.environ.get("DEEPGLEASON_REPO", Path.home() / "DeepGleason"))
DEEPGLEASON_MODEL = DEEPGLEASON_REPO / "models" / "model.ConvNeXtBase.hdf5"
DEEPGLEASON_PYTHON = os.environ.get(
    "DEEPGLEASON_PYTHON", str(Path.home() / ".conda" / "envs" / "deepgleason" / "bin" / "python")
)

# Exact column names from DeepGleason's own code/main.py (COL_NAMES) -- one
# row per tile, softmax probability per class; A_S/A_D are artefact classes
# (sponge / dust-debris), R is non-cancerous regular tissue, G3/G4/G5 are
# Gleason growth patterns 3-5 (the only classes that count toward grading).
TILE_CLASSES = ["A_S", "A_D", "R", "G3", "G4", "G5"]
TUMOR_CLASSES = ["G3", "G4", "G5"]
CLASS_LABELS = {
    "A_S": "Artefact (sponge)", "A_D": "Artefact (dust/debris)", "R": "Regular tissue",
    "G3": "Gleason pattern 3", "G4": "Gleason pattern 4", "G5": "Gleason pattern 5",
}

# Standard 2014 ISUP consensus grade groups -- all 9 (primary, secondary)
# combinations where primary/secondary are each in {3, 4, 5}. Order matters:
# 3+4 (ISUP 2) and 4+3 (ISUP 3) are clinically different, not interchangeable.
ISUP_GRADES = {
    (3, 3): 1,
    (3, 4): 2, (4, 3): 3,
    (4, 4): 4, (3, 5): 4, (5, 3): 4,
    (4, 5): 5, (5, 4): 5, (5, 5): 5,
}


def run_deepgleason(slide_path: str, output_dir: Path, generate_overlay: bool = False) -> tuple:
    """Runs DeepGleason's own CLI as a subprocess -- it's a separate,
    self-contained TensorFlow/AUCMEDI pipeline, not something to reimplement
    -- and returns (predictions_path, overlay_path). This is the expensive
    step (full WSI tiling + model inference over every tile); callers that
    want to try several confidence_threshold values (see aggregate_gleason)
    should call this once and re-aggregate from the same predictions CSV
    rather than re-invoking the subprocess. overlay_path is None unless
    generate_overlay=True -- DeepGleason only writes the classification
    BigTIFF overlay when --generate_overlay is passed; the path is
    predictable from its own naming convention (slide stem + "_gleason.tiff"
    in output_dir, confirmed against code/main.py)."""
    predictions_path = output_dir / "predictions.csv"
    cmd = [
        DEEPGLEASON_PYTHON, str(DEEPGLEASON_REPO / "code" / "main.py"),
        "--input", str(slide_path),
        "--output", str(output_dir),
        "--model", str(DEEPGLEASON_MODEL),
        "--predictions", str(predictions_path),
    ]
    overlay_path = None
    if generate_overlay:
        cmd.append("--generate_overlay")
        slide_stem = Path(slide_path).stem
        overlay_path = output_dir / f"{slide_stem}_gleason.tiff"
    subprocess.run(cmd, check=True)
    return predictions_path, overlay_path


def aggregate_gleason(predictions_path: Path, confidence_threshold: float = 0.0) -> dict:
    """Turns DeepGleason's per-tile soft-label predictions into a standard
    clinical result: the predicted class per tile (argmax over TILE_CLASSES),
    primary/secondary Gleason pattern (the two most common tumor patterns by
    tile count -- standard practice when true tumor area isn't available),
    Gleason score (their sum, e.g. "3+4"), and ISUP grade group. Returns
    tumor_found=False (with everything else None) if no tile was classified
    as any of G3/G4/G5.

    confidence_threshold: a tile only counts toward tile_counts/tumor_counts
    if its predicted class's softmax probability (the max over TILE_CLASSES,
    not just whichever happens to be argmax) meets this bar -- otherwise it's
    excluded as "Uncertain" rather than forced into whichever class edges out
    the others. Default 0.0 always counts every tile (original behavior,
    equivalent to plain argmax) since every probability is >= 0."""
    df = pd.read_csv(predictions_path, index_col=0)
    confidence = df[TILE_CLASSES].max(axis=1)
    predicted_class = df[TILE_CLASSES].idxmax(axis=1)
    predicted_class = predicted_class.where(confidence >= confidence_threshold, "Uncertain")
    tile_counts = predicted_class.value_counts().to_dict()
    tumor_counts = {cls: tile_counts.get(cls, 0) for cls in TUMOR_CLASSES}

    if sum(tumor_counts.values()) == 0:
        return {
            "tumor_found": False, "primary_pattern": None, "secondary_pattern": None,
            "gleason_score": None, "isup_grade": None,
            "tile_counts": tile_counts, "total_tiles": len(df),
        }

    ranked = sorted(tumor_counts.items(), key=lambda kv: kv[1], reverse=True)
    primary = int(ranked[0][0].removeprefix("G"))
    secondary = int(ranked[1][0].removeprefix("G")) if ranked[1][1] > 0 else primary

    return {
        "tumor_found": True, "primary_pattern": primary, "secondary_pattern": secondary,
        "gleason_score": f"{primary}+{secondary}", "isup_grade": ISUP_GRADES[(primary, secondary)],
        "tile_counts": tile_counts, "total_tiles": len(df),
    }


def render_overlay_preview(overlay_tiff_path: Path, preview_path: Path, max_dim: int = 1024) -> None:
    """Extracts a small, viewable PNG from DeepGleason's pyramid BigTIFF overlay -- the raw
    BigTIFF itself isn't something an image-viewing model can be hand a path to directly.
    pyvips (already a DeepGleason/main.py dependency, and confirmed present system-wide via
    `vips --version` on this machine) reads the shrink-on-load thumbnail directly rather than
    decoding the full pyramid level, which plain PIL.Image.open on a tiled BigTIFF does not
    reliably support across libtiff builds."""
    import pyvips  # pyright: ignore[reportMissingImports]
    # pyvips generates methods dynamically via __getattr__ (no stubs), so Pylance can't infer
    # thumbnail()'s return type and treats it as None -- it's a real vips Image at runtime.
    image = pyvips.Image.thumbnail(str(overlay_tiff_path), max_dim)  # type: ignore[attr-defined]
    image.write_to_file(str(preview_path) + "[compression=png]")  # type: ignore[attr-defined]


def run_tumor_detection(slide_path: str, output_dir, confidence_threshold: float = 0.0) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path, _ = run_deepgleason(slide_path, output_dir)
    return aggregate_gleason(predictions_path, confidence_threshold=confidence_threshold)


def main():
    parser = argparse.ArgumentParser(description="Run DeepGleason + Gleason-score aggregation on a whole-slide image")
    parser.add_argument("--slide", required=True, help="Path to a whole-slide prostate pathology image (OME-TIFF)")
    parser.add_argument("--output-dir", default="./deepgleason_output")
    parser.add_argument("--confidence-threshold", type=float, default=0.0,
                         help="Minimum per-tile softmax probability to count a tile toward Gleason-pattern "
                              "tallying; below it a tile is excluded as Uncertain. Default 0.0 = every tile counts.")
    args = parser.parse_args()

    result = run_tumor_detection(args.slide, args.output_dir, confidence_threshold=args.confidence_threshold)
    if result["tumor_found"]:
        print(
            f"Tumor found: Gleason score {result['gleason_score']} "
            f"(primary {result['primary_pattern']}, secondary {result['secondary_pattern']}), "
            f"ISUP grade group {result['isup_grade']}"
        )
    else:
        print("No tumor found.")
    print(f"Tile breakdown: {result['tile_counts']} (of {result['total_tiles']} total)")


if __name__ == "__main__":
    main()
