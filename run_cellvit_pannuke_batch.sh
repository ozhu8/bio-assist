#!/usr/bin/env bash
# Run 5 PanNuke images (spread across tissue types) through agentic_cellvit.py and
# merge the per-image PDF reports into one cellvit_results1.pdf.
#
# Intended to run on a machine with a working CUDA (or ROCm, which exposes the same
# torch.cuda API) GPU -- CellViT's own CellSegmentationInference hardcodes
# `self.device = f"cuda:{gpu}"` with no CPU fallback, so this will not run on a
# CPU-only or Intel-iGPU-only machine.
#
# Written for the AMD Ryzen AI MAX+ 395 (gfx1151, ROCm) NucBox referenced in this
# repo's CLAUDE.md -- the torch install step below mirrors the working recipe already
# verified there for manager_agent.py's .venv-manager. If you're running this on a
# different machine with a plain NVIDIA GPU instead, replace the "install torch last
# from the ROCm index" step with a normal `pip install torch torchvision`.
#
# Usage (from the bio-assist repo root):
#   ./run_cellvit_pannuke_batch.sh
#
# Assumes ANTHROPIC_API_KEY is set (or `ant auth login` has been run) -- agentic_cellvit.py
# needs it for the Claude orchestration/evaluation calls.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

CELLVIT_REPO="$REPO_ROOT/CellViT"
VENV_DIR="$REPO_ROOT/.venv-cellvit"
CHECKPOINT_DIR="$REPO_ROOT/models"
CHECKPOINT_PATH="$CHECKPOINT_DIR/CellViT-SAM-H-x40.pth"
# Direct-download Google Drive ID for CellViT-SAM-H (x40 magnification, the default this
# script and agentic_cellvit.py's --magnification=40 default expect) -- from the CellViT
# README's "Model checkpoints can be downloaded here" section.
CHECKPOINT_DRIVE_ID="1MvRKNzDW2eHbQb5rAgTEp6s2zAXHixRV"

SAMPLES_DIR="$REPO_ROOT/pannuke_cellvit_samples"
BATCH_OUTPUT_DIR="$REPO_ROOT/cellvit_pannuke_batch_output"
FINAL_PDF="$REPO_ROOT/cellvit_results1.pdf"
PROMPT="${CELLVIT_PROMPT:-Count and classify every nucleus in this tissue sample, broken down by cell type.}"

echo "=== 1. CellViT repo ==="
if [ ! -d "$CELLVIT_REPO" ]; then
  git clone https://github.com/TIO-IKIM/CellViT "$CELLVIT_REPO"
else
  echo "Already present at $CELLVIT_REPO"
fi

echo "=== 2. Python venv (.venv-cellvit) ==="
if [ ! -d "$VENV_DIR" ]; then
  python3 -m venv "$VENV_DIR"
fi
PY="$VENV_DIR/bin/python"
PIP="$VENV_DIR/bin/pip"

echo "=== 3. Install dependencies ==="
"$PIP" install --upgrade pip
"$PIP" install anthropic matplotlib pillow numpy fsspec pypdf gdown
# CellViT's own requirements.txt (needed by cell_segmentation/inference/cell_detection.py's
# imports and the model-building code it pulls in). If this fails on an old pinned version
# under your default python3, retry the venv creation with an older interpreter (the repo's
# README badges Python 3.9.7) -- e.g. `python3.10 -m venv "$VENV_DIR"`.
"$PIP" install -r "$CELLVIT_REPO/requirements.txt" || \
  echo "!! requirements.txt install had failures -- see CLAUDE.md note about old pins; you may need an older python3.x for the venv."
# Install torch LAST and separately, from AMD's gfx1151-specific ROCm nightly index --
# matches the working recipe already verified for .venv-manager in this repo's CLAUDE.md.
"$PIP" install --index-url https://rocm.nightlies.amd.com/v2/gfx1151/ torch torchvision

echo "=== 4. Verify GPU is visible to torch ==="
"$PY" - <<'EOF'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_properties(0).gcnArchName if hasattr(torch.cuda.get_device_properties(0), "gcnArchName") else torch.cuda.get_device_name(0))
EOF

echo "=== 5. Download CellViT-SAM-H-x40 checkpoint ==="
mkdir -p "$CHECKPOINT_DIR"
if [ ! -f "$CHECKPOINT_PATH" ]; then
  "$VENV_DIR/bin/gdown" --id "$CHECKPOINT_DRIVE_ID" -O "$CHECKPOINT_PATH"
  # Google Drive sometimes serves an HTML "can't scan for viruses" interstitial instead of
  # the file for large downloads. If gdown fails or CHECKPOINT_PATH looks tiny (a few KB),
  # download it manually in a browser from:
  #   https://drive.google.com/uc?export=download&id=1MvRKNzDW2eHbQb5rAgTEp6s2zAXHixRV
  # and scp it to CHECKPOINT_PATH instead.
else
  echo "Already present at $CHECKPOINT_PATH"
fi

echo "=== 6. Fetch 5 PanNuke images spread across tissue types ==="
"$PY" fetch_pannuke_cellvit_samples.py --n 5 --fold 1 --output-dir "$SAMPLES_DIR"

echo "=== 7. Run each image through agentic_cellvit.py ==="
mkdir -p "$BATCH_OUTPUT_DIR"
PDF_PATHS=()
"$PY" - "$SAMPLES_DIR/manifest.json" <<'EOF' > "$BATCH_OUTPUT_DIR/manifest_paths.txt"
import json, sys
manifest = json.load(open(sys.argv[1]))
for entry in manifest:
    print(entry["image_path"])
EOF

while IFS= read -r IMAGE_PATH; do
  STEM="$(basename "$IMAGE_PATH" .png)"
  IMAGE_OUTPUT_DIR="$BATCH_OUTPUT_DIR/$STEM"
  echo "--- $STEM ---"
  "$PY" agentic_cellvit.py \
    --image "$IMAGE_PATH" \
    --prompt "$PROMPT" \
    --checkpoint "$CHECKPOINT_PATH" \
    --cellvit-repo "$CELLVIT_REPO" \
    --output-dir "$IMAGE_OUTPUT_DIR"
  PDF_PATHS+=("$IMAGE_OUTPUT_DIR/cellvit_results.pdf")
done < "$BATCH_OUTPUT_DIR/manifest_paths.txt"

echo "=== 8. Merge per-image PDFs into cellvit_results1.pdf ==="
"$PY" - "$FINAL_PDF" "${PDF_PATHS[@]}" <<'EOF'
import sys
from pypdf import PdfWriter

out_path, *pdf_paths = sys.argv[1:]
writer = PdfWriter()
for p in pdf_paths:
    writer.append(p)
with open(out_path, "wb") as f:
    writer.write(f)
print(f"Merged {len(pdf_paths)} PDFs into {out_path}")
EOF

echo "=== Done ==="
echo "Combined PDF: $FINAL_PDF"
