"""
BBBC005 loader for CountGD ground-truth training samples (train_manager.py).

BBBC005 (Broad Bioimage Benchmark Collection, image set 005) is synthetic
fluorescence microscopy data with the true cell count baked directly into
each filename: SIMCEPImages_<well>_C<count>_F<blur>_s<sample>_w<stain>.TIF.
Only F1 (fully in-focus) images are used here -- the out-of-focus ones exist
specifically to test focus-robustness, not to serve as clean ground truth --
and w1 (cell body stain) is the default since this trains CountGD/the
manager on "count the individual cells", not nuclei (that's w2, and StarDist/
PanNuke's job anyway).

Downloads and caches the images zip locally the first time (~1.8GB); reruns
reuse the cache. Point BBBC005_CACHE_DIR (see below) at wherever you want
that download to live -- e.g. the Ubuntu box's disk, not a laptop's.

Source: https://bbbc.broadinstitute.org/BBBC005
"""
import os
import random
import re
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
from PIL import Image

IMAGES_URL = "https://data.broadinstitute.org/bbbc/BBBC005/BBBC005_v1_images.zip"
CACHE_DIR = Path(os.environ.get("BBBC005_CACHE_DIR", Path(__file__).resolve().parent / "bbbc005_data"))
FILENAME_RE = re.compile(r"SIMCEPImages_\w\d+_C(?P<count>\d+)_F(?P<blur>\d+)_s\d+_w(?P<stain>\d)\.TIF")


def _ensure_downloaded() -> Path:
    """Downloads+extracts BBBC005's images zip into CACHE_DIR on first use."""
    extracted_marker = CACHE_DIR / ".extracted"
    if extracted_marker.exists():
        return CACHE_DIR
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = CACHE_DIR / "BBBC005_v1_images.zip"
    if not zip_path.exists():
        print(f"Downloading BBBC005 images (~1.8GB) to {zip_path} ...")
        urlretrieve(IMAGES_URL, zip_path)
    print(f"Extracting {zip_path} ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(CACHE_DIR)
    extracted_marker.touch()
    return CACHE_DIR


def load_bbbc005_samples(n: int, blur: int = 1, stain: int = 1, seed: int = 0, split: str = "all"):
    """Return up to n (image, ground_truth_count) pairs from BBBC005's
    in-focus (F<blur>) subset. image is an RGB numpy array; ground_truth_count
    comes straight from the filename's C<count> field.

    split ("all"/"train"/"test") partitions the matching candidates by
    count-sorted index parity -- sort by ground-truth count first (so the
    partition is spread evenly across the whole count range, not clustered by
    whatever arbitrary order the filesystem happened to return), then take
    every other one. train_manager.py/evaluate_manager.py use this to
    guarantee train and test never share an image regardless of n on either
    side -- see their own --bbbc005-split docstrings."""
    root = _ensure_downloaded()
    candidates = []
    for path in root.rglob("*.TIF"):
        match = FILENAME_RE.match(path.name)
        if match and int(match["blur"]) == blur and int(match["stain"]) == stain:
            candidates.append((path, int(match["count"])))

    if not candidates:
        raise RuntimeError(
            f"No BBBC005 images matched F{blur}/w{stain} under {root} -- "
            "check the zip extracted correctly."
        )

    if split != "all":
        candidates.sort(key=lambda c: (c[1], c[0].name))
        if split == "train":
            candidates = candidates[0::2]
        elif split == "test":
            candidates = candidates[1::2]
        else:
            raise ValueError(f"split must be 'all', 'train', or 'test', got {split!r}")

    random.Random(seed).shuffle(candidates)
    return [
        (np.array(Image.open(path).convert("RGB")), count)
        for path, count in candidates[:n]
    ]
