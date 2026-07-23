"""
Fetches individual images -- and their ground-truth cell counts -- from BBBC005 (Broad
Bioimage Benchmark Collection, synthetic fluorescent cell images: 19,200 images at
https://data.broadinstitute.org/bbbc/BBBC005/) without downloading the full 1.8GB zip.
Mirrors agentic_stardist.py's PanNuke fetcher (_open_pannuke_zip): fsspec opens the
remote zip as a seekable HTTP file, so zipfile only needs to read the central directory
plus whichever specific member(s) are actually requested.

Filename convention: SIMCEPImages_{well}_C{cells}_F{blur}_s{sample}_w{stain}.TIF
  - cells: ground-truth cell count (1-100) -- this *is* the label. No separate manifest
    file exists or is needed; BBBC005's own ground-truth zip only has foreground/background
    segmentation masks for F1 images, not per-image counts, so it isn't used here.
  - blur: focus level (1-48); F1 is fully in focus.
  - stain: 1 = cell body stain, 2 = nuclei stain -- CountGD is pointed at the whole cell
    body, so this defaults to w1.

agentic_countgd.py is untouched -- this is a standalone fetcher manager_agent.py's
training loop imports, the same way it imports agentic_stardist.py's PanNuke fetcher.
"""
import re
import zipfile
from pathlib import Path
from typing import Any, cast

import fsspec # pyright: ignore[reportMissingImports]
import numpy as np # pyright: ignore[reportMissingImports]
from PIL import Image # pyright: ignore[reportMissingImports]

BBBC005_IMAGES_URL = "https://data.broadinstitute.org/bbbc/BBBC005/BBBC005_v1_images.zip"
_FILENAME_RE = re.compile(
    r"SIMCEPImages_(?P<well>[A-P]\d{2})_C(?P<cells>\d+)_F(?P<blur>\d+)_s(?P<sample>\d+)_w(?P<stain>\d+)\.TIF"
)


def _open_bbbc005_zip() -> zipfile.ZipFile:
    open_file = cast(Any, fsspec.open(BBBC005_IMAGES_URL))
    return zipfile.ZipFile(open_file.open())


def list_bbbc005_members(zf: zipfile.ZipFile, blur: int = 1, stain: int = 1) -> list:
    """Filter the zip's ~19,200 members down to a given focus level / stain channel --
    reading the central directory only costs one small request, well before any member's
    image bytes are fetched."""
    members = []
    for name in zf.namelist():
        m = _FILENAME_RE.search(Path(name).name)
        if not m:
            continue
        if int(m["blur"]) != blur or int(m["stain"]) != stain:
            continue
        members.append((name, int(m["cells"])))
    return members


def load_bbbc005_samples(n: int, blur: int = 1, stain: int = 1, split: str = "all") -> list:
    """Returns n (image: np.ndarray, ground_truth_count: int) pairs, evenly spread across
    the available cell-count range (1-100) rather than clustering near whichever well the
    zip happens to list first. Only the n chosen members' bytes are actually downloaded.

    split: "all" (default, original behavior), "train", or "test". BBBC005 has no official
    fold structure like PanNuke's, so this partitions the count-sorted member list by index
    parity (even indices -> train, odd -> test) before spacing -- unlike just picking a
    different n on the same full list (which doesn't guarantee disjointness: linspace always
    includes both endpoints regardless of n, and different n values can otherwise land on the
    same indices too), this guarantees train/test never share an image regardless of which n
    each side requests, while both halves still span the full 1-100 count range evenly (since
    consecutive, similarly-scored members alternate between the two halves rather than being
    split low-half/high-half, which would have made test systematically differ in difficulty).

    n <= 0 returns no samples without opening the remote zip at all -- same guard, and same
    reasoning, as agentic_stardist.select_diverse_indices's callers in manager_agent.py
    (StardistWorker.load_pannuke_diverse et al.): np.linspace(..., n) raises ValueError for a
    negative n, which otherwise surfaced as a raw crash from a caller like train_manager.py's
    build_countgd_tasks passing through a negative --countgd-n with no earlier, friendlier
    error."""
    if n <= 0:
        return []
    zf = _open_bbbc005_zip()
    try:
        members = sorted(list_bbbc005_members(zf, blur=blur, stain=stain), key=lambda pair: pair[1])
        if not members:
            raise ValueError(f"no BBBC005 members matched blur=F{blur} stain=w{stain}")
        if split == "train":
            members = members[0::2]
        elif split == "test":
            members = members[1::2]
        elif split != "all":
            raise ValueError(f"split must be 'all', 'train', or 'test', got {split!r}")
        if n >= len(members):
            chosen = members
        else:
            idxs = np.linspace(0, len(members) - 1, n).round().astype(int)
            chosen = [members[i] for i in idxs]

        samples = []
        for name, count in chosen:
            with zf.open(name) as fp:
                image = np.array(Image.open(fp))
            samples.append((image, count))
        return samples
    finally:
        zf.close()
