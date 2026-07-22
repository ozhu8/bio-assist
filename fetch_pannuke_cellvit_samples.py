"""
Fetch a handful of PanNuke images spread across different tissue types, for use as
--image input to agentic_cellvit.py.

Deliberately standalone and light on dependencies (fsspec + numpy + pillow only) --
agentic_stardist.py already has equivalent PanNuke-fetch code, but importing it here
would drag stardist/tensorflow into the CellViT venv along with it (see CLAUDE.md's
"separate venv per agent" note). The fetch logic below (partial-DEFLATE-decompression
reads via HTTP range requests, retry-by-reopening, diverse-tissue index selection) is
duplicated from agentic_stardist.py rather than reimplemented from scratch.

Usage:
    python fetch_pannuke_cellvit_samples.py --n 5 --fold 1 --output-dir pannuke_cellvit_samples
"""
import argparse
import contextlib
import json
import random
import zipfile
from pathlib import Path

import fsspec  # pyright: ignore[reportMissingImports]
import numpy as np  # pyright: ignore[reportMissingImports]
import numpy.lib.format as npy_format  # pyright: ignore[reportMissingImports]
from PIL import Image  # pyright: ignore[reportMissingImports]

PANNUKE_FOLD_URL = "https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke/fold_{fold}.zip"
PANNUKE_HTTP_BLOCK_SIZE = 512 * 1024
# See agentic_stardist.py's TISSUE_DIVERSITY_MAX_INDEX comment: images.npy/masks.npy are
# DEFLATE streams that can only be read sequentially from the start, so this caps how far
# into the fold we're willing to read while still covering most tissue types.
TISSUE_DIVERSITY_MAX_INDEX = 1500


@contextlib.contextmanager
def _open_pannuke_zip(fold: int, block_size: int | None = None):
    """Passes the fsspec HTTP file handle straight to zipfile.ZipFile instead of calling
    fp.read() to materialize it into an in-memory buffer first -- zipfile only seeks/reads the
    central directory plus whatever members are actually opened, so this never downloads the
    full ~700MB+ zip. An earlier version here called fp.read() unconditionally, which pulled
    the entire remote zip into memory with no retry around it -- that unbounded whole-file read
    is what produces FSTimeoutError on PanNuke's server (see agentic_stardist.py's identical
    fetcher, which hit and fixed this same issue first)."""
    with fsspec.open(PANNUKE_FOLD_URL.format(fold=fold), mode="rb", block_size=block_size) as fp:  # type: ignore[assignment]
        zf = zipfile.ZipFile(fp)  # type: ignore[arg-type]
        try:
            yield zf
        finally:
            zf.close()


def _read_exact(fp, total_bytes: int, chunk_size: int = 4 * 1024 * 1024) -> bytes:
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


def _read_array_prefix(zf: zipfile.ZipFile, member: str, n: int, retries: int = 4) -> tuple:
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            with zf.open(member) as fp:
                shape, _, dtype = _read_npy_header(fp)
                per_item_bytes = int(np.prod(shape[1:])) * dtype.itemsize if len(shape) > 1 else dtype.itemsize
                raw = _read_exact(fp, n * per_item_bytes)
                return raw, shape, dtype
        except Exception as exc:
            last_exc = exc
            print(f"  [retry {attempt}/{retries}] read of {member} dropped ({exc}); reopening and retrying...")
    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"Failed to read {member}: retries={retries}")


def load_pannuke_types(fold: int) -> list:
    with _open_pannuke_zip(fold) as zf:
        with zf.open(f"Fold {fold}/images/fold{fold}/types.npy") as tf:
            _, _, dtype = _read_npy_header(tf)
            types = np.frombuffer(tf.read(), dtype=dtype)
    return [str(t) for t in types]


def load_pannuke_images(fold: int, n: int):
    with _open_pannuke_zip(fold, block_size=PANNUKE_HTTP_BLOCK_SIZE) as zf:
        raw, shape, dtype = _read_array_prefix(zf, f"Fold {fold}/images/fold{fold}/images.npy", n)
        images = np.frombuffer(raw, dtype=dtype).reshape((n,) + shape[1:])
    return images.astype(np.uint8)


def select_diverse_indices(types: list, n: int, max_index: int | None = None, seed: int = 0) -> list:
    """Round-robins through shuffled tissue types, one index per tissue per pass, so the
    result isn't dominated by whichever tissue has the biggest contiguous block."""
    by_tissue: dict = {}
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


def main():
    parser = argparse.ArgumentParser(description="Fetch n PanNuke images spread across tissue types")
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--fold", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="pannuke_cellvit_samples")
    args = parser.parse_args()
    if args.n < 1:
        parser.error("--n must be at least 1")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fetching tissue-type layout for PanNuke fold {args.fold}...")
    all_types = load_pannuke_types(args.fold)
    selected = select_diverse_indices(all_types, args.n, max_index=TISSUE_DIVERSITY_MAX_INDEX, seed=args.seed)
    tissues = [all_types[i] for i in selected]
    n_tissues = len(set(tissues))
    print(
        f"Selected {len(selected)} images spanning {n_tissues} tissue types "
        f"(fold indices {selected[0]}-{selected[-1]}, {100 * (selected[-1] + 1) / len(all_types):.0f}% of the fold): "
        f"{list(zip(selected, tissues))}"
    )

    print(f"Downloading first {selected[-1] + 1} images from fold {args.fold} (only as much of the DEFLATE stream as needed)...")
    images_prefix = load_pannuke_images(args.fold, selected[-1] + 1)

    manifest = []
    for idx, tissue in zip(selected, tissues):
        stem = f"pannuke_fold{args.fold}_{idx:04d}_{tissue}"
        image_path = output_dir / f"{stem}.png"
        Image.fromarray(images_prefix[idx]).save(image_path)
        manifest.append({"index": idx, "tissue": tissue, "image_path": str(image_path)})
        print(f"  saved {image_path}")

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest written: {manifest_path}")


if __name__ == "__main__":
    main()
