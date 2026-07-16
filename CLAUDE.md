# bio-assist

## Local environment / venvs

This repo uses a separate venv per agent (package sets conflict / are large enough
that one shared env isn't practical). None of the venvs are tracked in git
(`.gitignore` excludes `.venv-*/`) -- they must be recreated with `python -m venv`
+ `pip install` before running anything.

**As of 2026-07-16, all three local venvs below were deleted** to free space (this
checkout lives under OneDrive, which was down to ~0.2GB of quota). Nothing about
the deletion is unusual or an error -- if you're starting a session here (or on a
new machine, e.g. the SSH server) and hit `ModuleNotFoundError`, this is why:
recreate the venv first using the exact commands below (verified working this
session, not guessed from imports).

### `.venv-countgd` -- for `agentic_countgd.py`
```
python -m venv .venv-countgd
.venv-countgd/Scripts/pip install anthropic matplotlib pillow gradio_client
```

### `.venv-stardist` -- for `agentic_stardist.py`
```
python -m venv .venv-stardist
.venv-stardist/Scripts/pip install anthropic fsspec aiohttp matplotlib csbdeep scikit-image stardist tensorflow
```
`stardist`/`csbdeep` do NOT pull in `tensorflow` automatically -- it's checked for
at import time, not declared as a pip dependency, so it must be installed
explicitly or you'll get `RuntimeError: Please install TensorFlow`.

### `.venv-manager` -- for `manager_agent.py` (Qwen3-VL manager)
```
python -m venv .venv-manager
.venv-manager/Scripts/pip install torch transformers accelerate qwen-vl-utils pillow gradio_client anthropic stardist csbdeep tensorflow aiohttp
```
Note the CPU-only torch here (this machine has no GPU): originally installed via
`pip install torch --index-url https://download.pytorch.org/whl/cpu`. **On the SSH
GPU server, install torch normally instead** (no `--index-url`, or the CUDA-specific
index for that server's driver) -- otherwise Qwen3-VL silently runs on CPU there too.
`manager_agent.py` also needs `stardist`/`csbdeep`/`tensorflow`/`aiohttp` (same
gotcha as above) because it imports from `agentic_stardist.py` at module load time.

Verified end-to-end with `python -c "import manager_agent"` succeeding in
`.venv-manager` before these venvs were deleted.

## Design notes (see also git history / commit messages)

- `manager_agent.py` uses Qwen3-VL-8B-Instruct (local, via `transformers`) instead
  of Claude to route tasks to CountGD/StarDist and drive the retry/feedback loop.
  Scored by MAE (CountGD) or Panoptic Quality (StarDist) when ground truth is
  available (BBBC005 manifest / PanNuke masks), falling back to Qwen's own 0-10
  visual scoring otherwise. `agentic_countgd.py`/`agentic_stardist.py` themselves
  are untouched -- the manager only imports their existing functions.
- No API key is required to run `manager_agent.py` -- Qwen runs locally, CountGD is
  a public hosted Gradio Space. The `anthropic` package is still a required import
  (transitively, via `agentic_countgd.py`/`agentic_stardist.py`) but is never called.
