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
.venv-manager/bin/pip install transformers accelerate qwen-vl-utils pillow gradio_client anthropic stardist csbdeep tensorflow aiohttp
.venv-manager/bin/pip install --index-url https://rocm.nightlies.amd.com/v2/gfx1151/ torch torchvision
```
`torchvision` is needed too -- `qwen_vl_utils` imports it at module load time (`ModuleNotFoundError:
No module named 'torchvision'` otherwise), even though nothing in this repo calls it directly.
(`bin/`, not `Scripts/` -- this machine is Linux. `Scripts/` in older notes here was
from a Windows checkout; use whichever matches `python -m venv`'s own layout on the
machine you're on.)

**This machine's GPU is an AMD Ryzen AI MAX+ 395 / Radeon 8060S iGPU (Strix Halo,
gfx1151), not CUDA.** The plain `pip install torch` (PyPI, CPU-only or CUDA builds)
does NOT include gfx1151 kernels, and `HSA_OVERRIDE_GFX_VERSION` tricks to
masquerade as a supported arch (e.g. gfx1100) do NOT work either -- both fail at
runtime with `HIP error: no kernel image is available for execution on the device`
even though `torch.cuda.is_available()` reports `True` (it detects the device, it
just has no matching compiled kernels). The fix is installing from AMD's own
per-arch nightly wheel index above (project: ROCm/TheRock), which bundles a
self-contained ROCm 7.x runtime built for gfx1151 specifically -- confirmed working
(`torch.cuda.get_device_properties(0).gcnArchName == 'gfx1151'`, real matmul +
full Qwen3-VL-8B `device_map="auto"` load onto the GPU, ~17.5GB VRAM, no code
changes needed in `manager_agent.py`). Install order matters: install torch from
that index *last* / separately, since installing everything from that index at
once also works but pulls rocm/triton versions pinned to whatever nightly is
current -- if `pip install torch` from the gfx1151 index ever fails to resolve,
drop the other packages from that command and let it pick its own pinned
`triton`/`rocm-sdk-*` versions instead of whatever's already in the venv.

**GPU access needs the `render` and `video` groups.** `id hannah` (from
`/etc/passwd` + `/etc/group`) includes both, but a shell whose *login session*
predates being added to those groups won't have them active (`groups` / `id` in
that shell won't show them, and `rocminfo`/torch will fail with `Unable to open
/dev/kfd read-write: Permission denied` or silently fail to see the device). Fix:
open a fresh terminal/session (group membership is resolved at login). If you
can't restart the session (e.g. this harness's shell), `sudo -n -u hannah <cmd>`
re-execs with freshly-resolved groups and works around it without a real
password prompt, since `hannah` is in `sudo`.

Verified end-to-end with `python -c "import manager_agent"` succeeding in
`.venv-manager` before these venvs were deleted, and (2026-07-16) with a real
Qwen3-VL-8B load + generation on the GPU after the ROCm reinstall above.

**TensorFlow (imported transitively by `agentic_stardist.py`, via `stardist`) cannot
share a process with GPU torch here.** TensorFlow bundles its own LLVM (for XLA);
ROCm's kernel compiler / Triton also bundles LLVM. The instant both are loaded in one
process and a GPU kernel actually runs, they collide over LLVM's global command-line-
option registry and the process aborts:
```
clang (LLVM option parsing): CommandLine Error: Option 'print-inst-addrs' registered more than once!
LLVM ERROR: inconsistency in registered CommandLine options
```
This was invisible before (CPU-only torch never loaded Triton's LLVM), and would
likely also hit the SSH GPU server once torch there is CUDA-enabled instead of
CPU-only, for the same reason. Fixed (2026-07-16) in `manager_agent.py` itself --
`agentic_stardist.py` is still untouched -- by moving all StarDist calls into a
persistent `StardistWorker` subprocess (spawned, not forked, so it never inherits
this process's already-loaded ROCm/Triton state) instead of importing
`agentic_stardist` directly. `manager_agent.py` no longer imports `stardist` or
`agentic_stardist` at module level at all -- only the worker-side functions do,
inside the spawned child. Verified: `import manager_agent` + a GPU matmul no longer
crashes, and a full `StardistWorker.init()`/`.run()` round trip (real model load,
inference, outline PNG) works with GPU matmuls succeeding both before and after.

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
