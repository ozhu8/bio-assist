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

### `.venv-btrack` -- for `agentic_btrack.py`
```
python -m venv .venv-btrack
.venv-btrack/Scripts/pip install anthropic matplotlib pillow numpy scikit-image tifffile imagecodecs btrack stardist csbdeep tensorflow fsspec aiohttp
```
Needs `stardist`/`csbdeep`/`tensorflow` for the same reason `manager_agent.py` does
(same gotcha as above): it imports `run_stardist` from `agentic_stardist.py` at
module load time, to segment each frame before btrack links them across time --
and since that pulls in all of `agentic_stardist.py`'s own imports too, it also
needs `fsspec`/`aiohttp` (the PanNuke partial-read deps `agentic_stardist.py`
uses) even though `agentic_btrack.py` itself never touches PanNuke. Needs
`imagecodecs` too -- Cell Tracking Challenge's raw/GT TIFFs are LZW-compressed,
and `tifffile.imread` raises `ValueError: <COMPRESSION.LZW: 5> requires the
'imagecodecs' package` without it.

**On Apple Silicon, build this venv with a native arm64 Python, not an
Intel/x86_64 one.** `python -m venv` on macOS uses whatever `python`/`python3`
resolves to on `PATH` -- if that happens to be an x86_64-only build (e.g. an old
`/Library/Frameworks/Python.framework` install), every run goes through Rosetta
2, and Rosetta's one-time ahead-of-time translation of TensorFlow's ~700MB
`libtensorflow_cc.2.dylib` took 13+ minutes and then hung in an uninterruptible
kernel wait (`ps` showed `UE` state, not even killable with `kill -9`) before
this was caught. Check with `file $(which python3)` before creating the venv
(want `arm64`, not `x86_64`); `/opt/anaconda3/bin/python3` was the arm64 option
on this machine. Rebuilding the venv with that interpreter dropped the same
import from 13+ minutes (hung) to ~4s warm / ~57s cold.

**`import btrack` alone does not expose `btrack.datasets`** -- `btrack/__init__.py`
(as of btrack 0.7.0) only imports `BayesianTracker`, not the `datasets`
submodule, even though `btrack.datasets.cell_config()` (used to fetch the
default tracker config) is real, documented API. Fixed in `agentic_btrack.py`
by adding an explicit `import btrack.datasets` alongside `import btrack`.

Verified end-to-end 2026-07-17: `--images-dir` against a synthetic 6-frame/
5-blob sequence (generated locally, not part of the repo) correctly produced 5
tracks whose plotted trajectories matched each blob's actual direction of
motion, plus a PDF report and `.h5` tracks file.

**`PyTrackObject.label` is NOT the source segmentation's per-frame instance/region
ID, despite the name** -- it's a classification *state* field that defaults to
`constants.States.NULL` for every object unless `segmentation_to_objects` is
called with `assign_class_ID=True`. Every object came back with the identical
`label` value regardless of which region it was built from -- confirmed by direct
inspection (`obj.label` was `5` for all 130 objects across 3 frames of visibly
different regions). This silently broke the (frame, label) -> btrack track ID
mapping `agentic_btrack.py`'s ground-truth scoring depends on (`pred_track_id_map`
had 2 entries instead of 86). Fixed in `track_sequence` by reconstructing each
object's original label from its 1-indexed position among objects sharing its
frame instead -- valid because `segmentation_to_objects` processes frames
strictly in order (single worker by default) and regionprops visits labels in
ascending order with no gaps (StarDist's own convention).

**Even with that fixed, `--ctc-dataset` tracking accuracy against ground truth is
currently bad (link_accuracy ~0), and it's NOT a script bug -- `max_search_radius`
does not act as a hard cutoff on link distance.** Tested directly against 3 frames
of `Fluo-N2DL-HeLa` (true same-cell displacement ~1-2px there): `max_search_radius=5`
and `max_search_radius=100` produced near-identical results -- same 43 links, same
973.8px maximum link distance, same ~370px mean. Whatever's actually gating which
objects get linked together is coming from elsewhere in `cell_config.json` (most
likely the motion model's own process/measurement noise parameters), not the
`max_search_radius` knob `agentic_btrack.py`'s retry loop (`propose_search_radius`)
is built around -- so as currently written, that retry loop will not converge to
good tracking on this dataset no matter how many iterations it runs, since it's
tuning a parameter that isn't the actual lever. **Paused here, not fixed**: the
real fix would mean investigating/retuning `cell_config.json`'s motion-model noise
parameters directly (its example config is presumably tuned for a reference
dataset with a very different displacement scale than this one), which is a
larger, more open-ended task than the mapping bug above. `imagecodecs` had to be
added too (see the install command above) to read Cell Tracking Challenge's
LZW-compressed TIFFs.

**On this Windows/OneDrive checkout specifically**, built at
`C:\Users\hanna\venvs\bio-assist\.venv-btrack` instead of `.venv-btrack` in the
repo root like the others -- deliberately outside the OneDrive-synced folder, to
avoid repeating the quota problem noted above (`tensorflow` alone is well over
500MB). Run scripts by pointing at that interpreter directly from the repo root,
e.g. `"C:\Users\hanna\venvs\bio-assist\.venv-btrack\Scripts\python" agentic_btrack.py ...`
Worth doing the same for the other three venvs next time they need recreating.
Also hit, and worked around without any system changes: `StarDist2D.from_pretrained`
tries to `symlink_to` its extracted model cache (`csbdeep`'s `get_model_folder`,
keras >= 3.6.0 codepath) and Windows requires either admin rights or Developer
Mode for unprivileged symlinks -- `OSError: [WinError 1314] A required privilege
is not held by the client`. Since the target content is already sitting right
there under the `*_extracted` suffix, a plain recursive copy to the non-suffixed
name (`cp -r ..._extracted ...` with the suffix stripped, under
`~/.keras/models/StarDist2D/<model>/`) satisfies the `path_folder.exists()` check
csbdeep does before attempting the symlink, so it's never attempted at all on
subsequent runs.

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

**This is a shared machine -- `gradio_client` calls (CountGD) fail with
`PermissionError: /tmp/gradio/...` unless `GRADIO_TEMP_DIR` is set.** By default
`gradio_client` downloads files to `$TMPDIR/gradio` (`/tmp/gradio`), and another
user on this box (`olivia`) already owns that directory with mode 775 -- `hannah`
can list it but can't `mkdir` inside it, which is what every CountGD call needs to
do to save its result image. Fix: export `GRADIO_TEMP_DIR=/home/hannah/.gradio_tmp`
(or any directory `hannah` owns) before running anything that calls CountGD
(`agentic_countgd.py`, `manager_agent.py`, `train_manager.py`). Verified
(2026-07-16): `train_manager.py --countgd-n 1 --stardist-n 1` failed with the
PermissionError above without this set, and completed cleanly (~3 min end to end,
including a real CountGD call, a real StarDist/PanNuke trial, and a real Qwen
batch-summarization call) with it set.

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
  Ground truth (BBBC005 counts / PanNuke masks) is never given to the manager
  itself -- it goes to a separate `ExpertReasoner` persona (the same loaded Qwen
  weights, reused under a different system prompt as a domain expert who privately
  holds the true answer plus extra "expert" context: BBBC005 focus/stain metadata
  parsed from the filename, or PanNuke tissue type + a rendered outline of the real
  ground-truth nucleus boundaries). Each iteration the manager gets up to 5
  question/answer turns with the expert (`run_expert_dialogue`) -- the expert
  explains its morphological/domain reasoning but is instructed to never state the
  ground-truth number or say accept/reject/correct/wrong -- and the manager decides
  accept/reject itself from that transcript (`decide_countgd_from_dialogue`/
  `decide_stardist_from_dialogue`). The old MAE/PQ threshold rule is still computed
  every iteration but purely for history/logging (`internal_mae`/`internal_pq`/
  `internal_would_accept`), to compare the new dialogue-driven judgments against the
  old hard-threshold ones after the fact -- it's never in the manager's own prompts.
  Falls back to Qwen's own 0-10 visual scoring (no expert, no dialogue) when no
  ground truth is supplied for the routed agent. `agentic_countgd.py`/
  `agentic_stardist.py` themselves are untouched -- the manager only imports their
  existing functions.
- No API key is required to run `manager_agent.py` -- Qwen runs locally, CountGD is
  a public hosted Gradio Space. The `anthropic` package is still a required import
  (transitively, via `agentic_countgd.py`/`agentic_stardist.py`) but is never called.
- `agentic_btrack.py` follows the same CountGD/StarDist shape but tracks cells across
  a frame sequence instead of scoring one image: it reuses `run_stardist` from
  `agentic_stardist.py` unchanged to segment each frame, then runs btrack to link
  those per-frame instances into tracks. Ground truth comes from Cell Tracking
  Challenge (celltrackingchallenge.net) training sequences instead of PanNuke, scored
  with a simplified link-accuracy proxy (not the official CTC TRA/AOGM metric -- see
  the script's docstring and `compute_link_accuracy`). The retry knob is btrack's
  `max_search_radius`, the only easily-revisable parameter, rather than StarDist's
  prob_thresh/nms_thresh pair. Written but not yet run end-to-end -- see the
  `.venv-btrack` note above.
