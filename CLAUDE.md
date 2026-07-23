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
.venv-manager/bin/pip install pandas tqdm ujson einops pandarallel shapely opencv-python-headless numba pyyaml
.venv-manager/bin/pip install pyvips
.venv-manager/bin/pip install --index-url https://rocm.nightlies.amd.com/v2/gfx1151/ torch torchvision
```
The second `pip install` line (`pandas`/`tqdm`/`ujson`/`einops`/`pandarallel`/`shapely`/
`opencv-python-headless`/`numba`/`pyyaml`) is for CellViT support (`agentic_cellvit.py`,
integrated into `manager_agent.py` 2026-07-22) -- the concrete net-new packages
`cell_segmentation.inference.cell_detection`'s actual import chain needs, on top of what's
already installed above for StarDist/Qwen. **Do NOT `pip install -r CellViT/requirements.txt`
here** -- it pins `tensorflow==2.12.0`, `numpy<1.24`, `stardist==0.8.5`, `keras==2.12.0`, which
will very likely fight the newer versions already resolved in this venv for its existing
StarDist support. If a further `ModuleNotFoundError` turns up when importing CellViT beyond
this list, install that one specific package rather than the full requirements.txt; watch
especially for numpy-version friction (deprecated aliases like `np.float`/`np.bool`, removed in
newer numpy than CellViT was pinned against) and fix the specific call site rather than
downgrading numpy repo-wide.

`pyvips` (2026-07-22, for DeepGleason support -- see below) needs the system `libvips` library,
not just the pip package -- already present on this machine (`/usr/bin/vips`, libvips 8.18.0,
confirmed via `pyvips.version(0..2)` after install); on a machine without it, `apt install
libvips-dev` (or equivalent) first. Only used here to downsample DeepGleason's pyramid BigTIFF
overlay (and a whole-slide image itself, for the routing-preview thumbnail) into something
small enough for Qwen to actually load -- `manager_agent.py` never runs the DeepGleason model
itself in this process (see below), so this is the only new native dependency it needs.
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

**CellViT's inference path (`cell_segmentation/inference/cell_detection.py`, via
`models/segmentation/cell_segmentation/cellvit.py`'s `calculate_instance_map`) transitively
imports `numba`** (`cell_segmentation/utils/tools.py`, `from numba import njit, prange`) --
numba bundles its own LLVM via `llvmlite`, the same category of dependency that forces
StarDist's TensorFlow into a subprocess above. This looked like it might hit the identical
"CommandLine Error: Option ... registered more than once!" crash when loaded in the same
process as the manager's own Qwen/ROCm-torch. **Empirically verified (2026-07-22) that it does
NOT crash here**: loaded Qwen (one real `.ask()` call), then imported
`cell_segmentation.inference.cell_detection.CellSegmentationInference`, loaded CellViT-SAM-H,
ran one real forward pass, then ran a second real Qwen `.ask()` call -- all in one process, no
crash. (The `@njit`-decorated functions in `tools.py` are only called by CellViT's own
training/experiment scripts, never by `cell_detection.py`'s inference path -- merely importing
numba, without ever triggering its JIT codegen, doesn't collide with ROCm/Triton's LLVM the way
TensorFlow's XLA does.) VRAM checked too: ~20.4GB allocated / ~22.9GB reserved with both models
resident and generating, ~45GB+ still free (`torch.cuda.mem_get_info()`) -- comfortable
headroom on this machine. Net effect: CellViT runs in-process in `manager_agent.py`
(`CellvitClient`, no `StardistWorker`-style subprocess needed) -- unlike StarDist, which must
stay isolated.

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
  ground truth is supplied for the routed agent. `agentic_countgd.py` is untouched;
  `agentic_stardist.py` only gained small additive per-class PanNuke ground-truth
  functions (`pannuke_class_counts`/`pannuke_class_instance_labels`/
  `load_pannuke_samples_with_classes`/`load_pannuke_sample_with_classes`, for CellViT
  below) -- the manager otherwise only imports existing functions from both.
- CellViT (`agentic_cellvit.py`) is a third routable agent (2026-07-22), alongside
  CountGD/StarDist, with the same ExpertReasoner/dialogue/escalation shape:
  `EXPERT_PERSONA_CELLVIT`/`build_cellvit_dossier`/`decide_cellvit_from_dialogue`/
  `run_cellvit_with_feedback`. Unlike CountGD (counts) and StarDist (segments, no
  typing), CellViT *classifies* nuclei into 5 fixed PanNuke types (Neoplastic,
  Inflammatory, Connective, Dead, Epithelial) -- `select_agent`'s routing prompt
  sends "how many/which are X-type cells" tasks to CellViT and generic "count/
  segment the nuclei" tasks to StarDist. Its internal (logged-only, never
  decision-driving) ground-truth metric is per-class mPQ + F1 -- the official
  PanNuke/CellViT paper protocol, not a naive count comparison -- computed by
  rasterizing CellViT's predicted contours into per-class instance-label arrays and
  reusing `compute_panoptic_quality` once per class via a new
  `StardistWorker.score_cellvit_predictions` method (needs the StarDist subprocess
  even though CellViT itself runs in-process, since `compute_panoptic_quality` lives
  in `agentic_stardist.py`). One real architectural limitation, unlike StarDist's
  prob_thresh/nms_thresh: CellViT's revision levers (`target_classes`/
  `prob_threshold`) can only fix recall/scope issues, never outright
  misclassification -- there's no tunable knob for "the model called an Epithelial
  cell Neoplastic," and `decide_cellvit_from_dialogue`'s prompt says so explicitly
  so the manager doesn't hallucinate a fix that doesn't exist.
- DeepGleason (`agentic_deepgleason.py`, added directly to GitHub by the repo owner rather than
  written in an agent session) is a fourth routable agent (2026-07-22), with the same
  ExpertReasoner/dialogue/escalation shape: `EXPERT_PERSONA_DEEPGLEASON`/
  `build_deepgleason_dossier`/`decide_deepgleason_from_dialogue`/`run_deepgleason_with_feedback`.
  Unlike the other three (all per-cell/per-nucleus tasks on a single tile-sized image),
  DeepGleason grades a whole-slide prostate biopsy's tumor severity (Gleason score / ISUP grade
  group 1-5) -- `select_agent`'s routing prompt sends "grade/stage this tumor" tasks here.
  DeepGleason itself originally had no tunable parameter at all (just `idxmax` over its per-tile
  softmax classification, always argmax regardless of confidence) -- `aggregate_gleason` gained
  a `confidence_threshold` (a tile only counts toward Gleason-pattern tallying if its predicted
  class's softmax probability clears the bar, otherwise it's "Uncertain") as the retry knob,
  analogous to CellViT's `type_prob`/StarDist's `prob_thresh`. Real architectural asymmetry vs.
  the other three: DeepGleason's underlying model call (`agentic_deepgleason.run_deepgleason`,
  a `subprocess.run` into a wholly separate conda environment -- full WSI tiling + TensorFlow
  inference over every tile) is expensive and produces a raw per-tile predictions CSV;
  re-aggregating that CSV with a different `confidence_threshold` (`aggregate_gleason`) is cheap
  and needs no rerun. `DeepGleasonClient.run_slide()` runs the subprocess exactly once per image;
  every retry iteration only calls the cheap `.aggregate()`. No `StardistWorker`-style subprocess
  isolation was needed here either, for a different reason than CellViT: DeepGleason is already
  maximally isolated by construction (a real OS subprocess into a different Python/conda
  environment entirely, not even the same interpreter family as `manager_agent.py`). Because
  `main.py`'s DeepGleason forward pass runs over pyramid BigTIFFs, none of it is ever loaded
  raw by Qwen -- `render_overlay_preview`/the CLI's routing-preview thumbnail both use `pyvips`
  to extract a small, viewable PNG instead (see the `.venv-manager` section above).
  **Scope decision, not yet done:** unlike PanNuke (StarDist/CellViT) or BBBC005 (CountGD),
  there's no automated ground-truth dataset integration for DeepGleason in `train_manager.py`
  -- the standard public source (the PANDA Gleason-grading challenge) is gigabytes-per-slide and
  needs Kaggle credentials, a much bigger lift than PanNuke's partial-zip-read trick. Ground
  truth is only suppliable manually right now (`--ground-truth-gleason-score`/
  `--ground-truth-isup-grade` on `manager_agent.py`'s own CLI) -- a real training pipeline
  against PANDA (or similar) is an explicit follow-up, deliberately scoped out for now rather
  than left implicit. The DeepGleason conda environment/repo itself is also not yet set up on
  this machine (needs `git clone` +
  `git lfs pull` + a Python 3.11 conda env per `agentic_deepgleason.py`'s own docstring) --
  `conda`/`git-lfs` are both already installed here, but the actual env/weights/end-to-end run
  are still to be done.
- No API key is required to run `manager_agent.py` -- Qwen runs locally, CountGD is
  a public hosted Gradio Space. The `anthropic` package is still a required import
  (transitively, via `agentic_countgd.py`/`agentic_stardist.py`) but is never called.
- Train/test splitting (2026-07-22): `agentic_stardist.select_diverse_indices` gained a `split`
  parameter ("all"/"train"/"test", same idea as `bbbc005.load_bbbc005_samples`'s existing one) --
  partitions the candidate PanNuke index pool by parity before the diverse-tissue selection runs,
  so train/test never share an image regardless of `n` on either side. Threaded through
  `StardistWorker.load_pannuke_diverse`/`load_pannuke_diverse_with_classes` and
  `train_manager.py`'s `--pannuke-split` CLI flag, covering StarDist and CellViT the same way
  `--bbbc005-split` already covered CountGD. **Important limitation this doesn't solve on its
  own**: `train_manager.py` updates `expert_notes` (and, in principle, `running_prompt`) from
  every image it processes -- running it a second time against `--pannuke-split test` would
  still train on that "test" data, not evaluate against it cleanly. `evaluate_manager.py` is the
  actual held-out-evaluation counterpart: it loads a trained checkpoint's `running_prompt`/
  `expert_notes` as fixed, read-only inputs (never reassigned or written back), reuses
  `train_manager.py`'s own task-building/trial-running/scoring functions against test-split
  data, and reports accuracy to a separate `eval_result.json`. It also passes a new
  `escalate=False` parameter (threaded through all four `run_*_with_feedback` functions and
  `ManagerAgent.run()`) so a held-out result that never gets accepted can't enter the
  `escalation_queue` either -- without that, a human later resolving that escalation via
  `resolve_escalations.py` would leak test-set corrections back into `expert_notes` through that
  back door, defeating the split. DeepGleason has no dataset integration yet (see above) so
  isn't part of either script.
