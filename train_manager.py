"""
Trains manager_agent.py's routing/tuning prompt against ground-truth data (BBBC005 for
CountGD, PanNuke for StarDist) instead of running the manager live. For each training
image, runs the existing retry loop (run_countgd_with_feedback / run_stardist_with_feedback,
scored against real ground truth -- MAE / Panoptic Quality) and records a structured note:
what was tried, whether it helped, and what the output looked like. Every --batch-size
images, code computes numeric stats over the batch, Qwen reads the batch's qualitative
notes and folds both into a running prompt (merged with whatever the previous batches
already found -- reinforcing patterns that still hold, dropping ones this batch
contradicts). Once every training image is processed, that running prompt is the final
trained prompt: meant to be handed to manager_agent.py at inference time, when there's no
ground truth to score against, so Qwen recognizes situations it has already seen patterns
for (e.g. "this output looks like the low-PQ cases from training -- probably still wrong
even though I can't compute PQ here").

agentic_countgd.py and bbbc005.py are untouched; agentic_stardist.py only gained small
additive per-class PanNuke ground-truth functions (for CellViT) -- this otherwise only calls
existing functions (via manager_agent.py), the same way manager_agent.py's own CLI does.

Usage:
    python train_manager.py --countgd-n 10 --stardist-n 10 --cellvit-n 10
"""
import argparse
import json
from itertools import zip_longest
from pathlib import Path

from PIL import Image # pyright: ignore[reportMissingImports]

from bbbc005 import load_bbbc005_samples
from manager_agent import (
    ManagerAgent, MODEL_ID, run_cellvit_with_feedback, run_countgd_with_feedback, run_stardist_with_feedback,
)


def build_note(qwen, agent: str, image_id: str, history: list, final_image_path, lower_is_better: bool,
                chosen_iteration: int) -> dict:
    """Turns one image's existing retry-loop history (already produced by
    run_countgd_with_feedback / run_stardist_with_feedback) into a structured training note.
    final_score/accepted must come from the iteration the manager actually chose
    (chosen_iteration), not history[-1] -- choose_best_output can revert to an earlier
    iteration than the last one attempted, and final_image_path already reflects that choice."""
    metric_key = "internal_mae" if agent == "countgd" else ("internal_mpq" if agent == "cellvit" else "pq")
    chosen_entry = next(h for h in history if h["iteration"] == chosen_iteration)
    initial_score = history[0][metric_key]
    final_score = chosen_entry[metric_key]
    accepted = bool(chosen_entry["accept"])

    adjustments = []
    for prev, cur in zip(history, history[1:]):
        if agent == "countgd":
            change = f"revised count_target -> {cur['count_target']!r}"
        elif agent == "cellvit":
            change = f"revised target_classes={cur['target_classes']!r}, prob_threshold={cur['prob_threshold']}"
        else:
            change = f"revised prob_thresh={cur['prob_thresh']}, nms_thresh={cur['nms_thresh']}"
        score_before, score_after = prev[metric_key], cur[metric_key]
        helped = (score_after < score_before) if lower_is_better else (score_after > score_before)
        adjustments.append({
            "iteration": cur["iteration"], "change": change,
            "score_before": score_before, "score_after": score_after, "helped": helped,
        })

    metric_name = {
        "countgd": "MAE (lower is better)",
        "cellvit": "mean per-class Panoptic Quality (higher is better)",
    }.get(agent, "Panoptic Quality (higher is better)")
    characteristics_prompt = (
        f"This {agent} run finished with a score of {final_score} ({metric_name}), "
        f"{'accepted' if accepted else 'not accepted within the iteration budget'}.\n"
        "Describe in 1-2 sentences the visual/output characteristics of this image and result "
        "that plausibly explain the score -- e.g. cell/nucleus density, overlap, contrast, image "
        "noise, how tight the boxes/outlines look. This will be used later to recognize similar "
        "cases when there's no ground truth to check the score against."
    )
    output_characteristics = qwen.ask(str(final_image_path), characteristics_prompt, max_new_tokens=150)

    note = {
        "image_id": image_id, "agent": agent, "lower_is_better": lower_is_better,
        "initial_score": initial_score, "final_score": final_score, "accepted": accepted,
        "adjustments": adjustments, "output_characteristics": output_characteristics,
    }
    # CellViT logs both headline PanNuke metrics (mPQ as the primary final_score above, F1
    # alongside it) -- see the CellViT internal-metric design in manager_agent.py's
    # run_cellvit_with_feedback.
    if agent == "cellvit" and chosen_entry.get("internal_f1") is not None:
        note["final_f1"] = chosen_entry["internal_f1"]
    return note


def run_countgd_trial(manager: ManagerAgent, image_path: str, image_id: str, ground_truth_count: int,
                       max_iterations: int, output_dir: Path, expert_notes: str = "",
                       escalate: bool = True) -> dict:
    result = run_countgd_with_feedback(
        manager.qwen, manager.countgd_client, image_path, "count the individual cells",
        max_iterations, output_dir, ground_truth_count=ground_truth_count, image_id=image_id,
        expert_notes=expert_notes, escalate=escalate,
    )
    return build_note(manager.qwen, "countgd", image_id, result["history"], result["annotated_image"],
                       lower_is_better=True, chosen_iteration=result["chosen_iteration"])


def run_stardist_trial(manager: ManagerAgent, image_path: str, image_id: str, ground_truth_labels,
                        max_iterations: int, output_dir: Path, expert_notes: str = "",
                        escalate: bool = True) -> dict:
    result = run_stardist_with_feedback(
        manager.qwen, manager.stardist_worker, image_path, "segment the individual nuclei",
        max_iterations, output_dir, ground_truth_labels=ground_truth_labels, image_id=image_id,
        expert_notes=expert_notes, escalate=escalate,
    )
    return build_note(manager.qwen, "stardist", image_id, result["history"], result["outlines_image"],
                       lower_is_better=False, chosen_iteration=result["chosen_iteration"])


def run_cellvit_trial(manager: ManagerAgent, image_path: str, image_id: str, ground_truth_counts_by_type: dict,
                       ground_truth_class_labels: dict, max_iterations: int, output_dir: Path,
                       expert_notes: str = "", escalate: bool = True) -> dict:
    result = run_cellvit_with_feedback(
        manager.qwen, manager.cellvit_client, image_path, "classify the individual nuclei by cell type",
        max_iterations, output_dir, ground_truth_counts_by_type=ground_truth_counts_by_type,
        ground_truth_class_labels=ground_truth_class_labels, stardist_worker=manager.stardist_worker,
        image_id=image_id, expert_notes=expert_notes, escalate=escalate,
    )
    return build_note(manager.qwen, "cellvit", image_id, result["history"], result["annotated_image"],
                       lower_is_better=False, chosen_iteration=result["chosen_iteration"])


def compute_batch_stats(batch: list) -> dict:
    """Pure numpy-free stdlib stats over a batch of structured notes -- no Qwen involved.
    Positive "improvement" always means better, regardless of whether the underlying metric
    is MAE (lower-is-better) or PQ (higher-is-better)."""
    def improvement(note):
        delta = note["final_score"] - note["initial_score"]
        return -delta if note["lower_is_better"] else delta

    adjustment_outcomes = {}
    for note in batch:
        for adj in note["adjustments"]:
            bucket = adjustment_outcomes.setdefault(adj["change"], {"helped": 0, "tried": 0})
            bucket["tried"] += 1
            bucket["helped"] += int(adj["helped"])

    worst = sorted(batch, key=improvement)[:2]

    return {
        "batch_size": len(batch),
        "accept_rate": round(sum(n["accepted"] for n in batch) / len(batch), 2),
        "mean_improvement": round(sum(improvement(n) for n in batch) / len(batch), 3),
        "adjustment_outcomes": adjustment_outcomes,
        "worst_image_ids": [n["image_id"] for n in worst],
    }


def summarize_and_merge(qwen, running_prompt: str, batch: list, stats: dict) -> str:
    """One Qwen text-only call (see QwenVLM.ask_text) that folds this batch's stats and
    qualitative notes into the running prompt -- reinforcing what still holds, dropping/
    downweighting whatever this batch's numbers now contradict."""
    notes_summary = json.dumps([
        {
            "image_id": n["image_id"], "agent": n["agent"],
            "initial_score": n["initial_score"], "final_score": n["final_score"],
            "accepted": n["accepted"], "output_characteristics": n["output_characteristics"],
        } for n in batch
    ], indent=2)

    prompt = (
        "You are refining a running set of notes that will guide a manager agent (you, in a "
        "future session) on how to route and tune three tools -- CountGD (counts objects), "
        "StarDist (segments nuclei with no typing), and CellViT (classifies nuclei into "
        "pathology cell types) -- when no ground truth is available to score against.\n\n"
        f"Current notes (may be empty on the first batch):\n{running_prompt or '(none yet)'}\n\n"
        f"Code-computed stats for this new batch of {len(batch)} images:\n{json.dumps(stats, indent=2)}\n\n"
        f"This batch's per-image details:\n{notes_summary}\n\n"
        "Update the notes: keep what's still supported, fold in new patterns from this batch, "
        "and downweight/remove anything this batch's stats now contradict. Focus on: (1) what "
        "kinds of feedback/adjustments reliably helped vs. didn't (flag unreliable ones as "
        "low-confidence), (2) what output characteristics tend to co-occur with a high vs. low "
        "score -- useful later when there's no ground truth to check against, (3) any "
        "image/input traits that consistently perform badly.\n"
        "Reply with ONLY the updated notes as plain text (no JSON, no preamble) -- this text is "
        "used directly as guidance in a future prompt."
    )
    return qwen.ask_text(prompt, max_new_tokens=800).strip()


def merge_escalation_feedback(qwen, running_prompt: str, image_id: str, expert_summary: str) -> str:
    """Escalation counterpart to summarize_and_merge: folds one human-informed correction (see
    resolve_escalations.py) into the running prompt instead of a whole batch's stats. Same
    keep-what-still-holds/drop-what's-contradicted framing, just triggered by a single flagged
    case instead of every --batch-size images."""
    prompt = (
        "You are refining a running set of notes that will guide a manager agent (you, in a "
        "future session) on how to route and tune three tools -- CountGD (counts objects), "
        "StarDist (segments nuclei with no typing), and CellViT (classifies nuclei into "
        "pathology cell types) -- when no ground truth is available to score against.\n\n"
        f"Current notes (may be empty):\n{running_prompt or '(none yet)'}\n\n"
        f"Image {image_id!r} was escalated to a human reviewer because the automated retry loop "
        f"never reached an acceptable result on its own. After looking at the final output, the "
        f"human talked it through with the domain expert, who summarized the correction as:\n"
        f"{expert_summary}\n\n"
        "Update the notes to fold in this correction -- reinforcing it if it's consistent with "
        "what's already there, or overriding/qualifying existing guidance if this contradicts it "
        "(a human-confirmed correction should carry more weight than an unconfirmed pattern from "
        "batch stats alone). Keep it general/transferable, not specific to this one image.\n"
        "Reply with ONLY the updated notes as plain text (no JSON, no preamble) -- this text is "
        "used directly as guidance in a future prompt."
    )
    return qwen.ask_text(prompt, max_new_tokens=800).strip()


def merge_expert_notes(qwen, expert_notes: str, image_id: str, expert_summary: str) -> str:
    """Expert-side counterpart to merge_escalation_feedback: that function updates the
    *manager*'s running_prompt (checkpoint.json) from a human-resolved escalation;this one
    updates the *expert*'s own persistent notes (checkpoint.json's expert_notes, folded into
    every future ExpertReasoner's dossier via manager_agent._apply_expert_notes) from the same
    conversation, so the domain-expert persona reasons consistently across images too, not just
    the manager. Same keep-what-still-holds/drop-what's-contradicted framing as
    merge_escalation_feedback, just aimed at a different reader."""
    prompt = (
        "You are refining a running set of private notes that will guide a domain-expert "
        "persona (you, in a future session, playing a senior specialist who privately holds "
        "ground truth) on how to reason consistently about specific regions/detections across "
        "different images of the same kind.\n\n"
        f"Current notes (may be empty):\n{expert_notes or '(none yet)'}\n\n"
        f"Image {image_id!r} was escalated to a human reviewer because the automated retry loop "
        f"never reached an acceptable result on its own. You (the expert) asked the human "
        f"questions about specific regions, and that conversation was summarized as:\n"
        f"{expert_summary}\n\n"
        "Update the notes to fold in this correction -- reinforcing it if it's consistent with "
        "what's already there, or overriding/qualifying existing guidance if this contradicts "
        "it. Keep it general/transferable morphological reasoning, not specific to this one "
        "image -- these notes will be handed to you as private background before you reason "
        "about entirely different images, so nothing image-specific belongs here.\n"
        "Reply with ONLY the updated notes as plain text (no JSON, no preamble) -- this text is "
        "used directly as private background in a future prompt."
    )
    return qwen.ask_text(prompt, max_new_tokens=800).strip()


def build_countgd_tasks(n: int, output_dir: Path, split: str = "all") -> list:
    tasks = []
    id_prefix = "bbbc005" if split == "all" else f"bbbc005_{split}"
    for i, (image, count) in enumerate(load_bbbc005_samples(n, split=split)):
        image_id = f"{id_prefix}_{i:03d}_C{count}"
        image_path = output_dir / f"{image_id}.png"
        Image.fromarray(image).save(image_path)
        tasks.append({
            "agent": "countgd", "image_id": image_id, "image_path": str(image_path),
            "ground_truth_count": count,
        })
    return tasks


def build_stardist_tasks(
    manager: ManagerAgent, n: int, fold: int, output_dir: Path, seed: int = 0, split: str = "all",
) -> list:
    """Uses StardistWorker.load_pannuke_diverse -- indices spread across as many PanNuke tissue
    types as possible, fetched in one batched read -- instead of the first n images (a single
    contiguous tissue block) via n separate from-scratch reads.

    n <= 0 (a CountGD-only run, e.g. --stardist-n 0) returns no tasks without touching
    manager.stardist_worker at all -- select_diverse_indices returns [] for n=0, and indexing
    that empty list's last element is what used to raise IndexError here before this guard.

    split ("all"/"train"/"test") is select_diverse_indices's own train/test partition (see its
    docstring in agentic_stardist.py) -- same idea as bbbc005.load_bbbc005_samples's split
    parameter for CountGD, guaranteeing train/test never share a PanNuke image regardless of n
    on either side."""
    if n <= 0:
        return []
    indices, images, gt_labels_list, tissues = manager.stardist_worker.load_pannuke_diverse(
        fold, n, seed=seed, split=split
    )
    id_prefix = f"pannuke_f{fold}" if split == "all" else f"pannuke_f{fold}_{split}"
    tasks = []
    for idx, image, ground_truth_labels, tissue in zip(indices, images, gt_labels_list, tissues):
        image_id = f"{id_prefix}_{idx:04d}_{tissue}"
        image_path = output_dir / f"{image_id}.png"
        Image.fromarray(image).save(image_path)
        tasks.append({
            "agent": "stardist", "image_id": image_id, "image_path": str(image_path),
            "ground_truth_labels": ground_truth_labels,
        })
    return tasks


def build_cellvit_tasks(
    manager: ManagerAgent, n: int, fold: int, output_dir: Path, seed: int = 1, split: str = "all",
) -> list:
    """CellViT counterpart to build_stardist_tasks -- same diverse-tissue index selection via
    StardistWorker (load_pannuke_diverse_with_classes), but per-class ground truth instead of one
    class-agnostic instance mask. Default seed differs from build_stardist_tasks's (0) so a
    combined run doesn't just train CellViT on the exact same images StarDist already trained on
    within the same fold. image_id is prefixed pannuke_cellvit_ (vs. StarDist's pannuke_f...) so
    the two never collide in save_checkpoint's completed_ids even if their diverse-index
    selections happen to overlap. split: see build_stardist_tasks's docstring -- same parameter,
    same guarantee, applied to CellViT's own index selection."""
    indices, images, class_counts_list, class_labels_list, tissues = manager.stardist_worker.load_pannuke_diverse_with_classes(
        fold, n, seed=seed, split=split
    )
    id_prefix = f"pannuke_cellvit_f{fold}" if split == "all" else f"pannuke_cellvit_f{fold}_{split}"
    tasks = []
    for idx, image, ground_truth_counts_by_type, ground_truth_class_labels, tissue in zip(
        indices, images, class_counts_list, class_labels_list, tissues
    ):
        image_id = f"{id_prefix}_{idx:04d}_{tissue}"
        image_path = output_dir / f"{image_id}.png"
        Image.fromarray(image).save(image_path)
        tasks.append({
            "agent": "cellvit", "image_id": image_id, "image_path": str(image_path),
            "ground_truth_counts_by_type": ground_truth_counts_by_type,
            "ground_truth_class_labels": ground_truth_class_labels,
        })
    return tasks


def interleave_tasks(*task_lists: list) -> list:
    """Round-robins N agents' task lists instead of running all of one before any of the
    others, so a time-boxed run that gets cut short still has every agent represented
    proportionally instead of the cutoff landing entirely inside whichever list came first."""
    tasks = []
    for round_tasks in zip_longest(*task_lists):
        for t in round_tasks:
            if t is not None:
                tasks.append(t)
    return tasks


def save_checkpoint(path: Path, notes: list, running_prompt: str, expert_notes: str, total_tasks: int) -> None:
    completed_ids = [n["image_id"] for n in notes]
    path.write_text(json.dumps({
        "completed_ids": completed_ids, "running_prompt": running_prompt, "expert_notes": expert_notes,
        "notes": notes,
    }, indent=2, default=str))
    print(f"[checkpoint] {path} ({len(completed_ids)}/{total_tasks} images)")


def main():
    parser = argparse.ArgumentParser(description="Train manager_agent.py's prompt against ground-truth data")
    parser.add_argument("--countgd-n", type=int, default=5, help="Number of BBBC005 training images for CountGD")
    parser.add_argument("--stardist-n", type=int, default=5, help="Number of PanNuke training images for StarDist")
    parser.add_argument("--cellvit-n", type=int, default=0, help="Number of PanNuke training images for CellViT")
    parser.add_argument("--cellvit-checkpoint", default=None, help="Path to a CellViT model checkpoint (.pth) -- required if --cellvit-n > 0")
    parser.add_argument("--cellvit-repo", default=None, help="Path to a local clone of TIO-IKIM/CellViT")
    parser.add_argument("--cellvit-gpu", type=int, default=0, help="CUDA/ROCm GPU id for CellViT inference")
    parser.add_argument("--pannuke-fold", type=int, default=1, choices=[1, 2, 3],
                         help="PanNuke ships 3 official folds -- use a different one (2 or 3) than "
                              "your training runs (1) to get an untouched held-out StarDist/CellViT set")
    parser.add_argument("--bbbc005-split", default="all", choices=["all", "train", "test"],
                         help="BBBC005 has no official folds; 'train'/'test' partition by count-sorted "
                              "index parity so the two never share an image regardless of --countgd-n "
                              "on either side. Use 'train' for training runs and 'test' for held-out eval.")
    parser.add_argument("--pannuke-split", default="all", choices=["all", "train", "test"],
                         help="Same idea as --bbbc005-split, applied to StarDist/CellViT's PanNuke "
                              "diverse-index selection within a fold (see select_diverse_indices) -- "
                              "'train'/'test' partition by index parity so the two never share an image "
                              "regardless of --stardist-n/--cellvit-n on either side. NOTE: this only "
                              "controls which images get selected -- running this script again on the "
                              "'test' split still trains on it (updates running_prompt/expert_notes); "
                              "for a real held-out evaluation that doesn't do that, use "
                              "evaluate_manager.py instead.")
    parser.add_argument("--max-iterations", type=int, default=5, help="Retry budget per training image")
    parser.add_argument("--batch-size", type=int, default=5, help="Images per aggregation round")
    parser.add_argument("--output-dir", default="./train_manager_output")
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument(
        "--resume-from", default=None,
        help="Path to a checkpoint.json (written every batch) to continue from -- already-"
             "completed image_ids are skipped and their notes/running_prompt are reloaded. "
             "--countgd-n/--stardist-n/--cellvit-n/--pannuke-fold/--pannuke-split/--bbbc005-split "
             "should match (or exceed) the run that produced the checkpoint so the same/superset "
             "image set is built.",
    )
    args = parser.parse_args()
    if args.max_iterations < 1:
        parser.error("--max-iterations must be at least 1")
    if args.cellvit_n > 0 and not args.cellvit_checkpoint:
        parser.error("--cellvit-checkpoint is required when --cellvit-n > 0")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "checkpoint.json"

    manager = ManagerAgent(
        model_id=args.model_id, cellvit_checkpoint=args.cellvit_checkpoint,
        cellvit_repo=args.cellvit_repo, cellvit_gpu=args.cellvit_gpu,
    )
    tasks = interleave_tasks(
        build_countgd_tasks(args.countgd_n, output_dir, split=args.bbbc005_split),
        build_stardist_tasks(manager, args.stardist_n, args.pannuke_fold, output_dir, split=args.pannuke_split),
        build_cellvit_tasks(manager, args.cellvit_n, args.pannuke_fold, output_dir, split=args.pannuke_split)
        if args.cellvit_n > 0 else [],
    )
    if not tasks:
        parser.error("at least one of --countgd-n / --stardist-n / --cellvit-n must be > 0")

    notes = []
    running_prompt = ""
    expert_notes = ""
    if args.resume_from:
        checkpoint = json.loads(Path(args.resume_from).read_text())
        notes = checkpoint["notes"]
        running_prompt = checkpoint["running_prompt"]
        expert_notes = checkpoint.get("expert_notes", "")  # absent in checkpoints written before this existed
        print(f"Resumed from {args.resume_from}: {len(notes)} images already done, "
              f"running prompt is {len(running_prompt)} chars, expert notes are {len(expert_notes)} chars.")

    completed_ids = {n["image_id"] for n in notes}
    remaining_tasks = [t for t in tasks if t["image_id"] not in completed_ids]
    if len(remaining_tasks) < len(tasks):
        print(f"Skipping {len(tasks) - len(remaining_tasks)} already-completed images from the resumed checkpoint.")

    batch_start = 0
    new_notes = []
    for i, task in enumerate(remaining_tasks, 1):
        print(f"=== [{i}/{len(remaining_tasks)}] {task['image_id']} ({task['agent']}) ===")
        if task["agent"] == "countgd":
            note = run_countgd_trial(
                manager, task["image_path"], task["image_id"], task["ground_truth_count"],
                args.max_iterations, output_dir, expert_notes=expert_notes,
            )
        elif task["agent"] == "cellvit":
            note = run_cellvit_trial(
                manager, task["image_path"], task["image_id"], task["ground_truth_counts_by_type"],
                task["ground_truth_class_labels"], args.max_iterations, output_dir, expert_notes=expert_notes,
            )
        else:
            note = run_stardist_trial(
                manager, task["image_path"], task["image_id"], task["ground_truth_labels"],
                args.max_iterations, output_dir, expert_notes=expert_notes,
            )
        notes.append(note)
        new_notes.append(note)
        print(f"  initial={note['initial_score']}  final={note['final_score']}  accepted={note['accepted']}")

        if i % args.batch_size == 0 or i == len(remaining_tasks):
            batch = new_notes[batch_start:i]
            batch_start = i
            stats = compute_batch_stats(batch)
            print(f"--- batch stats ---\n{json.dumps(stats, indent=2)}")
            running_prompt = summarize_and_merge(manager.qwen, running_prompt, batch, stats)
            print(f"--- updated training prompt ---\n{running_prompt}\n")
            save_checkpoint(checkpoint_path, notes, running_prompt, expert_notes, len(tasks))

    out_path = output_dir / "training_result.json"
    out_path.write_text(json.dumps({"final_prompt": running_prompt, "notes": notes}, indent=2, default=str))
    print(f"\nFinal training prompt written to {out_path}")

    if manager._stardist_worker is not None:
        manager._stardist_worker.shutdown()


if __name__ == "__main__":
    main()
