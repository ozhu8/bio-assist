"""
Evaluates manager_agent.py's trained running_prompt/expert_notes against held-out test-split
data, WITHOUT further training on it -- the genuine train/test counterpart to train_manager.py.

train_manager.py updates expert_notes (and, in principle, running_prompt -- see the note below)
from EVERY image it processes, including whatever --*-split you point it at, so running it a
second time against a "test" split still trains on that data; it is not a real held-out
evaluation on its own. This script instead loads an already-trained checkpoint.json's
running_prompt/expert_notes as FIXED, read-only inputs (never reassigned, never written back),
reuses train_manager.py's own task-building/trial-running/scoring machinery
(build_countgd_tasks/build_stardist_tasks/build_cellvit_tasks, run_*_trial,
compute_batch_stats) against test-split images, and reports accuracy without ever touching the
trained checkpoint -- or the escalation_queue (escalate=False on every trial here, so a
held-out result that never gets accepted can't leak back into training later via
resolve_escalations.py either, which is the other back door a real train/test split needs
closed, not just non-overlapping image selection).

Note on running_prompt: it's loaded and reported here for visibility, but nothing in the
retry loop (run_*_with_feedback) actually consumes it today -- only expert_notes does (via
manager_agent._apply_expert_notes). It's meant for eventual use as manager_agent.py's own
system-prompt guidance at real inference time, not something this script's scoring depends on.

DeepGleason is intentionally not included -- there is no dataset/ground-truth integration for
it yet (see CLAUDE.md), so there is no test split to evaluate against.

Usage:
    python evaluate_manager.py --trained-checkpoint ./train_manager_output/checkpoint.json \
        --stardist-n 20 --cellvit-n 20 --cellvit-checkpoint models/CellViT-SAM-H-x40.pth \
        --cellvit-repo CellViT --pannuke-fold 1 --output-dir ./evaluate_manager_output
"""
import argparse
import json
from pathlib import Path

from manager_agent import ManagerAgent, MODEL_ID, format_qa_log
from train_manager import (
    build_cellvit_tasks, build_countgd_tasks, build_stardist_tasks, compute_batch_stats,
    interleave_tasks, run_cellvit_trial, run_countgd_trial, run_stardist_trial,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--trained-checkpoint", required=True,
        help="Path to a checkpoint.json written by train_manager.py -- its running_prompt/"
             "expert_notes are loaded as FIXED inputs for this evaluation, never updated or "
             "written back to that file.",
    )
    parser.add_argument("--countgd-n", type=int, default=0, help="Number of BBBC005 test images for CountGD")
    parser.add_argument("--stardist-n", type=int, default=0, help="Number of PanNuke test images for StarDist")
    parser.add_argument("--cellvit-n", type=int, default=0, help="Number of PanNuke test images for CellViT")
    parser.add_argument("--cellvit-checkpoint", default=None, help="Path to a CellViT model checkpoint (.pth) -- required if --cellvit-n > 0")
    parser.add_argument("--cellvit-repo", default=None, help="Path to a local clone of TIO-IKIM/CellViT")
    parser.add_argument("--cellvit-gpu", type=int, default=0, help="CUDA/ROCm GPU id for CellViT inference")
    parser.add_argument("--pannuke-fold", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--bbbc005-split", default="test", choices=["all", "train", "test"],
                         help="Defaults to 'test' here (unlike train_manager.py's 'all' default) -- "
                              "evaluation should normally run against data the checkpoint never saw.")
    parser.add_argument("--pannuke-split", default="test", choices=["all", "train", "test"],
                         help="Same default-to-test reasoning as --bbbc005-split, for StarDist/CellViT.")
    parser.add_argument("--max-iterations", type=int, default=5)
    parser.add_argument("--output-dir", default="./evaluate_manager_output")
    parser.add_argument("--model-id", default=MODEL_ID)
    args = parser.parse_args()
    if args.max_iterations < 1:
        parser.error("--max-iterations must be at least 1")
    if args.cellvit_n > 0 and not args.cellvit_checkpoint:
        parser.error("--cellvit-checkpoint is required when --cellvit-n > 0")

    trained_checkpoint = json.loads(Path(args.trained_checkpoint).read_text())
    running_prompt = trained_checkpoint.get("running_prompt", "")
    expert_notes = trained_checkpoint.get("expert_notes", "")
    escalation_qa_log = trained_checkpoint.get("escalation_qa_log", [])
    # Same fold-in as train_manager.py's expert_context -- the raw, never-paraphrased Q&A log
    # alongside the LLM-summarized expert_notes -- so held-out evaluation sees exactly what a
    # real training/inference run would, not a subset of it. Still read-only: this local
    # variable is never written back to args.trained_checkpoint.
    expert_context = expert_notes + (("\n\n" + format_qa_log(escalation_qa_log)) if escalation_qa_log else "")
    print(f"Loaded {args.trained_checkpoint}: running_prompt is {len(running_prompt)} chars, "
          f"expert_notes are {len(expert_notes)} chars, escalation_qa_log has "
          f"{len(escalation_qa_log)} resolved case(s). All held FIXED for this run -- none of "
          f"it is updated or written back to that checkpoint.")

    for flag_name, split_value in (("--bbbc005-split", args.bbbc005_split), ("--pannuke-split", args.pannuke_split)):
        if split_value == "all":
            print(f"WARNING: {flag_name}=all evaluates against the FULL dataset, which may overlap "
                  f"with whatever images {args.trained_checkpoint} was actually trained on -- pass "
                  f"'test' (the default) unless you specifically mean to do this.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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
    for i, task in enumerate(tasks, 1):
        print(f"=== [{i}/{len(tasks)}] {task['image_id']} ({task['agent']}) ===")
        if task["agent"] == "countgd":
            note = run_countgd_trial(
                manager, task["image_path"], task["image_id"], task["ground_truth_count"],
                args.max_iterations, output_dir, expert_notes=expert_context, escalate=False,
            )
        elif task["agent"] == "cellvit":
            note = run_cellvit_trial(
                manager, task["image_path"], task["image_id"], task["ground_truth_counts_by_type"],
                task["ground_truth_class_labels"], args.max_iterations, output_dir,
                expert_notes=expert_context, escalate=False,
            )
        else:
            note = run_stardist_trial(
                manager, task["image_path"], task["image_id"], task["ground_truth_labels"],
                args.max_iterations, output_dir, expert_notes=expert_context, escalate=False,
            )
        notes.append(note)
        print(f"  initial={note['initial_score']}  final={note['final_score']}  accepted={note['accepted']}")

    stats_by_agent = {agent: compute_batch_stats([n for n in notes if n["agent"] == agent])
                       for agent in sorted({n["agent"] for n in notes})}
    overall_stats = compute_batch_stats(notes)

    result = {
        "trained_checkpoint": str(args.trained_checkpoint),
        "splits": {"bbbc005_split": args.bbbc005_split, "pannuke_split": args.pannuke_split},
        "overall_stats": overall_stats, "stats_by_agent": stats_by_agent, "notes": notes,
    }
    out_path = output_dir / "eval_result.json"
    out_path.write_text(json.dumps(result, indent=2, default=str))

    print(f"\n=== Evaluation summary ({len(notes)} images, trained_checkpoint={args.trained_checkpoint}) ===")
    for agent, stats in stats_by_agent.items():
        print(f"  {agent}: n={stats['batch_size']}  accept_rate={stats['accept_rate']}  "
              f"mean_improvement={stats['mean_improvement']}")
    print(f"  overall: accept_rate={overall_stats['accept_rate']}  mean_improvement={overall_stats['mean_improvement']}")
    print(f"\nFull results written to {out_path}")

    if manager._stardist_worker is not None:
        manager._stardist_worker.shutdown()


if __name__ == "__main__":
    main()
