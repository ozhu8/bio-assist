"""
Runs the manager's real feedback loop (ManagerAgent.run -> select_agent, then
run_stardist_with_feedback/run_countgd_with_feedback -- up to --max-iterations retries, each
with an expert dialogue + accept/reject decision) on the exact same 10 images
combined_routing_eval.py already ran blindly (same --pannuke-fold/--seed/--split defaults),
for a direct before/after comparison against that blind baseline.

Two deviations from manager_agent.py's own defaults, both budget-driven (this hardware's
per-Qwen-call cost -- see CLAUDE.md's ROCm/disk-offload notes -- makes the full defaults
impractically slow for a 10-image comparison):
  - --max-iterations 2 instead of the default 5
  - expert dialogue capped to 1 turn/iteration instead of MAX_EXPERT_TURNS=3 (monkeypatches
    run_expert_dialogue's default -- run_stardist_with_feedback/run_countgd_with_feedback never
    pass max_turns explicitly, so patching the module-level constant alone would not have
    reached the already-bound default argument)

One more substitution, NOT budget-driven: ExpertReasoner's own docstring says it "reuses the
manager's own already-loaded QwenVLM under a different persona/system framing rather than
loading a second 8B model" -- but run_stardist_with_feedback/run_countgd_with_feedback
currently construct it via get_claude_vlm() unconditionally, which needs ANTHROPIC_API_KEY (not
set in this environment, no .env in the repo either). This monkeypatches
manager_agent.get_claude_vlm to return the already-loaded ManagerAgent.qwen instead --
restoring the behavior ExpertReasoner's own docstring (and CLAUDE.md's "No API key is required
to run manager_agent.py") already describes, rather than introducing new behavior, and avoiding
both the API key and a second 8B model load.

escalate=False throughout: this is a one-off comparison run, not training, so an unaccepted
result should not get queued into the escalation_queue a human might later resolve.
"""
import argparse
import json
from pathlib import Path

import manager_agent
from combined_routing_eval import build_bbbc005_tasks, build_pannuke_tasks, interleave
from manager_agent import MODEL_ID, ManagerAgent


def _patch_expert_to_local_qwen(manager: ManagerAgent) -> None:
    manager_agent.get_claude_vlm = lambda: manager.qwen


def _cap_expert_turns(max_turns: int) -> None:
    original = manager_agent.run_expert_dialogue

    def capped(qwen, expert, task_description, predicted_summary, manager_image_path,
               expert_image_paths, max_turns=max_turns):
        return original(qwen, expert, task_description, predicted_summary, manager_image_path,
                         expert_image_paths, max_turns=max_turns)

    manager_agent.run_expert_dialogue = capped


def main():
    parser = argparse.ArgumentParser(description="Full manager feedback-loop run, for comparison against combined_routing_eval.py's blind baseline")
    parser.add_argument("--pannuke-n", type=int, default=5)
    parser.add_argument("--bbbc005-n", type=int, default=5)
    parser.add_argument("--pannuke-fold", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--pannuke-split", default="all", choices=["all", "train", "test"])
    parser.add_argument("--bbbc005-split", default="all", choices=["all", "train", "test"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-iterations", type=int, default=2)
    parser.add_argument("--expert-turns", type=int, default=1)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--output-dir", default="./feedback_loop_eval_output")
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from <output-dir>/checkpoint.json if it exists -- skips images already "
             "recorded there (matched by image_id). Checkpoint is rewritten after every image.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "checkpoint.json"

    results = []
    if args.resume and checkpoint_path.exists():
        results = json.loads(checkpoint_path.read_text())["results"]
        print(f"Resumed from {checkpoint_path}: {len(results)} image(s) already done.")
    completed_ids = {r["image_id"] for r in results}

    print(f"Loading Qwen ({args.model_id})...")
    manager = ManagerAgent(model_id=args.model_id)
    _patch_expert_to_local_qwen(manager)
    _cap_expert_turns(args.expert_turns)
    print(f"Expert dialogue now backed by the manager's own local Qwen (no API key needed), capped at {args.expert_turns} turn(s)/iteration.")

    print(f"Fetching {args.pannuke_n} PanNuke + {args.bbbc005_n} BBBC005 image(s) (same fold/seed/split as combined_routing_eval.py)...")
    pannuke_tasks = build_pannuke_tasks(manager.stardist_worker, args.pannuke_n, args.pannuke_fold, args.pannuke_split, args.seed, output_dir)
    bbbc005_tasks = build_bbbc005_tasks(args.bbbc005_n, args.bbbc005_split, output_dir)
    tasks = interleave(pannuke_tasks, bbbc005_tasks)
    print(f"{len(tasks)} total task(s): {len(pannuke_tasks)} pannuke, {len(bbbc005_tasks)} bbbc005")

    for task in tasks:
        image_id = task["image_id"]
        if image_id in completed_ids:
            print(f"[{image_id}] already in checkpoint, skipping")
            continue

        try:
            result = manager.run(
                task["task_description"], task["image_path"], args.max_iterations, str(output_dir),
                ground_truth_count=task.get("ground_truth_count"),
                ground_truth_labels=task.get("ground_truth_labels"),
                tissue=task.get("tissue"), image_id=image_id, escalate=False,
            )
        except Exception as e:
            print(f"[{image_id}] FAILED: {e!r}")
            results.append({"image_id": image_id, "dataset": task["dataset"], "known_agent": task["known_agent"], "error": repr(e)})
            checkpoint_path.write_text(json.dumps({"results": results}, indent=2, default=str))
            continue

        history = result["history"]
        final_entry = next(h for h in history if h["iteration"] == result["chosen_iteration"])
        print(f"[{image_id}] agent={result['agent']} chosen_iteration={result['chosen_iteration']}/{len(history)} accept={final_entry['accept']}")

        record = {
            "image_id": image_id,
            "dataset": task["dataset"],
            "known_agent": task["known_agent"],
            "routed_agent": result["agent"],
            "routing_correct": result["agent"] == task["known_agent"],
            "chosen_iteration": result["chosen_iteration"],
            "iterations_run": len(history),
            "accepted": final_entry["accept"],
            "history": [{k: v for k, v in h.items() if k != "dialogue"} for h in history],
        }
        if result["agent"] == "stardist":
            record["final_num_nuclei"] = result["num_nuclei"]
            record["final_pq"] = final_entry.get("pq")
            record["final_mean_iou"] = final_entry.get("mean_iou")
        elif result["agent"] == "countgd":
            record["final_count"] = result["count"]
            record["final_mae"] = final_entry.get("internal_mae")

        results.append(record)
        checkpoint_path.write_text(json.dumps({"results": results}, indent=2, default=str))

    manager.stardist_worker.shutdown()

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps({"results": results}, indent=2, default=str))

    ok_results = [r for r in results if "error" not in r]
    stardist_results = [r for r in ok_results if r["routed_agent"] == "stardist"]
    countgd_results = [r for r in ok_results if r["routed_agent"] == "countgd"]
    print(f"\n=== Summary ({len(results)} image(s), {len(ok_results)} succeeded) ===")
    if stardist_results:
        mean_pq = sum(r["final_pq"] for r in stardist_results if r["final_pq"] is not None) / len(stardist_results)
        mean_iou = sum(r["final_mean_iou"] for r in stardist_results if r["final_mean_iou"] is not None) / len(stardist_results)
        print(f"StarDist (n={len(stardist_results)}): mean final PQ={mean_pq:.3f}  mean final IoU={mean_iou:.3f}")
    if countgd_results:
        mean_mae = sum(r["final_mae"] for r in countgd_results if r["final_mae"] is not None) / len(countgd_results)
        print(f"CountGD (n={len(countgd_results)}): mean final MAE={mean_mae:.2f}")
    print(f"Full results written to {summary_path}")


if __name__ == "__main__":
    main()
