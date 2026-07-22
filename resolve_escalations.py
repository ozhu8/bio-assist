"""
Resolves cases train_manager.py/manager_agent.py queued for human review (write_escalation in
manager_agent.py) -- written whenever an image's automated retry loop (CountGD, StarDist, or
CellViT) exhausted its iterations without the manager ever reaching accept=True.

This is deliberately a separate, synchronous script rather than something the batch run itself
blocks on: train_manager.py never waits for a human, it just queues the case and moves on to the
next image. Run this whenever you have time to work through the queue; it's fine to run it
while train_manager.py is still going on other images (they don't touch each other's files).

For each pending escalation: shows you the final output image, then the domain expert (the same
ExpertReasoner used during training, still privately holding ground truth) leads -- it asks you a
specific, case-relevant question about the image (ask_human_question) rather than you guessing
what to say into a blank prompt, the same role the manager plays with the expert during training,
just with you standing in for the manager. Answer each question; an empty answer ends the
conversation. The expert then summarizes it into transferable guidance and it gets folded into
the run's running_prompt (checkpoint.json) via merge_escalation_feedback, the same
keep-what-holds/drop-what's-contradicted merge summarize_and_merge already does every batch.

Usage:
    python resolve_escalations.py --output-dir ./train_manager_output_7-21
"""
import argparse
import json
from pathlib import Path

import numpy as np  # pyright: ignore[reportMissingImports]

from manager_agent import (
    EXPERT_PERSONA_CELLVIT, EXPERT_PERSONA_COUNTGD, EXPERT_PERSONA_STARDIST, MODEL_ID, NO_GROUND_TRUTH_DOSSIER,
    ExpertReasoner, QwenVLM, build_cellvit_dossier, build_countgd_dossier, build_stardist_dossier,
    run_human_expert_dialogue,
)
from train_manager import merge_escalation_feedback

# NO_GROUND_TRUTH_DOSSIER (imported above) is used when an escalation came from a run with no
# ground truth at all -- write_escalation stores None for ground_truth_path/ground_truth_value in
# that case. Calling build_stardist_dossier(None, ...) would crash on ground_truth_labels.max();
# build_countgd_dossier(None, ...) would produce a nonsensical "Verified true object count: None."
# Both are guarded below to fall back to it instead, same as every run_*_with_feedback loop in
# manager_agent.py already does when it has no real ground truth to build a dossier from.


def load_pending(queue_dir: Path) -> list:
    records = []
    for path in sorted(queue_dir.glob("*.json")):
        record = json.loads(path.read_text())
        if record.get("status") == "pending":
            records.append((path, record))
    return records


def resolve_one(qwen: QwenVLM, record_path: Path, record: dict, checkpoint_path: Path) -> None:
    image_id = record["image_id"]
    print(f"\n=== Escalation: {image_id} ({record['agent']}) ===")
    print(f"Task: {record['task_description']}")
    print(f"Final output image: {record['final_image_path']}")
    print("Open the image above yourself -- the expert will ask you about specific things in it.\n")

    if record["agent"] == "stardist":
        ground_truth_labels = np.load(record["ground_truth_path"]) if record.get("ground_truth_path") else None
        dossier = (
            build_stardist_dossier(ground_truth_labels, record.get("tissue"))
            if ground_truth_labels is not None else NO_GROUND_TRUTH_DOSSIER
        )
        expert = ExpertReasoner(
            qwen, EXPERT_PERSONA_STARDIST, dossier,
            forbidden_values=[int(ground_truth_labels.max())] if ground_truth_labels is not None else [],
        )
    elif record["agent"] == "cellvit":
        ground_truth_counts_by_type = record.get("ground_truth_value")
        dossier = (
            build_cellvit_dossier(ground_truth_counts_by_type, record.get("tissue"))
            if ground_truth_counts_by_type is not None else NO_GROUND_TRUTH_DOSSIER
        )
        expert = ExpertReasoner(
            qwen, EXPERT_PERSONA_CELLVIT, dossier,
            forbidden_values=[int(v) for v in ground_truth_counts_by_type.values()]
            if ground_truth_counts_by_type is not None else [],
        )
    else:
        ground_truth_count = record.get("ground_truth_value")
        dossier = (
            build_countgd_dossier(ground_truth_count, record["original_image_path"])
            if ground_truth_count is not None else NO_GROUND_TRUTH_DOSSIER
        )
        expert = ExpertReasoner(
            qwen, EXPERT_PERSONA_COUNTGD, dossier,
            forbidden_values=[ground_truth_count] if ground_truth_count is not None else [],
        )

    conversation = run_human_expert_dialogue(expert, [record["final_image_path"]], record["task_description"], input)
    if not conversation:
        print("No answer given -- leaving this escalation pending.")
        return

    expert_summary = expert.summarize_for_manager(record["task_description"], image_id, conversation)
    print(f"\n[expert -> manager] {expert_summary}")

    checkpoint = json.loads(checkpoint_path.read_text())
    checkpoint["running_prompt"] = merge_escalation_feedback(
        qwen, checkpoint["running_prompt"], image_id, expert_summary
    )
    checkpoint_path.write_text(json.dumps(checkpoint, indent=2, default=str))
    print(f"Merged into {checkpoint_path}'s running prompt.")

    record["status"] = "resolved"
    record["human_conversation"] = conversation
    record["expert_summary_for_manager"] = expert_summary
    record_path.write_text(json.dumps(record, indent=2, default=str))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", required=True, help="Same --output-dir a train_manager.py run used")
    parser.add_argument("--model-id", default=MODEL_ID)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    queue_dir = output_dir / "escalation_queue"
    checkpoint_path = output_dir / "checkpoint.json"
    if not queue_dir.exists():
        print(f"No escalation queue at {queue_dir} -- nothing to resolve.")
        return
    if not checkpoint_path.exists():
        print(f"No checkpoint.json at {checkpoint_path} -- can't merge feedback anywhere. "
              f"(Escalations are still in {queue_dir}, un-resolved.)")
        return

    pending = load_pending(queue_dir)
    if not pending:
        print(f"No pending escalations in {queue_dir}.")
        return
    print(f"{len(pending)} pending escalation(s) in {queue_dir}.")

    qwen = QwenVLM(args.model_id)
    for record_path, record in pending:
        resolve_one(qwen, record_path, record, checkpoint_path)


if __name__ == "__main__":
    main()
