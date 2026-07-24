"""
One-off repair for escalation_queue/*.json records written before the run_*_with_feedback
filename fix in manager_agent.py (saved_path = output_dir / f"{image_id or '<agent>'}_iteration_{i}.png").
Before that fix, every image in a batch run shared the same "cellvit_iteration_N.png" /
"stardist_iteration_N.png" / "countgd_iteration_N.png" filenames, so by the time a batch run
finished, final_image_path in most escalation records pointed at whatever image was LAST to
write to that shared filename -- not the actual image the escalation is about.

This only re-derives the CellViT case (the one that showed up broken): CellViT inference is
deterministic given the same image + target_classes + prob_threshold, and each escalation
record's history already has both, plus original_image_path, which IS saved with a unique
name per image and was never overwritten. So re-running CellViT with the last history entry's
own params against original_image_path reconstructs the exact annotated overlay that was lost,
and this script re-points final_image_path at the newly-saved, uniquely-named file instead.

Usage:
    .venv-manager/bin/python repair_escalation_images.py \
        --escalation-dir train_manager_output_cellvit100_overnight/escalation_queue \
        --cellvit-checkpoint models/CellViT-SAM-H-x40.pth --cellvit-repo CellViT
"""
import argparse
import json
from pathlib import Path

from manager_agent import CellvitClient


def repair_one(cellvit_client: CellvitClient, record_path: Path) -> None:
    record = json.loads(record_path.read_text())
    if record.get("status") != "pending":
        print(f"  skip {record_path.name}: status={record.get('status')!r}, not pending")
        return
    if record["agent"] != "cellvit":
        print(f"  skip {record_path.name}: agent={record['agent']!r}, not cellvit")
        return

    last = record["history"][-1]
    target_classes = set(last["target_classes"])
    prob_threshold = last["prob_threshold"]
    iteration = last["iteration"]
    image_id = record["image_id"]
    original_image_path = record["original_image_path"]

    annotated, matched_cells, counts_by_type, all_cells = cellvit_client.run(
        original_image_path, target_classes, prob_threshold
    )
    predicted_count = len(matched_cells)
    if predicted_count != last["predicted_count"] or counts_by_type != last["counts_by_type"]:
        print(
            f"  WARNING {image_id}: re-run doesn't match recorded history "
            f"(predicted_count {predicted_count} vs {last['predicted_count']}, "
            f"counts_by_type {counts_by_type} vs {last['counts_by_type']}) -- saving anyway, "
            "but double-check this one."
        )

    new_path = Path(original_image_path).parent / f"{image_id}_iteration_{iteration}.png"
    annotated.save(new_path)

    record["final_image_path"] = str(new_path)
    record_path.write_text(json.dumps(record, indent=2, default=str))
    print(f"  repaired {record_path.name} -> {new_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--escalation-dir", required=True)
    parser.add_argument("--cellvit-checkpoint", required=True)
    parser.add_argument("--cellvit-repo", default=None)
    parser.add_argument("--cellvit-gpu", type=int, default=0)
    args = parser.parse_args()

    queue_dir = Path(args.escalation_dir)
    record_paths = sorted(queue_dir.glob("*.json"))
    print(f"{len(record_paths)} escalation record(s) in {queue_dir}.")

    cellvit_client = CellvitClient(args.cellvit_checkpoint, cellvit_repo=args.cellvit_repo, gpu=args.cellvit_gpu)
    for record_path in record_paths:
        repair_one(cellvit_client, record_path)


if __name__ == "__main__":
    main()
