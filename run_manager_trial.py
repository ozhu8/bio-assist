"""
One-off trial of the new ExpertReasoner-based manager_agent.py: runs 3 CountGD (BBBC005)
and 3 StarDist (PanNuke, distinct tissue types) images, and records:
  - the full manager<->expert dialogue transcript for every iteration
  - the specialist's initial prediction (iteration 1, before any adjustment)
  - the specialist's final prediction (after the manager's final accept/revert decision)

manager_agent.py/agentic_countgd.py/agentic_stardist.py are all untouched by this script --
it only calls ManagerAgent.run() the same way the CLI does, then reshapes result["history"]
(which already contains every iteration's "dialogue" list) into a compact log.

Usage:
    GRADIO_TEMP_DIR=/home/hannah/.gradio_tmp .venv-manager/bin/python run_manager_trial.py
"""
import json
from pathlib import Path

from PIL import Image

from bbbc005 import load_bbbc005_samples
from manager_agent import ManagerAgent

OUTPUT_DIR = Path("./manager_trial_output")
PANNUKE_FOLD = 1


def summarize_run(image_id: str, agent: str, ground_truth, tissue, result: dict) -> dict:
    history = result["history"]
    initial_prediction = history[0]["predicted_count"]
    final_prediction = result["count"] if agent == "countgd" else result["num_nuclei"]
    return {
        "image_id": image_id,
        "agent": agent,
        "ground_truth": ground_truth,
        "tissue": tissue,
        "initial_prediction": initial_prediction,
        "final_prediction": final_prediction,
        "num_iterations": len(history),
        "iterations": [
            {
                "iteration": h["iteration"],
                "predicted_count": h["predicted_count"],
                "dialogue": h.get("dialogue", []),
                "accept": h["accept"],
                "feedback": h["feedback"],
                "internal_mae": h.get("internal_mae"),
                "internal_would_accept": h.get("internal_would_accept"),
                "internal_pq": h.get("pq"),
            }
            for h in history
        ],
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manager = ManagerAgent()
    results = []

    print("=== CountGD (BBBC005) x3 ===")
    for i, (image, count) in enumerate(load_bbbc005_samples(3)):
        image_id = f"bbbc005_{i:03d}_C{count}"
        image_path = OUTPUT_DIR / f"{image_id}.png"
        Image.fromarray(image).save(image_path)
        print(f"\n--- {image_id} (ground truth count={count}) ---")
        result = manager.run(
            "count the individual cells", str(image_path), max_iterations=3,
            output_dir=str(OUTPUT_DIR / image_id), ground_truth_count=count,
        )
        results.append(summarize_run(image_id, "countgd", count, None, result))

    print("\n=== StarDist (PanNuke, distinct tissues) x3 ===")
    indices, images, gt_labels_list, tissues = manager.stardist_worker.load_pannuke_diverse(
        PANNUKE_FOLD, 3, seed=0
    )
    assert len(set(tissues)) == len(tissues), f"expected distinct tissues, got {tissues}"
    for idx, image, gt_labels, tissue in zip(indices, images, gt_labels_list, tissues):
        image_id = f"pannuke_f{PANNUKE_FOLD}_{idx:04d}_{tissue}"
        image_path = OUTPUT_DIR / f"{image_id}.png"
        Image.fromarray(image).save(image_path)
        gt_count = int(gt_labels.max())
        print(f"\n--- {image_id} (tissue={tissue}, ground truth nuclei={gt_count}) ---")
        result = manager.run(
            "segment the individual nuclei", str(image_path), max_iterations=3,
            output_dir=str(OUTPUT_DIR / image_id), ground_truth_labels=gt_labels, tissue=tissue,
        )
        results.append(summarize_run(image_id, "stardist", gt_count, tissue, result))

    out_path = OUTPUT_DIR / "trial_log.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nWrote trial log to {out_path}")

    if manager._stardist_worker is not None:
        manager._stardist_worker.shutdown()


if __name__ == "__main__":
    main()
