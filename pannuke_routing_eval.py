"""
Two-part evaluation against a PanNuke sample:

  1. Routing accuracy: for each image, ask manager_agent.select_agent (the real Qwen3-VL
     routing call, same code path ManagerAgent.run uses) to route a generic segmentation task
     ("segment the individual nuclei"). Every PanNuke image's known-correct specialist for that
     task is stardist (PanNuke has no free-text object to count, and the task never names a
     specific cell type, so countgd/cellvit/deepgleason are all wrong picks here) -- accuracy is
     just the fraction of images select_agent actually routes to stardist. CellViT itself is
     never executed (no checkpoint/repo on this machine); a wrong "cellvit" routing pick is still
     recorded as a miss, same as any other wrong pick.

  2. Blind StarDist baseline: run_stardist (bypassing run_stardist_with_feedback's retry/Qwen-
     scoring loop entirely) once per image at StarDist's own untuned default thresholds
     (STARDIST_INITIAL_PROB_THRESH, the pretrained model's own nms threshold) -- a single-shot
     pass with no feedback loop, no ground truth involved in the run itself. Ground truth is
     only used afterwards, to score the blind prediction's panoptic quality for reporting.

No CellViT checkpoint, DeepGleason repo, or GPU is required to run this.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from manager_agent import MODEL_ID, QwenVLM, StardistWorker, select_agent

TASK_DESCRIPTION = "segment the individual nuclei"
KNOWN_AGENT = "stardist"


def main():
    parser = argparse.ArgumentParser(description="PanNuke routing accuracy + blind StarDist baseline")
    parser.add_argument("--n", type=int, default=10, help="Number of PanNuke images to evaluate")
    parser.add_argument("--fold", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--split", default="all", choices=["all", "train", "test"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--output-dir", default="./pannuke_routing_eval_output")
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from <output-dir>/checkpoint.json if it exists -- skips images already "
             "recorded there (matched by image_id) instead of redoing them. The checkpoint is "
             "rewritten to disk after every single image (not just every batch), so a run that "
             "gets interrupted mid-way -- killed, disconnected, machine put to sleep -- loses at "
             "most the one image that was in flight when it stopped.",
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

    print(f"Loading {args.n} diverse PanNuke fold {args.fold} image(s) (split={args.split})...")
    stardist_worker = StardistWorker()
    indices, images, gt_labels_list, tissues = stardist_worker.load_pannuke_diverse(
        args.fold, args.n, seed=args.seed, split=args.split
    )

    print(f"Loading Qwen ({args.model_id})...")
    qwen = QwenVLM(args.model_id)

    for idx, image, gt_labels, tissue in zip(indices, images, gt_labels_list, tissues):
        image_id = f"pannuke_f{args.fold}_{idx:04d}_{tissue}"
        if image_id in completed_ids:
            print(f"[{image_id}] already in checkpoint, skipping")
            continue
        image_path = str(output_dir / f"{image_id}.png")
        Image.fromarray(image).save(image_path)

        routed_agent = select_agent(qwen, TASK_DESCRIPTION, image_path)
        routing_correct = routed_agent == KNOWN_AGENT
        print(f"[{image_id}] tissue={tissue}  routed={routed_agent}  correct={routing_correct}")

        outlines_path = output_dir / f"{image_id}_blind_outlines.png"
        init_image, prob_thresh, nms_thresh = stardist_worker.init(image_path)
        blind = stardist_worker.run(init_image, prob_thresh, nms_thresh, gt_labels, outlines_path)
        pq_result = blind["pq_result"]
        print(
            f"[{image_id}] blind StarDist: prob_thresh={prob_thresh} nms_thresh={nms_thresh} "
            f"pred_nuclei={int(blind['labels'].max())} gt_nuclei={int(gt_labels.max())} "
            f"pq={pq_result['pq']:.3f} mean_iou={pq_result['mean_iou']:.3f}"
        )

        results.append({
            "image_id": image_id,
            "pannuke_index": idx,
            "tissue": tissue,
            "task_description": TASK_DESCRIPTION,
            "known_agent": KNOWN_AGENT,
            "routed_agent": routed_agent,
            "routing_correct": routing_correct,
            "blind_stardist": {
                "prob_thresh": prob_thresh,
                "nms_thresh": nms_thresh,
                "predicted_nuclei": int(blind["labels"].max()),
                "ground_truth_nuclei": int(gt_labels.max()),
                "pq": pq_result["pq"],
                "mean_iou": pq_result["mean_iou"],
                "tp": pq_result["tp"],
                "fp": pq_result["fp"],
                "fn": pq_result["fn"],
            },
        })
        # Written after every image (not batched) -- each Qwen call can take minutes on this
        # machine (see CLAUDE.md / no GPU here), so batching the checkpoint would still lose a
        # lot of wall-clock progress to a mid-run interruption.
        checkpoint_path.write_text(json.dumps({"results": results}, indent=2))

    stardist_worker.shutdown()

    n = len(results)
    routing_accuracy = sum(r["routing_correct"] for r in results) / n
    mean_pq = sum(r["blind_stardist"]["pq"] for r in results) / n
    mean_iou = sum(r["blind_stardist"]["mean_iou"] for r in results) / n

    summary = {
        "n": n,
        "fold": args.fold,
        "split": args.split,
        "routing_accuracy": routing_accuracy,
        "blind_stardist_mean_pq": mean_pq,
        "blind_stardist_mean_iou": mean_iou,
        "routed_agent_counts": {
            agent: sum(1 for r in results if r["routed_agent"] == agent)
            for agent in sorted({r["routed_agent"] for r in results})
        },
        "results": results,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print("\n=== Summary ===")
    print(f"n={n}  routing_accuracy={routing_accuracy:.1%}  routed_agent_counts={summary['routed_agent_counts']}")
    print(f"blind StarDist: mean_pq={mean_pq:.3f}  mean_iou={mean_iou:.3f}")
    print(f"Full results written to {summary_path}")


if __name__ == "__main__":
    main()
