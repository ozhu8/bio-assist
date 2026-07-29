"""
Combined PanNuke + BBBC005 routing/blind-baseline evaluation.

For each image (PanNuke -> a generic "segment the individual nuclei" task, known-correct
specialist stardist; BBBC005 -> "count the individual cells", known-correct specialist
countgd):

  1. Routing: call manager_agent.select_agent (the real Qwen3-VL routing call
     ManagerAgent.run uses) and record what it picks. routing_correct = picked ==
     known_agent. CellViT is a routable option in select_agent's own prompt (so a wrong
     "cellvit" pick is recorded as a miss, same as any other wrong pick) but is never
     executed here -- no CellViT checkpoint/repo on this machine (see CLAUDE.md) -- there is
     no PanNuke-typed-cell or BBBC005 task in this dataset whose *known*-correct answer is
     cellvit, so it never contributes a blind-baseline row either.

  2. Blind baseline: run the image's own known-correct specialist once, at its default
     settings, with no Qwen retry/feedback loop and no ground truth influencing the run
     itself (ground truth is only used afterwards to score the blind prediction) --
     pannuke_routing_eval.py's stardist half, plus the CountGD counterpart (run_countgd +
     interpret_countgd_target, scored by absolute error against BBBC005's known count).

If BBBC005 can't be fetched at all (network failure reaching
data.broadinstitute.org), falls back automatically to a PanNuke-only run (bumped up to
--pannuke-fallback-n images) with CountGD left out entirely, per instruction.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from bbbc005 import load_bbbc005_samples
from manager_agent import (
    MODEL_ID, COUNTGD_SPACE, QwenVLM, StardistWorker, interpret_countgd_target,
    mae_accept_tolerance, run_countgd, select_agent,
)
from gradio_client import Client # pyright: ignore[reportMissingImports]

PANNUKE_TASK_DESCRIPTION = "segment the individual nuclei"
BBBC005_TASK_DESCRIPTION = "count the individual cells"


def build_pannuke_tasks(stardist_worker: StardistWorker, n: int, fold: int, split: str, seed: int, output_dir: Path) -> list:
    if n <= 0:
        return []
    indices, images, gt_labels_list, tissues = stardist_worker.load_pannuke_diverse(fold, n, seed=seed, split=split)
    tasks = []
    for idx, image, gt_labels, tissue in zip(indices, images, gt_labels_list, tissues):
        image_id = f"pannuke_f{fold}_{idx:04d}_{tissue}"
        image_path = output_dir / f"{image_id}.png"
        Image.fromarray(image).save(image_path)
        tasks.append({
            "dataset": "pannuke", "image_id": image_id, "image_path": str(image_path),
            "task_description": PANNUKE_TASK_DESCRIPTION, "known_agent": "stardist",
            "ground_truth_labels": gt_labels, "tissue": tissue,
        })
    return tasks


def build_bbbc005_tasks(n: int, split: str, output_dir: Path) -> list:
    if n <= 0:
        return []
    tasks = []
    id_prefix = "bbbc005" if split == "all" else f"bbbc005_{split}"
    for i, (image, count) in enumerate(load_bbbc005_samples(n, split=split)):
        image_id = f"{id_prefix}_{i:03d}_C{count}"
        image_path = output_dir / f"{image_id}.png"
        Image.fromarray(image).save(image_path)
        tasks.append({
            "dataset": "bbbc005", "image_id": image_id, "image_path": str(image_path),
            "task_description": BBBC005_TASK_DESCRIPTION, "known_agent": "countgd",
            "ground_truth_count": count,
        })
    return tasks


def interleave(*task_lists: list) -> list:
    from itertools import zip_longest
    tasks = []
    for round_tasks in zip_longest(*task_lists):
        for t in round_tasks:
            if t is not None:
                tasks.append(t)
    return tasks


def run_blind_stardist(stardist_worker: StardistWorker, task: dict, output_dir: Path) -> dict:
    outlines_path = output_dir / f"{task['image_id']}_blind_outlines.png"
    init_image, prob_thresh, nms_thresh = stardist_worker.init(task["image_path"])
    gt_labels = task["ground_truth_labels"]
    blind = stardist_worker.run(init_image, prob_thresh, nms_thresh, gt_labels, outlines_path)
    pq_result = blind["pq_result"]
    return {
        "prob_thresh": prob_thresh, "nms_thresh": nms_thresh,
        "predicted_nuclei": int(blind["labels"].max()), "ground_truth_nuclei": int(gt_labels.max()),
        "pq": pq_result["pq"], "mean_iou": pq_result["mean_iou"],
        "tp": pq_result["tp"], "fp": pq_result["fp"], "fn": pq_result["fn"],
    }


def run_blind_countgd(qwen: QwenVLM, countgd_client: Client, task: dict) -> dict:
    count_target = interpret_countgd_target(qwen, task["task_description"], task["image_path"])
    _, predicted_count = run_countgd(countgd_client, task["image_path"], count_target)
    ground_truth_count = task["ground_truth_count"]
    mae = abs(predicted_count - ground_truth_count)
    return {
        "count_target": count_target, "predicted_count": predicted_count,
        "ground_truth_count": ground_truth_count, "mae": mae,
        "within_tolerance": mae <= mae_accept_tolerance(ground_truth_count),
    }


def main():
    parser = argparse.ArgumentParser(description="Combined PanNuke + BBBC005 routing accuracy + blind specialist baselines")
    parser.add_argument("--pannuke-n", type=int, default=20)
    parser.add_argument("--bbbc005-n", type=int, default=25)
    parser.add_argument("--pannuke-fallback-n", type=int, default=50, help="Used instead of --pannuke-n if BBBC005 can't be fetched")
    parser.add_argument("--pannuke-fold", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--pannuke-split", default="all", choices=["all", "train", "test"])
    parser.add_argument("--bbbc005-split", default="all", choices=["all", "train", "test"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--output-dir", default="./combined_routing_eval_output")
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
        checkpoint = json.loads(checkpoint_path.read_text())
        results = checkpoint["results"]
        print(f"Resumed from {checkpoint_path}: {len(results)} image(s) already done.")
    completed_ids = {r["image_id"] for r in results}

    stardist_worker = StardistWorker()

    print(f"Fetching {args.bbbc005_n} BBBC005 image(s) (split={args.bbbc005_split})...")
    bbbc005_failed = False
    try:
        bbbc005_tasks = build_bbbc005_tasks(args.bbbc005_n, args.bbbc005_split, output_dir)
    except Exception as e:
        print(f"BBBC005 fetch failed ({e!r}) -- falling back to PanNuke-only, CountGD left out.")
        bbbc005_failed = True
        bbbc005_tasks = []

    pannuke_n = args.pannuke_fallback_n if bbbc005_failed else args.pannuke_n
    print(f"Fetching {pannuke_n} diverse PanNuke fold {args.pannuke_fold} image(s) (split={args.pannuke_split})...")
    pannuke_tasks = build_pannuke_tasks(stardist_worker, pannuke_n, args.pannuke_fold, args.pannuke_split, args.seed, output_dir)

    tasks = interleave(pannuke_tasks, bbbc005_tasks)
    print(f"{len(tasks)} total task(s): {len(pannuke_tasks)} pannuke, {len(bbbc005_tasks)} bbbc005")

    print(f"Loading Qwen ({args.model_id})...")
    qwen = QwenVLM(args.model_id)
    countgd_client = Client(COUNTGD_SPACE) if bbbc005_tasks else None

    for task in tasks:
        image_id = task["image_id"]
        if image_id in completed_ids:
            print(f"[{image_id}] already in checkpoint, skipping")
            continue

        routed_agent = select_agent(qwen, task["task_description"], task["image_path"])
        routing_correct = routed_agent == task["known_agent"]
        print(f"[{image_id}] dataset={task['dataset']}  routed={routed_agent}  known={task['known_agent']}  correct={routing_correct}")

        if task["known_agent"] == "stardist":
            blind = run_blind_stardist(stardist_worker, task, output_dir)
            print(f"[{image_id}] blind stardist: pq={blind['pq']:.3f} mean_iou={blind['mean_iou']:.3f}")
        else:
            assert countgd_client is not None
            blind = run_blind_countgd(qwen, countgd_client, task)
            print(f"[{image_id}] blind countgd: target={blind['count_target']!r} pred={blind['predicted_count']} gt={blind['ground_truth_count']} mae={blind['mae']}")

        results.append({
            "image_id": image_id,
            "dataset": task["dataset"],
            "task_description": task["task_description"],
            "known_agent": task["known_agent"],
            "routed_agent": routed_agent,
            "routing_correct": routing_correct,
            "blind_result": blind,
        })
        checkpoint_path.write_text(json.dumps({"results": results}, indent=2))

    stardist_worker.shutdown()

    n = len(results)
    routing_accuracy = sum(r["routing_correct"] for r in results) / n if n else 0.0
    by_dataset = {}
    for dataset in sorted({r["dataset"] for r in results}):
        subset = [r for r in results if r["dataset"] == dataset]
        by_dataset[dataset] = {
            "n": len(subset),
            "routing_accuracy": sum(r["routing_correct"] for r in subset) / len(subset),
        }
    stardist_results = [r["blind_result"] for r in results if r["known_agent"] == "stardist"]
    countgd_results = [r["blind_result"] for r in results if r["known_agent"] == "countgd"]
    if stardist_results:
        by_dataset["pannuke"]["blind_stardist_mean_pq"] = sum(b["pq"] for b in stardist_results) / len(stardist_results)
        by_dataset["pannuke"]["blind_stardist_mean_iou"] = sum(b["mean_iou"] for b in stardist_results) / len(stardist_results)
    if countgd_results:
        by_dataset["bbbc005"]["blind_countgd_mean_mae"] = sum(b["mae"] for b in countgd_results) / len(countgd_results)
        by_dataset["bbbc005"]["blind_countgd_within_tolerance_rate"] = sum(b["within_tolerance"] for b in countgd_results) / len(countgd_results)

    summary = {
        "n": n,
        "bbbc005_fetch_failed": bbbc005_failed,
        "routing_accuracy": routing_accuracy,
        "routed_agent_counts": {
            agent: sum(1 for r in results if r["routed_agent"] == agent)
            for agent in sorted({r["routed_agent"] for r in results})
        },
        "by_dataset": by_dataset,
        "results": results,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print("\n=== Summary ===")
    print(f"n={n}  overall routing_accuracy={routing_accuracy:.1%}  routed_agent_counts={summary['routed_agent_counts']}")
    for dataset, stats in by_dataset.items():
        print(f"  {dataset}: {stats}")
    print(f"Full results written to {summary_path}")


if __name__ == "__main__":
    main()
