from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml
from ultralytics import YOLO
from ultralytics.utils import SETTINGS

from domain_robust_detection.dataset import build_adverse_split, write_yolo_dataset_yaml
from domain_robust_detection.metrics import robustness_summary
from domain_robust_detection.plotting import plot_metric_by_condition


def evaluate(model: YOLO, data: str, condition: str, imgsz: int, device: str | None) -> dict:
    metrics = model.val(
        data=data,
        imgsz=imgsz,
        device=device,
        plots=False,
        save_json=False,
        verbose=False,
    )
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": str(getattr(model, "ckpt_path", "model")),
        "condition": condition,
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run an end-to-end adverse-condition object-detection benchmark."
    )
    parser.add_argument("--config", default="configs/coco8_smoke.yaml")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    output = Path(config.get("output_dir", "results/coco8_smoke"))
    output.mkdir(parents=True, exist_ok=True)

    model = YOLO(config.get("model", "yolo11n.pt"))
    dataset = config.get("dataset", "coco8.yaml")
    dataset_root_name = config.get("dataset_root_name", Path(dataset).stem)
    image_subdir = config.get("image_subdir", "images/val")
    label_subdir = config.get("label_subdir", "labels/val")
    benchmark_name = config.get("benchmark_name", dataset_root_name)
    imgsz = int(config.get("imgsz", 640))
    device = config.get("device", "cpu")
    severity = int(config.get("severity", 3))
    seed = int(config.get("seed", 42))
    conditions = list(config.get("conditions", []))

    rows = [evaluate(model, dataset, "clean", imgsz, device)]

    dataset_root = Path(SETTINGS["datasets_dir"]) / dataset_root_name
    image_root = dataset_root / image_subdir
    label_root = dataset_root / label_subdir

    if not image_root.exists() or not label_root.exists():
        raise RuntimeError(
            f"Expected labeled data under {image_root} and {label_root}, but it was not found."
        )

    preparation = {}

    for condition in conditions:
        condition_root = output / "datasets" / condition
        summary = build_adverse_split(
            image_root=image_root,
            output_image_root=condition_root / "images" / "val",
            label_root=label_root,
            output_label_root=condition_root / "labels" / "val",
            condition=condition,
            severity=severity,
            seed=seed,
        )
        preparation[condition] = summary

        condition_yaml = write_yolo_dataset_yaml(
            condition_root / "data.yaml",
            dataset_root=condition_root,
            names=model.names,
        )
        rows.append(evaluate(model, str(condition_yaml), condition, imgsz, device))

    metrics_csv = output / "condition_metrics.csv"
    with metrics_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    frame = pd.DataFrame(rows)
    summary = robustness_summary(frame, clean_condition="clean")
    summary_csv = output / "robustness_summary.csv"
    summary.to_csv(summary_csv, index=False)

    plot_path = output / "map50_95_by_condition.png"
    plot_metric_by_condition(
        summary,
        plot_path,
        metric="map50_95",
        title=f"{benchmark_name} - adverse conditions at severity {severity}",
    )

    metadata = {
        "config": str(config_path),
        "benchmark_name": benchmark_name,
        "dataset": dataset,
        "dataset_root": str(dataset_root),
        "image_subdir": image_subdir,
        "label_subdir": label_subdir,
        "evaluated_images": int(preparation[conditions[0]]["processed_images"]) if conditions else None,
        "severity": severity,
        "seed": seed,
        "conditions": conditions,
        "preparation": preparation,
        "note": config.get(
            "note",
            "Controlled adverse-condition benchmark; interpret results according to dataset size and scope.",
        ),
    }
    (output / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print(summary.to_string(index=False))
    print(f"Metrics: {metrics_csv}")
    print(f"Robustness summary: {summary_csv}")
    print(f"Plot: {plot_path}")


if __name__ == "__main__":
    main()
