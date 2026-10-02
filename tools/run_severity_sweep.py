from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml
from ultralytics import YOLO
from ultralytics.utils import SETTINGS

from domain_robust_detection.dataset import build_adverse_split, write_yolo_dataset_yaml


def evaluate(
    model: YOLO,
    data: str,
    condition: str,
    severity: int,
    imgsz: int,
    device: str | None,
) -> dict:
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
        "severity": severity,
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
    }


def add_clean_relative_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    clean = frame.loc[frame["condition"] == "clean"].iloc[0]
    result = frame.copy()

    for metric in ("precision", "recall", "map50", "map50_95"):
        reference = float(clean[metric])
        result[f"{metric}_drop"] = reference - result[metric].astype(float)
        result[f"{metric}_drop_percent"] = (
            100.0 * (reference - result[metric].astype(float)) / reference
            if reference != 0
            else 0.0
        )
    return result


def plot_curves(frame: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6), dpi=170)

    clean_value = float(frame.loc[frame["condition"] == "clean", "map50_95"].iloc[0])
    severities = sorted(
        int(value)
        for value in frame.loc[frame["condition"] != "clean", "severity"].unique()
    )
    ax.plot(
        severities,
        [clean_value] * len(severities),
        marker="o",
        linestyle="--",
        label="clean reference",
    )

    for condition, subset in frame.loc[frame["condition"] != "clean"].groupby("condition"):
        subset = subset.sort_values("severity")
        ax.plot(
            subset["severity"],
            subset["map50_95"],
            marker="o",
            label=condition.replace("_", " "),
        )

    ax.set_xlabel("Adverse-condition severity")
    ax.set_ylabel("mAP@0.5:0.95")
    ax.set_title("COCO128 robustness across severity levels")
    ax.set_xticks(severities)
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_relative_drop(frame: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    adverse = frame.loc[frame["condition"] != "clean"].copy()

    fig, ax = plt.subplots(figsize=(10, 6), dpi=170)
    for condition, subset in adverse.groupby("condition"):
        subset = subset.sort_values("severity")
        ax.plot(
            subset["severity"],
            subset["map50_95_drop_percent"],
            marker="o",
            label=condition.replace("_", " "),
        )

    ax.axhline(0.0, linewidth=1)
    ax.set_xlabel("Adverse-condition severity")
    ax.set_ylabel("Relative mAP@0.5:0.95 drop from clean (%)")
    ax.set_title("COCO128 relative robustness degradation")
    ax.set_xticks(sorted(int(value) for value in adverse["severity"].unique()))
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a multi-severity adverse-condition robustness study."
    )
    parser.add_argument("--config", default="configs/coco128_severity_sweep.yaml")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    output = Path(config.get("output_dir", "results/coco128_severity_sweep"))
    output.mkdir(parents=True, exist_ok=True)

    model = YOLO(config.get("model", "yolo11n.pt"))
    dataset = config.get("dataset", "coco128.yaml")
    dataset_root_name = config.get("dataset_root_name", Path(dataset).stem)
    image_subdir = config.get("image_subdir", "images/train2017")
    label_subdir = config.get("label_subdir", "labels/train2017")
    benchmark_name = config.get("benchmark_name", "COCO128 severity sweep")
    imgsz = int(config.get("imgsz", 640))
    device = config.get("device", "cpu")
    seed = int(config.get("seed", 42))
    severities = [int(x) for x in config.get("severities", [1, 2, 3, 4, 5])]
    conditions = list(config.get("conditions", []))

    rows = [evaluate(model, dataset, "clean", 0, imgsz, device)]

    dataset_root = Path(SETTINGS["datasets_dir"]) / dataset_root_name
    image_root = dataset_root / image_subdir
    label_root = dataset_root / label_subdir
    if not image_root.exists() or not label_root.exists():
        raise RuntimeError(
            f"Expected labeled data under {image_root} and {label_root}, but it was not found."
        )

    preparation: dict[str, dict] = {}

    for severity in severities:
        for condition in conditions:
            key = f"{condition}_s{severity}"
            condition_root = output / "datasets" / key

            prep = build_adverse_split(
                image_root=image_root,
                output_image_root=condition_root / "images" / "val",
                label_root=label_root,
                output_label_root=condition_root / "labels" / "val",
                condition=condition,
                severity=severity,
                seed=seed,
            )
            preparation[key] = prep

            condition_yaml = write_yolo_dataset_yaml(
                condition_root / "data.yaml",
                dataset_root=condition_root,
                names=model.names,
            )
            rows.append(
                evaluate(
                    model,
                    str(condition_yaml),
                    condition,
                    severity,
                    imgsz,
                    device,
                )
            )

    frame = pd.DataFrame(rows)
    summary = add_clean_relative_metrics(frame)

    metrics_csv = output / "severity_metrics.csv"
    summary.to_csv(metrics_csv, index=False)

    pivot = summary.loc[summary["condition"] != "clean"].pivot(
        index="severity",
        columns="condition",
        values="map50_95",
    )
    pivot_csv = output / "map50_95_by_severity.csv"
    pivot.to_csv(pivot_csv)

    curve_path = output / "map50_95_severity_curves.png"
    drop_path = output / "relative_drop_severity_curves.png"
    plot_curves(summary, curve_path)
    plot_relative_drop(summary, drop_path)

    adverse = summary.loc[summary["condition"] != "clean"]
    worst = adverse.loc[adverse["map50_95"].idxmin()]
    final_severity = max(severities)
    final_rows = adverse.loc[adverse["severity"] == final_severity].sort_values(
        "map50_95_drop_percent",
        ascending=False,
    )

    metadata = {
        "config": str(config_path),
        "benchmark_name": benchmark_name,
        "dataset": dataset,
        "evaluated_images": int(preparation[next(iter(preparation))]["processed_images"]),
        "severities": severities,
        "conditions": conditions,
        "seed": seed,
        "worst_observation": {
            "condition": str(worst["condition"]),
            "severity": int(worst["severity"]),
            "map50_95": float(worst["map50_95"]),
            "relative_drop_percent": float(worst["map50_95_drop_percent"]),
        },
        "severity_5_ranking": [
            {
                "condition": str(row["condition"]),
                "map50_95": float(row["map50_95"]),
                "relative_drop_percent": float(row["map50_95_drop_percent"]),
            }
            for _, row in final_rows.iterrows()
        ],
        "note": config.get("note", ""),
    }
    (output / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    with (output / "severity_metrics.csv").open("r", encoding="utf-8") as handle:
        print(handle.read())
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
