from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path

from ultralytics import YOLO


def append_csv(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=row.keys())
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate YOLO and log condition-wise metrics.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--condition", default="clean")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default="results/metrics/condition_metrics.csv")
    args = parser.parse_args()

    model = YOLO(args.model)
    metrics = model.val(
        data=args.data,
        imgsz=args.imgsz,
        device=args.device,
        plots=False,
        save_json=False,
    )

    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": args.model,
        "condition": args.condition,
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
    }

    output = Path(args.output)
    append_csv(output, summary)
    print(summary)
    print(f"Appended metrics to: {output}")

    maps = getattr(metrics.box, "maps", None)
    if maps is not None:
        class_path = output.parent / f"class_ap_{args.condition}.csv"
        with class_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["class_id", "class_name", "ap50_95"])
            for class_id, ap in enumerate(maps):
                writer.writerow([class_id, model.names.get(class_id, str(class_id)), float(ap)])
        print(f"Saved class-wise AP to: {class_path}")


if __name__ == "__main__":
    main()
