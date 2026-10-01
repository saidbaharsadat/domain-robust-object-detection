from pathlib import Path
import argparse
import csv
from datetime import datetime
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a YOLO detector and log condition-wise metrics.")
    parser.add_argument("--data", required=True, help="Path to Ultralytics dataset YAML.")
    parser.add_argument("--model", default="yolo11n.pt", help="Ultralytics model weights.")
    parser.add_argument("--condition", default="clean", help="Condition label written to the CSV.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def append_csv(path: Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()

    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main():
    args = parse_args()
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

    metrics_path = Path("results/metrics/condition_metrics.csv")
    append_csv(metrics_path, summary)

    print("\nCondition-wise metrics")
    for key, value in summary.items():
        print(f"{key}: {value}")
    print(f"\nAppended summary to: {metrics_path}")

    if getattr(metrics.box, "maps", None) is not None:
        class_path = Path("results/metrics") / f"class_ap_{args.condition}.csv"
        class_path.parent.mkdir(parents=True, exist_ok=True)

        names = model.names
        with class_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["class_id", "class_name", "ap50_95"])
            for class_id, ap in enumerate(metrics.box.maps):
                writer.writerow([class_id, names.get(class_id, str(class_id)), float(ap)])

        print(f"Saved class-wise AP to: {class_path}")


if __name__ == "__main__":
    main()
