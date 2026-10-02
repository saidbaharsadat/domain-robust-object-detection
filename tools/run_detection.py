from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a pretrained YOLO detector.")
    parser.add_argument("--source", required=True)
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default="results/baseline")
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)
    model.predict(
        source=args.source,
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
        save=True,
        project=str(output),
        name="predictions",
        exist_ok=True,
    )
    print(f"Saved predictions to: {output / 'predictions'}")


if __name__ == "__main__":
    main()
