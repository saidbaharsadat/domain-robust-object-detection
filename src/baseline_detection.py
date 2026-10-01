from pathlib import Path
import argparse
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Run a pretrained YOLO detector on images, video, webcam, or a directory.")
    parser.add_argument("--source", required=True, help="Image, video, webcam index, URL, or directory.")
    parser.add_argument("--model", default="yolo11n.pt", help="Ultralytics model weights.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--device", default=None, help="Device such as 0, cpu, or cuda.")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path("results/baseline")
    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)
    model.predict(
        source=args.source,
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
        save=True,
        project=str(output_dir),
        name="predictions",
        exist_ok=True,
    )

    print(f"Saved predictions to: {output_dir / 'predictions'}")


if __name__ == "__main__":
    main()
