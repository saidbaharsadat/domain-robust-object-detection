from __future__ import annotations

import argparse
import json
from pathlib import Path

from domain_robust_detection.dataset import build_adverse_split
from domain_robust_detection.degradations import TRANSFORMS


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create an adverse YOLO validation split while preserving label geometry."
    )
    parser.add_argument("--images", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--condition", choices=sorted(TRANSFORMS), required=True)
    parser.add_argument("--severity", type=int, choices=range(1, 6), default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output = Path(args.output)
    summary = build_adverse_split(
        image_root=Path(args.images),
        output_image_root=output / "images" / "val",
        label_root=Path(args.labels),
        output_label_root=output / "labels" / "val",
        condition=args.condition,
        severity=args.severity,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
