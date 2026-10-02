from __future__ import annotations

import argparse
from pathlib import Path

from domain_robust_detection.dataset import build_adverse_split
from domain_robust_detection.degradations import TRANSFORMS


def main() -> None:
    parser = argparse.ArgumentParser(description="Create geometry-preserving adverse images.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--condition", choices=sorted(TRANSFORMS), required=True)
    parser.add_argument("--severity", type=int, choices=range(1, 6), default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    summary = build_adverse_split(
        image_root=Path(args.input),
        output_image_root=Path(args.output),
        condition=args.condition,
        severity=args.severity,
        seed=args.seed,
    )
    print(
        f"Processed {summary['processed_images']} images "
        f"with condition={args.condition}, severity={args.severity}"
    )


if __name__ == "__main__":
    main()
