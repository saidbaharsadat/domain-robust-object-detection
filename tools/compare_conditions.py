from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from domain_robust_detection.metrics import robustness_summary
from domain_robust_detection.plotting import plot_metric_by_condition


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare adverse conditions against a clean baseline.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--plot", required=True)
    parser.add_argument("--clean-condition", default="clean")
    args = parser.parse_args()

    frame = pd.read_csv(args.input)
    summary = robustness_summary(frame, clean_condition=args.clean_condition)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output, index=False)
    plot_metric_by_condition(summary, args.plot, metric="map50_95")

    print(summary.to_string(index=False))
    print(f"Saved robustness summary to: {output}")
    print(f"Saved plot to: {args.plot}")


if __name__ == "__main__":
    main()
