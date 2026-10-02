from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_metric_by_condition(
    frame: pd.DataFrame,
    output_path: str | Path,
    metric: str = "map50_95",
    title: str = "Detection robustness by visual condition",
) -> Path:
    if metric not in frame.columns:
        raise ValueError(f"metric column '{metric}' was not found")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5), dpi=160)
    ax.bar(frame["condition"].astype(str), frame[metric].astype(float))
    ax.set_ylabel(metric)
    ax.set_xlabel("Visual condition")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path
