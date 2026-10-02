from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


REQUIRED_COLUMNS = {
    "condition",
    "precision",
    "recall",
    "map50",
    "map50_95",
}


def robustness_summary(
    rows: Iterable[dict] | pd.DataFrame,
    clean_condition: str = "clean",
) -> pd.DataFrame:
    frame = rows.copy() if isinstance(rows, pd.DataFrame) else pd.DataFrame(list(rows))
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"missing metric columns: {sorted(missing)}")

    clean_rows = frame.loc[frame["condition"] == clean_condition]
    if clean_rows.empty:
        raise ValueError(f"clean reference condition '{clean_condition}' was not found")

    clean = clean_rows.iloc[0]
    result = frame.copy()

    for metric in ("precision", "recall", "map50", "map50_95"):
        reference = float(clean[metric])
        result[f"{metric}_drop"] = reference - result[metric].astype(float)
        if reference == 0:
            result[f"{metric}_drop_percent"] = 0.0
        else:
            result[f"{metric}_drop_percent"] = (
                100.0 * (reference - result[metric].astype(float)) / reference
            )

    return result
