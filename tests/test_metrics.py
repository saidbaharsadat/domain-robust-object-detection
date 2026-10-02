import pandas as pd
import pytest

from domain_robust_detection.metrics import robustness_summary


def test_robustness_summary_uses_clean_reference():
    frame = pd.DataFrame(
        [
            {"condition": "clean", "precision": 0.8, "recall": 0.7, "map50": 0.6, "map50_95": 0.5},
            {"condition": "blur", "precision": 0.7, "recall": 0.6, "map50": 0.48, "map50_95": 0.35},
        ]
    )
    result = robustness_summary(frame)
    blur = result.loc[result["condition"] == "blur"].iloc[0]

    assert blur["map50_95_drop"] == pytest.approx(0.15)
    assert blur["map50_95_drop_percent"] == pytest.approx(30.0)


def test_missing_clean_reference_raises():
    frame = pd.DataFrame(
        [{"condition": "blur", "precision": 0.7, "recall": 0.6, "map50": 0.5, "map50_95": 0.4}]
    )
    with pytest.raises(ValueError):
        robustness_summary(frame)
