import numpy as np
import pytest

from domain_robust_detection.degradations import TRANSFORMS, apply_degradation


def test_all_degradations_preserve_shape_and_dtype():
    image = np.full((32, 48, 3), 120, dtype=np.uint8)
    for condition in TRANSFORMS:
        output = apply_degradation(image, condition, severity=3, seed=7)
        assert output.shape == image.shape
        assert output.dtype == np.uint8


def test_low_light_reduces_mean_intensity():
    image = np.full((20, 20, 3), 180, dtype=np.uint8)
    output = apply_degradation(image, "low_light", severity=3)
    assert output.mean() < image.mean()


def test_noise_is_deterministic_for_same_seed():
    image = np.full((20, 20, 3), 100, dtype=np.uint8)
    first = apply_degradation(image, "noise", severity=3, seed=123)
    second = apply_degradation(image, "noise", severity=3, seed=123)
    assert np.array_equal(first, second)


def test_invalid_severity_raises():
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        apply_degradation(image, "blur", severity=6)
