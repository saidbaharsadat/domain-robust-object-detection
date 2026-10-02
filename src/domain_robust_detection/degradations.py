from __future__ import annotations

from typing import Callable

import cv2
import numpy as np


Severity = int


def validate_severity(severity: int) -> int:
    if severity not in {1, 2, 3, 4, 5}:
        raise ValueError("severity must be an integer from 1 to 5")
    return severity


def low_light(image: np.ndarray, severity: Severity, seed: int = 42) -> np.ndarray:
    del seed
    factor = {1: 0.80, 2: 0.65, 3: 0.50, 4: 0.35, 5: 0.22}[validate_severity(severity)]
    return np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def blur(image: np.ndarray, severity: Severity, seed: int = 42) -> np.ndarray:
    del seed
    kernel = {1: 3, 2: 5, 3: 7, 4: 9, 5: 13}[validate_severity(severity)]
    return cv2.GaussianBlur(image, (kernel, kernel), 0)


def fog(image: np.ndarray, severity: Severity, seed: int = 42) -> np.ndarray:
    del seed
    alpha = {1: 0.12, 2: 0.20, 3: 0.30, 4: 0.42, 5: 0.55}[validate_severity(severity)]
    white = np.full_like(image, 255)
    return cv2.addWeighted(image, 1.0 - alpha, white, alpha, 0)


def rain(image: np.ndarray, severity: Severity, seed: int = 42) -> np.ndarray:
    severity = validate_severity(severity)
    result = image.copy()
    height, width = image.shape[:2]
    rng = np.random.default_rng(seed)
    count = {1: 80, 2: 140, 3: 220, 4: 320, 5: 450}[severity]
    length = {1: 8, 2: 10, 3: 12, 4: 15, 5: 18}[severity]

    for _ in range(count):
        x = int(rng.integers(0, max(1, width)))
        y = int(rng.integers(0, max(1, height)))
        end = (min(width - 1, x + 3), min(height - 1, y + length))
        cv2.line(result, (x, y), end, (210, 210, 210), 1)

    return cv2.GaussianBlur(result, (3, 3), 0)


def noise(image: np.ndarray, severity: Severity, seed: int = 42) -> np.ndarray:
    severity = validate_severity(severity)
    sigma = {1: 5, 2: 10, 3: 18, 4: 28, 5: 40}[severity]
    rng = np.random.default_rng(seed)
    gaussian = rng.normal(0, sigma, image.shape)
    return np.clip(image.astype(np.float32) + gaussian, 0, 255).astype(np.uint8)


def low_contrast(image: np.ndarray, severity: Severity, seed: int = 42) -> np.ndarray:
    del seed
    factor = {1: 0.85, 2: 0.72, 3: 0.60, 4: 0.48, 5: 0.36}[validate_severity(severity)]
    mean = np.mean(image, axis=(0, 1), keepdims=True)
    return np.clip((image.astype(np.float32) - mean) * factor + mean, 0, 255).astype(np.uint8)


def color_shift(image: np.ndarray, severity: Severity, seed: int = 42) -> np.ndarray:
    del seed
    shift = {1: 8, 2: 16, 3: 25, 4: 35, 5: 50}[validate_severity(severity)]
    blue, green, red = cv2.split(image)
    red = np.clip(red.astype(np.int16) + shift, 0, 255).astype(np.uint8)
    blue = np.clip(blue.astype(np.int16) - shift // 2, 0, 255).astype(np.uint8)
    return cv2.merge([blue, green, red])


TRANSFORMS: dict[str, Callable[[np.ndarray, int, int], np.ndarray]] = {
    "low_light": low_light,
    "blur": blur,
    "fog": fog,
    "rain": rain,
    "noise": noise,
    "low_contrast": low_contrast,
    "color_shift": color_shift,
}


def apply_degradation(
    image: np.ndarray,
    condition: str,
    severity: int = 3,
    seed: int = 42,
) -> np.ndarray:
    if condition not in TRANSFORMS:
        choices = ", ".join(sorted(TRANSFORMS))
        raise ValueError(f"unknown condition '{condition}'. Available: {choices}")
    if image is None or image.size == 0:
        raise ValueError("image must be a non-empty NumPy array")
    return TRANSFORMS[condition](image, validate_severity(severity), seed)
