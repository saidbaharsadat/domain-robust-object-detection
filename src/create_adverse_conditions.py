from pathlib import Path
import argparse
import cv2
import numpy as np


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def clamp_severity(severity: int) -> int:
    return max(1, min(5, severity))


def low_light(image, severity):
    factor = {1: 0.8, 2: 0.65, 3: 0.5, 4: 0.35, 5: 0.22}[severity]
    return np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def blur(image, severity):
    kernel = {1: 3, 2: 5, 3: 7, 4: 9, 5: 13}[severity]
    return cv2.GaussianBlur(image, (kernel, kernel), 0)


def fog(image, severity):
    alpha = {1: 0.12, 2: 0.2, 3: 0.3, 4: 0.42, 5: 0.55}[severity]
    white = np.full_like(image, 255)
    return cv2.addWeighted(image, 1.0 - alpha, white, alpha, 0)


def rain(image, severity):
    result = image.copy()
    h, w = image.shape[:2]
    rng = np.random.default_rng(42)
    count = {1: 80, 2: 140, 3: 220, 4: 320, 5: 450}[severity]
    length = {1: 8, 2: 10, 3: 12, 4: 15, 5: 18}[severity]

    for _ in range(count):
        x = int(rng.integers(0, max(1, w)))
        y = int(rng.integers(0, max(1, h)))
        cv2.line(result, (x, y), (min(w - 1, x + 3), min(h - 1, y + length)), (210, 210, 210), 1)

    return cv2.GaussianBlur(result, (3, 3), 0)


def noise(image, severity):
    sigma = {1: 5, 2: 10, 3: 18, 4: 28, 5: 40}[severity]
    rng = np.random.default_rng(42)
    gaussian = rng.normal(0, sigma, image.shape)
    return np.clip(image.astype(np.float32) + gaussian, 0, 255).astype(np.uint8)


def low_contrast(image, severity):
    factor = {1: 0.85, 2: 0.72, 3: 0.6, 4: 0.48, 5: 0.36}[severity]
    mean = np.mean(image, axis=(0, 1), keepdims=True)
    return np.clip((image.astype(np.float32) - mean) * factor + mean, 0, 255).astype(np.uint8)


def color_shift(image, severity):
    shift = {1: 8, 2: 16, 3: 25, 4: 35, 5: 50}[severity]
    b, g, r = cv2.split(image)
    r = np.clip(r.astype(np.int16) + shift, 0, 255).astype(np.uint8)
    b = np.clip(b.astype(np.int16) - shift // 2, 0, 255).astype(np.uint8)
    return cv2.merge([b, g, r])


TRANSFORMS = {
    "low_light": low_light,
    "blur": blur,
    "fog": fog,
    "rain": rain,
    "noise": noise,
    "low_contrast": low_contrast,
    "color_shift": color_shift,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Create geometry-preserving adverse visual conditions.")
    parser.add_argument("--input", required=True, help="Input image file or directory.")
    parser.add_argument("--output", required=True, help="Output directory.")
    parser.add_argument("--condition", choices=TRANSFORMS.keys(), required=True)
    parser.add_argument("--severity", type=int, default=3, help="Severity level from 1 to 5.")
    return parser.parse_args()


def process_file(src: Path, dst: Path, condition: str, severity: int):
    image = cv2.imread(str(src))
    if image is None:
        print(f"Skipping unreadable image: {src}")
        return

    transformed = TRANSFORMS[condition](image, severity)
    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), transformed)


def main():
    args = parse_args()
    severity = clamp_severity(args.severity)
    src = Path(args.input)
    out = Path(args.output)

    if src.is_file():
        process_file(src, out / src.name, args.condition, severity)
        print(f"Saved transformed image to: {out / src.name}")
        return

    files = [p for p in src.rglob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS]
    for file in files:
        relative = file.relative_to(src)
        process_file(file, out / relative, args.condition, severity)

    print(f"Processed {len(files)} images with condition={args.condition}, severity={severity}")
    print(f"Output directory: {out}")


if __name__ == "__main__":
    main()
