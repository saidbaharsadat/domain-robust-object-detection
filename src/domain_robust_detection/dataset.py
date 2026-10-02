from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import yaml

from .degradations import apply_degradation


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def image_files(root: str | Path) -> list[Path]:
    root = Path(root)
    return sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def matching_label(image_path: Path, image_root: Path, label_root: Path) -> Path:
    relative = image_path.relative_to(image_root).with_suffix(".txt")
    return label_root / relative


def build_adverse_split(
    image_root: str | Path,
    output_image_root: str | Path,
    condition: str,
    severity: int = 3,
    seed: int = 42,
    label_root: str | Path | None = None,
    output_label_root: str | Path | None = None,
) -> dict[str, int]:
    image_root = Path(image_root)
    output_image_root = Path(output_image_root)
    label_root = Path(label_root) if label_root is not None else None
    output_label_root = Path(output_label_root) if output_label_root is not None else None

    processed = 0
    copied_labels = 0
    missing_labels = 0

    for index, source in enumerate(image_files(image_root)):
        relative = source.relative_to(image_root)
        target = output_image_root / relative
        image = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if image is None:
            continue

        transformed = apply_degradation(
            image,
            condition=condition,
            severity=severity,
            seed=seed + index,
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(target), transformed):
            raise RuntimeError(f"failed to write transformed image: {target}")
        processed += 1

        if label_root is not None and output_label_root is not None:
            source_label = matching_label(source, image_root, label_root)
            target_label = output_label_root / relative.with_suffix(".txt")
            if source_label.exists():
                target_label.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_label, target_label)
                copied_labels += 1
            else:
                missing_labels += 1

    return {
        "processed_images": processed,
        "copied_labels": copied_labels,
        "missing_labels": missing_labels,
    }


def write_yolo_dataset_yaml(
    output_path: str | Path,
    dataset_root: str | Path,
    names: dict[int, str] | list[str],
    train: str = "images/val",
    val: str = "images/val",
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "path": str(Path(dataset_root).resolve()),
        "train": train,
        "val": val,
        "names": names,
    }
    output_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return output_path
