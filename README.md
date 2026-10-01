# Domain-Robust Object Detection Under Adverse Visual Conditions

**Status:** Ongoing Computer Vision Project

**Topics:** Computer Vision · Object Detection · Domain Generalization · Image Processing

This repository contains an ongoing study of how object detectors behave when the visual domain changes because of low illumination, blur, fog, rain-like artifacts, image noise, reduced contrast, and camera/color shifts.

The project is intentionally being developed in stages. The current repository focuses on establishing a reproducible baseline and measuring the performance gap between clean and degraded visual conditions before adding more advanced robustness methods.

## Current research goals

- Measure object-detection performance under clean and adverse visual conditions.
- Compare precision, recall, mAP@0.5, and mAP@0.5:0.95 across conditions.
- Identify which types of visual degradation cause the largest detection failures.
- Document missed detections, false positives, confidence drops, and localization errors.
- Later compare augmentation, simple image enhancement, and domain-robust training strategies.

## Milestone 1 — Baseline and adverse-condition evaluation

The first implementation stage includes:

1. A lightweight pretrained YOLO baseline.
2. Prediction on normal images or videos.
3. Synthetic generation of several adverse visual conditions.
4. Evaluation on a labeled YOLO-format dataset.
5. CSV-based logging of condition-wise detection metrics.

The repository does **not** yet claim a completed domain-generalization method. Robust training and ablation studies are planned for later milestones.

## Repository structure

```text
domain-robust-object-detection/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── README.md
├── experiments/
│   └── README.md
├── results/
│   └── README.md
└── src/
    ├── baseline_detection.py
    ├── create_adverse_conditions.py
    └── evaluate_detector.py
```

## Setup

Python 3.10+ is recommended.

```bash
python -m venv .venv
```

Activate the environment, then install:

```bash
pip install -r requirements.txt
```

Ultralytics automatically downloads the selected pretrained YOLO weight the first time it is used.

## 1. Run baseline detection

Use a single image, video, webcam, or image directory:

```bash
python src/baseline_detection.py --source path/to/images
```

Example with an explicit model:

```bash
python src/baseline_detection.py --source path/to/images --model yolo11n.pt
```

Predictions are saved under `results/baseline/`.

## 2. Create adverse visual conditions

Generate one degradation at a time:

```bash
python src/create_adverse_conditions.py \
  --input path/to/clean/images \
  --output data/adverse/low_light \
  --condition low_light \
  --severity 3
```

Available conditions:

- `low_light`
- `blur`
- `fog`
- `rain`
- `noise`
- `low_contrast`
- `color_shift`

Severity is an integer from 1 to 5.

These transformations preserve image geometry, so existing bounding-box labels can be reused when the same filenames and directory structure are maintained.

## 3. Evaluate a labeled dataset

The evaluation script expects a standard Ultralytics/YOLO dataset YAML file.

```bash
python src/evaluate_detector.py \
  --data path/to/data.yaml \
  --condition clean
```

Run the same command for each adverse-condition dataset and change `--condition` accordingly.

The script appends results to:

```text
results/metrics/condition_metrics.csv
```

and writes class-wise AP values when they are available.

## Metrics

The initial comparison records:

- Precision
- Recall
- mAP@0.5
- mAP@0.5:0.95
- Per-class AP where available

No experimental numbers are included in the repository until they are produced by actual runs.

## Planned next milestones

**Milestone 2 — Robust augmentation**

Train a small detector using mixed brightness, blur, noise, and weather-style augmentation and compare it with the clean baseline.

**Milestone 3 — Image enhancement**

Test simple preprocessing such as gamma correction, CLAHE, and denoising before detection.

**Milestone 4 — Ablation and failure analysis**

Compare baseline, augmentation, enhancement, and combined strategies, then organize representative failure cases by visual condition.

## Project scope

This is a practical research project rather than a finished benchmark. The implementation is kept intentionally moderate so each method can be added, tested, and documented separately as the study progresses.
