# Domain-Robust Object Detection Under Adverse Visual Conditions

An ongoing computer-vision project for studying how object detectors behave when illumination, blur, weather-like effects, image noise, contrast, or camera appearance changes. The current implementation establishes a reproducible clean-vs-adverse benchmark pipeline; robust training and broader real-world domain evaluation remain future work.

## Project goals

This project explores object-detection robustness when the visual domain changes.

The current work focuses on:

- evaluating a pretrained detector on clean images;
- generating controlled adverse visual conditions without changing bounding-box geometry;
- measuring precision, recall, mAP@0.5, and mAP@0.5:0.95 across conditions;
- quantifying the performance gap between clean and degraded inputs;
- recording class-wise and condition-wise behavior;
- preparing the same evaluation protocol for later augmentation, image-enhancement, and domain-robust training experiments.

## Current project stage

The repository now contains a reusable Python package, experiment tools, configuration, unit tests, continuous integration, and a small reproducibility benchmark based on the Ultralytics COCO8 validation split.

COCO8 is used only as a **smoke benchmark** to verify that the complete clean-to-adverse evaluation pipeline runs correctly. Its four validation images are not large enough to support research-level conclusions. Larger and real adverse-condition datasets are planned for later experiments.

## Pipeline

```text
clean labeled images
        |
        +--------------------------+
        |                          |
        v                          v
 baseline detector          adverse-condition generator
        |                          |
        |                    low light / blur
        |                    fog / rain / noise
        |                    low contrast / color shift
        |                          |
        +------------+-------------+
                     |
                     v
            condition-wise detection
                     |
                     v
       precision / recall / mAP metrics
                     |
                     v
          clean-to-adverse gap analysis
                     |
          +----------+-----------+
          |                      |
          v                      v
   class-wise analysis      failure examples
          |                      |
          +----------+-----------+
                     v
       later robustness strategies
```

## Tools and software used so far

| Tool / software | Current use |
| --- | --- |
| **Python** | Main implementation language |
| **Ultralytics YOLO** | Baseline object detection and validation |
| **OpenCV** | Image loading and adverse-condition transformations |
| **NumPy** | Numerical image operations and deterministic corruptions |
| **Pandas** | Condition-wise metric tables and robustness summaries |
| **PyYAML** | Experiment configuration |
| **Matplotlib** | Metric comparison plots |
| **pytest** | Unit tests for degradation and robustness utilities |
| **Git / GitHub** | Version control, documentation, and experiment tracking |
| **GitHub Actions** | Reproducible unit tests and smoke benchmark execution |
| **COCO8** | Small current smoke benchmark for validating the complete workflow |

### Dataset status

The present stage is **dataset-based**. No robotics simulator is required for this project. Synthetic adverse conditions are first applied to a clean labeled validation set so that the same ground-truth boxes can be reused and the effect of visual degradation can be isolated.

## Adverse conditions implemented

The current implementation supports five severity levels for:

- low illumination;
- Gaussian blur;
- fog-like contrast loss;
- rain-like streaks;
- Gaussian image noise;
- reduced contrast;
- camera/color shift.

All current transformations preserve image dimensions and object geometry.

## Planned datasets and extensions

The next research stage will move beyond the tiny smoke benchmark.

| Dataset / method | Planned use |
| --- | --- |
| **Larger COCO subset** | More stable clean-vs-synthetic corruption measurements |
| **BDD100K or another driving dataset** | Natural variation in illumination, weather, and camera scenes |
| **ExDark or another low-light detection dataset** | Real low-illumination evaluation |
| **Mixed-condition augmentation** | Train with controlled adverse transformations |
| **Gamma correction / CLAHE / denoising** | Test simple preprocessing before detection |
| **Domain-robust training strategies** | Reduce the cross-condition detection gap |
| **Failure-case analysis** | Organize misses, false positives, confidence drops, and localization errors |

Candidate external datasets will be selected according to task fit, annotation compatibility, and licensing before larger experiments are reported.

## Repository layout

```text
.github/workflows/             CI and reproducible smoke benchmark
configs/                       experiment configuration
data/                          local datasets (large files ignored)
docs/                          milestones and experiment log
results/                       generated outputs (ignored except documentation)
src/domain_robust_detection/   reusable Python package
tests/                         unit tests
tools/                         command-line experiment scripts
```

## Setup

Python 3.10+ is recommended.

```bash
python -m venv .venv
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install the project:

```bash
pip install -e .
```

## 1. Run baseline detection

```bash
python tools/run_detection.py \
  --source path/to/images \
  --model yolo11n.pt
```

Predictions are written under `results/baseline/predictions/`.

## 2. Create an adverse image set

```bash
python tools/create_adverse_conditions.py \
  --input path/to/clean/images \
  --output data/adverse/low_light \
  --condition low_light \
  --severity 3
```

## 3. Prepare a labeled adverse validation split

When YOLO-format labels are available:

```bash
python tools/prepare_adverse_dataset.py \
  --images path/to/images/val \
  --labels path/to/labels/val \
  --output data/adverse/blur \
  --condition blur \
  --severity 3
```

The image geometry is preserved, so label files are copied without changing normalized bounding boxes.

## 4. Evaluate one condition

```bash
python tools/evaluate_detector.py \
  --data path/to/data.yaml \
  --condition clean \
  --output results/metrics/condition_metrics.csv
```

## 5. Compare conditions

After clean and adverse evaluations have been written to one CSV:

```bash
python tools/compare_conditions.py \
  --input results/metrics/condition_metrics.csv \
  --output results/metrics/robustness_summary.csv \
  --plot results/metrics/map50_95_by_condition.png
```

The summary reports absolute and relative metric drops against the clean baseline.

## Reproducible smoke benchmark

The repository includes a small end-to-end benchmark configuration:

```bash
python tools/run_smoke_benchmark.py --config configs/coco8_smoke.yaml
```

The command:

1. evaluates a pretrained YOLO model on clean COCO8 validation images;
2. generates each configured adverse condition;
3. evaluates the same model on every transformed validation set;
4. saves a condition-wise metrics CSV;
5. computes clean-to-adverse robustness gaps;
6. renders a mAP@0.5:0.95 comparison plot.

The GitHub Actions workflow runs the same pipeline and uploads the generated experiment outputs as artifacts.

## Metrics

The current protocol records:

- precision;
- recall;
- mAP@0.5;
- mAP@0.5:0.95;
- class-wise AP when available;
- absolute metric drop from clean;
- relative metric drop from clean.

## Current benchmark result

A complete end-to-end smoke benchmark has been executed successfully in GitHub Actions using **YOLO11n**, the **COCO8 validation split (4 images, 17 labeled instances)**, and severity level **3** for seven controlled adverse conditions.

| Condition | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 | mAP@0.5:0.95 change vs. clean |
| --- | ---: | ---: | ---: | ---: | ---: |
| Clean | 0.570 | 0.850 | 0.846 | 0.630 | reference |
| Low light | 0.609 | 0.839 | 0.861 | 0.651 | +3.3% |
| Blur | 0.772 | 0.633 | 0.877 | 0.579 | -8.2% |
| Fog | 0.795 | 0.623 | 0.847 | 0.627 | -0.5% |
| Rain | 0.591 | 0.900 | 0.889 | 0.570 | -9.5% |
| Noise | 0.617 | 0.643 | 0.564 | 0.383 | **-39.2%** |
| Low contrast | 0.892 | 0.701 | 0.933 | 0.635 | +0.7% |
| Color shift | 0.895 | 0.701 | 0.840 | 0.641 | +1.6% |

In this small reproducibility run, Gaussian image noise produced the largest mAP@0.5:0.95 degradation, followed by rain and blur. These values are **not treated as research-scale conclusions** because COCO8 contains only four validation images; they demonstrate that the full condition-generation, label-preservation, evaluation, logging, and robustness-gap pipeline works end to end.

The completed GitHub Actions run also generated:

- `condition_metrics.csv`;
- `robustness_summary.csv`;
- `map50_95_by_condition.png`;
- `run_metadata.json`.

A stronger **COCO128 (128-image)** robustness baseline has now been added to the workflow and is the next dataset-scale step before moving to real adverse-domain datasets.

## Current evidence

At the current stage, the repository demonstrates:

- a reusable adverse-condition generation package;
- deterministic corruption controls and severity levels;
- YOLO clean/adverse validation tools;
- label-preserving adverse dataset preparation;
- condition-wise metric logging;
- clean-to-adverse gap analysis;
- automated metric visualization;
- unit tests;
- continuous integration;
- an end-to-end reproducible smoke-benchmark workflow.

Numerical smoke-test results should be interpreted only as pipeline validation because COCO8 contains very few validation images.

## Future work

Development is intentionally incremental:

1. Keep the completed COCO8 run as a reproducibility smoke test.
2. Run and document the configured COCO128 baseline for a more stable 128-image synthetic-corruption comparison.
3. Add mixed-condition training augmentation.
4. Test simple image enhancement before detection.
5. Compare baseline, augmentation, enhancement, and combined strategies.
6. Add one or more real adverse-condition datasets.
7. Expand class-wise and failure-case analysis.
8. Study whether robustness improvements transfer across datasets and camera domains.

See `docs/milestones.md` for the implementation roadmap and the separation between current work and future extensions.
