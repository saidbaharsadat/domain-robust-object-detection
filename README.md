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

The repository now contains a reusable Python package, experiment tools, configuration, unit tests, continuous integration, a COCO8 smoke benchmark, and completed COCO128 single-severity and multi-severity synthetic-corruption studies.

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
| **GitHub Actions** | Unit tests, smoke checks, and COCO128 benchmark execution |
| **COCO8** | Four-image smoke benchmark for validating the complete workflow |
| **COCO128** | Completed 128-image baseline and severity 1–5 synthetic-corruption study |

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

The completed COCO128 studies extend the four-image smoke benchmark. The next research stage will expand beyond this small controlled subset.

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

The retained single-severity baseline is a completed **COCO128 robustness experiment** using **YOLO11n**, **128 labeled images**, severity level **3**, and seven controlled adverse visual conditions.

| Condition | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 | Relative mAP@0.5:0.95 drop |
| --- | ---: | ---: | ---: | ---: | ---: |
| Clean | 0.663 | 0.589 | 0.670 | **0.502** | reference |
| Low light | 0.749 | 0.541 | 0.661 | 0.500 | 0.4% |
| Blur | 0.683 | 0.541 | 0.635 | 0.462 | **8.0%** |
| Fog | 0.728 | 0.564 | 0.646 | 0.478 | 4.8% |
| Rain | 0.647 | 0.534 | 0.607 | 0.446 | **11.2%** |
| Noise | 0.540 | 0.459 | 0.501 | **0.367** | **26.9%** |
| Low contrast | 0.674 | 0.580 | 0.639 | 0.475 | 5.4% |
| Color shift | 0.683 | 0.559 | 0.651 | 0.490 | 2.5% |

![COCO128 mAP comparison](docs/images/coco128_map50_95_by_condition.svg)

In this controlled 128-image run, Gaussian image noise produced the largest drop in mAP@0.5:0.95, followed by rain and blur. The experiment was executed successfully in GitHub Actions and its measured outputs are retained permanently in the repository under `docs/results/`.

A detailed experiment note is available in `docs/coco128_baseline.md`.

The earlier **COCO8 four-image run** is retained only as a smoke test for end-to-end reproducibility and should not be treated as the main experimental result.

## Completed COCO128 multi-severity study

The [2026-10-02 Actions run](https://github.com/saidbaharsadat/domain-robust-object-detection/actions/runs/37008467779) at commit `f55827508066558b7bba7996865502dbe5372d11` completed the severity-sweep step successfully. Its [job log](https://github.com/saidbaharsadat/domain-robust-object-detection/actions/runs/37008467779/job/110842222399) records all **36 evaluations**: one clean reference and seven conditions at five severity levels, each on **128 images / 929 instances**. Configuration: pretrained **YOLO11n**, image size **640**, **CPU**, corruption seed **42**.

mAP@0.5:0.95 (clean reference **0.5024**):

| Condition | S1 | S2 | S3 | S4 | S5 | S5 relative drop |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| low light | 0.5062 | 0.5082 | 0.5004 | 0.4853 | 0.4505 | 10.34% |
| blur | 0.4971 | 0.4803 | 0.4622 | 0.4434 | 0.3889 | 22.60% |
| fog | 0.4991 | 0.4924 | 0.4782 | 0.4506 | 0.4198 | 16.45% |
| rain | 0.4849 | 0.4770 | 0.4461 | 0.4127 | 0.3513 | 30.09% |
| noise | 0.5017 | 0.4507 | 0.3671 | 0.2727 | 0.1657 | 67.02% |
| low contrast | 0.5039 | 0.5001 | 0.4752 | 0.4586 | 0.4303 | 14.36% |
| color shift | 0.4963 | 0.4943 | 0.4901 | 0.4785 | 0.4660 | 7.25% |

Blur, fog, rain, noise, low contrast, and color shift decrease monotonically across severities 1–5. Low light is non-monotonic at mild severity (0.5062 → 0.5082), then falls to 0.4505. Mild low light and severity-1 low contrast slightly exceed clean mAP; this single small-set run does not establish a general improvement. Noise has the lowest mAP from severity 2 onward, reaching **0.1657 (67.02% relative decrease)** at severity 5.

The clean and severity-3 measurements exactly match the retained baseline, including **noise 0.3671 (26.94% relative decrease)**. Full precision, recall, mAP, and drop values are retained in [severity metrics](docs/results/coco128_severity_metrics.csv), with [metadata](docs/results/coco128_severity_run_metadata.json), a [severity pivot](docs/results/coco128_map50_95_by_severity.csv), and a [provenance/reproduction note](docs/coco128_severity_study.md).

The artifact upload succeeded, but its ZIP is no longer available (14-day retention; artifact listing empty and download returns 404 on 2026-10-18). The retained CSV and metadata were recovered from the job log; the original plot PNGs were not recovered. This is verified historical execution, not a new independent rerun.

These results measure **synthetic corruption robustness on the same COCO128 images**, not real-world domain generalization. They do not evaluate robust training, independent adverse-domain datasets, repeated seeds, or statistical uncertainty.

Reproduce after [setup](#setup):

```bash
python tools/run_severity_sweep.py --config configs/coco128_severity_sweep.yaml
```

The command writes severity metrics, a severity pivot, two robustness-curve PNGs, and metadata under `results/coco128_severity_sweep/`.

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
- an end-to-end reproducible smoke-benchmark workflow;
- a completed COCO128 severity 1–5 study with 36 logged evaluations and retained numerical results.

Numerical smoke-test results should be interpreted only as pipeline validation because COCO8 contains very few validation images.

## Future work

Development is intentionally incremental:

1. Extend the completed COCO128 severity study to a larger independent validation set, repeated seeds, and uncertainty estimates.
2. Add qualitative failure-case examples for clean, blur, rain, and noise.
3. Add mixed-condition training augmentation.
4. Test simple image enhancement before detection.
5. Compare baseline, augmentation, enhancement, and combined strategies.
6. Add one or more real adverse-condition datasets.
7. Expand class-wise and failure-case analysis.
8. Study whether robustness improvements transfer across datasets and camera domains.

See `docs/milestones.md` for the implementation roadmap and the separation between current work and future extensions.
