# COCO128 Robustness Baseline

## Purpose

This experiment extends the repository's four-image COCO8 smoke test to a 128-image controlled benchmark. The goal is to measure how a fixed pretrained detector changes under several geometry-preserving visual degradations while keeping the evaluation images and annotations aligned.

## Configuration

- **Model:** YOLO11n
- **Dataset:** Ultralytics COCO128
- **Images evaluated:** 128
- **Adverse-condition severity:** 3 / 5
- **Random seed:** 42
- **Conditions:** low light, blur, fog, rain, Gaussian noise, low contrast, and color shift
- **Metrics:** precision, recall, mAP@0.5, and mAP@0.5:0.95
- **Execution:** GitHub Actions on 2026-10-02

For transformed copies, 126 label files were copied and two images had no corresponding label file in the source split. No bounding-box coordinates were modified because the implemented transformations preserve image geometry.

## Results

| Condition | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 | Relative mAP@0.5:0.95 drop |
| --- | ---: | ---: | ---: | ---: | ---: |
| Clean | 0.663 | 0.589 | 0.670 | 0.502 | reference |
| Low light | 0.749 | 0.541 | 0.661 | 0.500 | 0.4% |
| Blur | 0.683 | 0.541 | 0.635 | 0.462 | 8.0% |
| Fog | 0.728 | 0.564 | 0.646 | 0.478 | 4.8% |
| Rain | 0.647 | 0.534 | 0.607 | 0.446 | 11.2% |
| Noise | 0.540 | 0.459 | 0.501 | 0.367 | **26.9%** |
| Low contrast | 0.674 | 0.580 | 0.639 | 0.475 | 5.4% |
| Color shift | 0.683 | 0.559 | 0.651 | 0.490 | 2.5% |

![COCO128 mAP comparison](images/coco128_map50_95_by_condition.svg)

## Current interpretation

The controlled 128-image run shows a clear robustness gap for several adverse conditions. At severity 3, Gaussian noise had the largest effect on mAP@0.5:0.95, decreasing it from 0.5024 on clean images to 0.3671, a relative reduction of approximately 26.9%. Rain and blur produced smaller but still visible reductions of approximately 11.2% and 8.0%.

Low-light performance remained close to the clean result at this severity. This should not be interpreted as evidence that low illumination is generally harmless: it reflects this particular synthetic transformation, model, dataset, and severity level.

## Limitations

COCO128 is substantially stronger than the COCO8 smoke test, but it is still a small subset and the adverse conditions are synthetic. These results establish a reproducible baseline rather than a final domain-generalization conclusion.

The next experiments should add multiple severity levels and at least one real adverse-domain dataset so synthetic corruption robustness can be separated from natural domain shift.
