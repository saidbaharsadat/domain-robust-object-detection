# Experiment Log

| Date | Experiment | Dataset | Model | Conditions | Severity | Main output | Notes |
|---|---|---|---|---|---:|---|---|
| 2026-10-02 | End-to-end smoke benchmark | COCO8 validation (4 images, 17 instances) | YOLO11n | clean + 7 adverse conditions | 3 | condition metrics + robustness summary + plot | Completed successfully in GitHub Actions. Clean mAP50-95 = 0.6304. Noise was the strongest degradation at 0.3835 (-39.18%). Rain = 0.5703 (-9.54%); blur = 0.5789 (-8.17%). Small-set reproducibility result only. |
| 2026-10-02 | COCO128 robustness baseline | COCO128 (128 images) | YOLO11n | clean + 7 adverse conditions | 3 | condition/severity robustness comparison | Configured and added to GitHub Actions as the next stronger baseline. Numerical results are not recorded here until a completed run is available. |
| TBD | Larger multi-severity benchmark | TBD | YOLO11n | clean + adverse conditions, severities 1-5 | TBD | condition/severity robustness curves | Planned next experiment after baseline verification. |
| TBD | Robust augmentation comparison | TBD | TBD | mixed adverse training | TBD | baseline vs robust model | Future work. |
| TBD | Real adverse-domain evaluation | TBD | TBD | real low-light/weather domains | N/A | cross-domain metrics | Future work. |
