# Experiment Log

| Date | Experiment | Dataset | Model | Conditions | Severity | Main output | Notes |
|---|---|---|---|---|---:|---|---|
| 2026-10-02 | End-to-end smoke benchmark | COCO8 validation (4 images, 17 instances) | YOLO11n | clean + 7 adverse conditions | 3 | condition metrics + robustness summary + plot | Completed successfully in GitHub Actions. Reproducibility check only; not used as the main research result. |
| 2026-10-02 | COCO128 robustness baseline | COCO128 (128 images) | YOLO11n | clean + 7 adverse conditions | 3 | metrics CSV + robustness summary + result figure | Completed successfully. Clean mAP50-95 = 0.5024. Noise = 0.3671 (-26.94%), rain = 0.4461 (-11.21%), blur = 0.4622 (-8.00%). Primary current baseline. |
| TBD | Multi-severity robustness benchmark | COCO128 or larger subset | YOLO11n | clean + adverse conditions, severities 1-5 | 1-5 | condition/severity robustness curves | Planned next experiment. |
| TBD | Robust augmentation comparison | TBD | TBD | mixed adverse training | TBD | baseline vs robust model | Future work. |
| TBD | Real adverse-domain evaluation | TBD | TBD | real low-light/weather domains | N/A | cross-domain metrics | Future work. |
