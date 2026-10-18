# Experiment Log

| Date | Experiment | Dataset | Model | Conditions | Severity | Main output | Notes |
|---|---|---|---|---|---:|---|---|
| 2026-10-02 | End-to-end smoke benchmark | COCO8 validation (4 images, 17 instances) | YOLO11n | clean + 7 adverse conditions | 3 | condition metrics + robustness summary + plot | Completed successfully in GitHub Actions. Reproducibility check only; not used as the main research result. |
| 2026-10-02 | COCO128 robustness baseline | COCO128 (128 images) | YOLO11n | clean + 7 adverse conditions | 3 | metrics CSV + robustness summary + result figure | Completed successfully. Clean mAP50-95 = 0.5024. Noise = 0.3671 (-26.94%), rain = 0.4461 (-11.21%), blur = 0.4622 (-8.00%). Retained single-severity baseline. |
| 2026-10-02 | Multi-severity robustness benchmark | COCO128 (128 images, 929 instances) | YOLO11n | clean + 7 adverse conditions | 1-5 | 36-row metrics CSV + severity pivot + curves + metadata | Verified successful Actions run 37008467779. Clean = 0.5024; S5 noise = 0.1657 (-67.02%), rain = 0.3513 (-30.09%), blur = 0.3889 (-22.60%). Mild low light is non-monotonic. Synthetic corruption study only. |
| TBD | Robust augmentation comparison | TBD | TBD | mixed adverse training | TBD | baseline vs robust model | Future work. |
| TBD | Real adverse-domain evaluation | TBD | TBD | real low-light/weather domains | N/A | cross-domain metrics | Future work. |

## Multi-severity execution verification (2026-10-18)

[Run 37008467779](https://github.com/saidbaharsadat/domain-robust-object-detection/actions/runs/37008467779), commit `f55827508066558b7bba7996865502dbe5372d11`, and its multi-severity step succeeded. The [job log](https://github.com/saidbaharsadat/domain-robust-object-detection/actions/runs/37008467779/job/110842222399) contains all 36 observations and metadata. Clean and all severity-3 metrics exactly match the retained baseline, including noise 0.3671 and its 26.94% relative decrease.

| Condition | S1 | S2 | S3 | S4 | S5 | S5 relative drop |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| low light | 0.5062 | 0.5082 | 0.5004 | 0.4853 | 0.4505 | 10.34% |
| blur | 0.4971 | 0.4803 | 0.4622 | 0.4434 | 0.3889 | 22.60% |
| fog | 0.4991 | 0.4924 | 0.4782 | 0.4506 | 0.4198 | 16.45% |
| rain | 0.4849 | 0.4770 | 0.4461 | 0.4127 | 0.3513 | 30.09% |
| noise | 0.5017 | 0.4507 | 0.3671 | 0.2727 | 0.1657 | 67.02% |
| low contrast | 0.5039 | 0.5001 | 0.4752 | 0.4586 | 0.4303 | 14.36% |
| color shift | 0.4963 | 0.4943 | 0.4901 | 0.4785 | 0.4660 | 7.25% |

Values are mAP@0.5:0.95; relative drops use the full-precision clean reference 0.5024436941561841. Six conditions decrease monotonically across severities 1–5; low light improves slightly from S1 to S2 before declining. Mild low light and S1 low contrast slightly exceed clean; no statistical significance is established.

Artifact 11227425998 was uploaded successfully with 14-day retention. On verification, the run's artifact list was empty and downloading that ID returned 404. Numerical CSV and metadata are recovered from logged output, not from the unavailable ZIP; original plots are not retained. See [study, provenance, reproduction, and limitations](coco128_severity_study.md).

Larger independent validation sets, repeated seeds/uncertainty, qualitative and class-wise failure analysis, robust augmentation, enhancement comparisons, and real adverse-domain evaluation remain open. Synthetic corruptions on the same images do not demonstrate real-world domain generalization.
