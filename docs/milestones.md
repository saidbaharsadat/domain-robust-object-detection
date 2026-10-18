# Implementation Milestones

> Completed scope: **M0-M3 plus the COCO128 portion of M4**. Larger independent-set validation remains open within M4; M5-M9 are future research extensions.

## M0 — Repository and reproducibility
- Create the repository and installable Python package.
- Add configuration, documentation, unit tests, and continuous integration.
- Keep datasets, generated predictions, experiment outputs, and model weights out of Git.

**Evidence:** repository structure, passing unit tests, and CI workflow.

## M1 — Controlled adverse-condition generation
- Implement low light, blur, fog, rain, noise, low contrast, and color shift.
- Use severity levels 1-5.
- Preserve image size and geometry so existing YOLO boxes remain valid.
- Make stochastic effects reproducible with a fixed seed.

**Evidence:** tested image transformations and label-preserving dataset preparation.

## M2 — Detection baseline and condition-wise evaluation
- Run a pretrained YOLO detector on a clean labeled split.
- Evaluate the same model on transformed versions of the same images.
- Record precision, recall, mAP@0.5, and mAP@0.5:0.95.
- Save class-wise AP where available.

**Evidence:** condition metrics CSV files and detector outputs.

## M3 — Robustness-gap analysis
- Use the clean run as the reference.
- Compute absolute and relative performance drops for each adverse condition.
- Render condition-wise metric plots.
- Run a small end-to-end smoke benchmark in GitHub Actions.

**Evidence:** robustness summary CSV, metric plot, and workflow artifacts.

## M4 — Synthetic-corruption severity benchmark (COCO128 completed)
- Completed COCO128 evaluation: 128 images, YOLO11n, seven adverse conditions, severities 1–5, plus a clean reference (36 evaluations).
- Successfully executed in [Actions run 37008467779](https://github.com/saidbaharsadat/domain-robust-object-detection/actions/runs/37008467779) on 2026-10-02 at commit `f55827508066558b7bba7996865502dbe5372d11`.
- Clean mAP@0.5:0.95 = 0.5024; severity-5 noise = 0.1657 (67.02% relative decrease), rain = 0.3513 (30.09%), blur = 0.3889 (22.60%). Low light is non-monotonic at mild severities; see the [full study](coco128_severity_study.md).
- Generated severity-wise metrics, a pivot table, two robustness curves, and metadata. The uploaded ZIP is unavailable after its 14-day retention window; full numerical metrics and metadata were recovered from the job log into `docs/results/`. Original plot PNGs were not recovered.

**Evidence:** successful sweep step, complete logged 36-row CSV, 128-image/929-instance validation summaries, metadata, and successful artifact-upload log. See [provenance and reproduction](coco128_severity_study.md).

**Remaining M4 work:** larger independent validation subsets, repeated seeds, uncertainty estimates, and stronger environment/model/dataset version pinning. This completed controlled synthetic study does not establish natural cross-domain generalization.

## Future M5 — Robust augmentation training
- Fine-tune a small detector with mixed brightness, blur, noise, and weather-style augmentation.
- Compare against the same clean-trained baseline.

## Future M6 — Image enhancement
- Test gamma correction, CLAHE, and selected denoising methods before detection.
- Measure whether preprocessing helps or harms each condition.

## Future M7 — Ablation study
- Compare baseline, augmentation, enhancement, and combined strategies.
- Keep model architecture, data split, and evaluation settings controlled.

## Future M8 — Real adverse-domain evaluation
- Add selected real low-light, weather, or camera-domain datasets.
- Separate synthetic corruption robustness from natural cross-domain generalization.

## Future M9 — Detailed failure analysis
- Organize missed detections, false positives, low-confidence detections, and localization failures.
- Compare failure patterns by class and visual condition.
