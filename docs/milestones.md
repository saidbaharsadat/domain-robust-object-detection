# Implementation Milestones

> Current project scope: **M0-M3**. M4 and later are planned research extensions and should not be presented as completed work.

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

## Future M4 — Larger synthetic-corruption benchmark
- Replace the tiny smoke set with a larger labeled validation subset.
- Repeat multiple severity levels.
- Report condition-wise and severity-wise robustness curves.

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
