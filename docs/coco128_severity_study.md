# COCO128 multi-severity robustness study

## Verified execution and provenance

The study completed on **2026-10-02**, rather than remaining a configured future experiment.

- Source commit: [f55827508066558b7bba7996865502dbe5372d11](https://github.com/saidbaharsadat/domain-robust-object-detection/commit/f55827508066558b7bba7996865502dbe5372d11).
- Workflow: `.github/workflows/benchmark-smoke.yml`, “Adverse-condition detection benchmark”.
- [Run 37008467779](https://github.com/saidbaharsadat/domain-robust-object-detection/actions/runs/37008467779), attempt 1: success.
- [Benchmark job 110842222399](https://github.com/saidbaharsadat/domain-robust-object-detection/actions/runs/37008467779/job/110842222399): success, including “Run COCO128 multi-severity study” and artifact upload.
- The job log prints the entire `severity_metrics.csv` at 12:52:15 UTC, followed by metadata; the outputs step prints the metadata again. All 36 evaluation summaries report 128 images and 929 instances.
- Artifact `adverse-condition-detection-benchmarks`, ID **11227425998**, successfully uploaded 13 files, 342145 bytes, at 12:52:17 UTC. Logged ZIP SHA256: `cafa65c58128f9cab7fc6d17a9fd9b1f3313b89f3640a0d4335a9b8eace4a225`.
- On **2026-10-18**, the run artifact endpoint returned an empty list and downloading the known ID returned **404**. The workflow specified **14-day retention**; the unavailable ZIP is consistent with that window, but the deletion reason is not independently established.

The numerical evidence below was recovered from the surviving job log, not inferred from the workflow configuration or success badge. This verification did not rerun model inference.

## Protocol

Configuration: [coco128_severity_sweep.yaml](../configs/coco128_severity_sweep.yaml). Implementation: [run_severity_sweep.py](../tools/run_severity_sweep.py).

| Setting | Value |
| --- | --- |
| Model | Pretrained `yolo11n.pt` (YOLO11n), no fine-tuning |
| Dataset | `coco128.yaml`; 128 labeled images, 929 instances |
| Source images / labels | `images/train2017` / `labels/train2017` within COCO128 |
| Image size / device | 640 / CPU |
| Corruption seed | 42; per-image seed = 42 + sorted image index |
| Conditions | low light, Gaussian blur, fog, rain, Gaussian noise, low contrast, color shift |
| Severities | 1, 2, 3, 4, 5 |
| Evaluations | One clean reference (severity 0 sentinel) + 35 adverse observations |
| Metrics | Precision, recall, mAP@0.5, mAP@0.5:0.95, absolute and relative drops |

The same detector and underlying images/labels are used throughout. Transformations preserve geometry; transformed images are written with OpenCV using their original filename extension. Severity levels are implementation-specific controls, not physically calibrated weather or sensor measurements.

## Retained numerical evidence

- [Full metrics CSV](results/coco128_severity_metrics.csv): logged header and all 36 rows, retaining full printed precision and timestamps; only the Actions log-prefix timestamps are removed.
- [Run metadata JSON](results/coco128_severity_run_metadata.json): parsed from logged JSON, with whitespace normalized and values unchanged.
- [mAP severity pivot](results/coco128_map50_95_by_severity.csv): reconstructed from the retained CSV, not recovered from the ZIP.

The original two PNG curve files were not recovered. The reproduction command regenerates them.

Clean mAP@0.5:0.95: **0.5024436941561841**. Relative drop = **100 × (clean − adverse) / clean**; negative drops indicate measurements above the clean reference.

| Condition | S1 | S2 | S3 | S4 | S5 | S5 relative drop |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| low light | 0.5062 | 0.5082 | 0.5004 | 0.4853 | 0.4505 | 10.34% |
| blur | 0.4971 | 0.4803 | 0.4622 | 0.4434 | 0.3889 | 22.60% |
| fog | 0.4991 | 0.4924 | 0.4782 | 0.4506 | 0.4198 | 16.45% |
| rain | 0.4849 | 0.4770 | 0.4461 | 0.4127 | 0.3513 | 30.09% |
| noise | 0.5017 | 0.4507 | 0.3671 | 0.2727 | 0.1657 | 67.02% |
| low contrast | 0.5039 | 0.5001 | 0.4752 | 0.4586 | 0.4303 | 14.36% |
| color shift | 0.4963 | 0.4943 | 0.4901 | 0.4785 | 0.4660 | 7.25% |

Noise decreases from 0.5017 at severity 1 to 0.1657 at severity 5 (67.02% below clean), and is the worst condition from severity 2 onward. At severity 5, rain and blur follow with relative drops of 30.09% and 22.60%. Blur, fog, rain, noise, low contrast, and color shift decrease monotonically across severities 1–5. Low light increases slightly from 0.5062 to 0.5082 at severities 1–2, then decreases to 0.4505. Mild low light and severity-1 low contrast exceed clean mAP slightly; statistical significance is not established.

The clean and all severity-3 precision, recall, mAP@0.5, and mAP@0.5:0.95 values exactly match the [retained single-severity baseline](coco128_baseline.md). In particular, the baseline remains **clean 0.5024 → noise 0.3671, a 26.94% relative decrease**.

## Reproduction

Check out the source commit above, then follow the README installation instructions. Run:

```bash
python tools/run_severity_sweep.py --config configs/coco128_severity_sweep.yaml
```

The script writes the following under `results/coco128_severity_sweep/`:

- `severity_metrics.csv`
- `map50_95_by_severity.csv`
- `map50_95_severity_curves.png`
- `relative_drop_severity_curves.png`
- `run_metadata.json`

Observed runner environment: Ubuntu 24.04.5, Python 3.11.16, Ultralytics 8.4.171, PyTorch 2.14.1+cpu, torchvision 0.29.1+cpu, NumPy 2.4.6, OpenCV 5.0.0.93, pandas 3.0.6, Matplotlib 3.11.2, PyYAML 6.0.3. These versions are recorded from the job log, not a dependency lockfile.

Validation of the recovered evidence: 36 unique condition/severity keys; exactly one clean row and every seven-condition × five-severity pair; all four absolute/relative metric-drop calculations agree; clean and severity-3 metrics exactly match committed baseline values. Logged metadata agrees with the worst observation and severity-5 ranking. The historical workflow also reports six passing unit tests.

## Interpretation and remaining research

This is a completed **small controlled synthetic-corruption robustness study**. COCO128 is a small COCO training-image subset used for evaluation here; it is not an independent held-out real adverse-domain test. A pretrained COCO detector may already have seen these images. Absolute mAP and small improvements must be interpreted accordingly.

The study does not demonstrate real-world domain generalization, robustness-training gains, or statistical reliability across seeds. No model-weight/dataset checksums, complete environment lock, or repeat-run uncertainty estimates were retained; the configured seed controls corruption generation rather than establishing bitwise inference reproducibility.

Remaining tasks: larger independent validation sets; repeated stochastic seeds and uncertainty estimates; qualitative and class-wise failure analysis; augmentation training; enhancement and controlled ablations; and real low-light/weather/camera-domain evaluation. Pin dependency versions and model/dataset checksums for stronger future reproducibility, and preserve result reports before transient artifact retention expires.
