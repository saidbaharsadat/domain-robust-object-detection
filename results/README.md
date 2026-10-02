# Results

Generated predictions, transformed benchmark images, CSV metrics, plots, and model outputs are intentionally ignored by Git.

Typical outputs include:

```text
results/
├── baseline/
│   └── predictions/
├── metrics/
│   ├── condition_metrics.csv
│   ├── robustness_summary.csv
│   └── map50_95_by_condition.png
└── coco8_smoke/
    ├── condition_metrics.csv
    ├── robustness_summary.csv
    ├── map50_95_by_condition.png
    └── run_metadata.json
```

GitHub Actions uploads the smoke-benchmark outputs as temporary workflow artifacts. Research-scale results should only be documented after actual experiments have completed.
