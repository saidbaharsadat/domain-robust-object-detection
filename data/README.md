# Data

Large datasets are intentionally excluded from Git.

The current reproducibility workflow uses Ultralytics COCO8 only as a tiny smoke benchmark. The benchmark runner resolves it through Ultralytics and generates adverse validation copies under the ignored results directory.

For local research experiments, a typical YOLO-format layout is:

```text
data/
└── dataset_name/
    ├── images/
    │   ├── train/
    │   └── val/
    ├── labels/
    │   ├── train/
    │   └── val/
    └── data.yaml
```

Synthetic transformations in this repository preserve image geometry, so YOLO bounding-box labels can be copied unchanged when filenames and split structure remain aligned.
