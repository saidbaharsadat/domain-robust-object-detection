# Data

Dataset files are not committed to this repository.

For the first milestone, use a YOLO-format object-detection dataset with images, labels, and a dataset YAML file.

Suggested working layout:

```text
data/
├── clean/
│   ├── images/
│   └── labels/
└── adverse/
    ├── low_light/
    ├── blur/
    ├── fog/
    ├── rain/
    ├── noise/
    ├── low_contrast/
    └── color_shift/
```

The synthetic transformations in `src/create_adverse_conditions.py` do not change image geometry. Bounding-box labels can therefore be reused for transformed copies as long as filenames and split structure remain aligned.
