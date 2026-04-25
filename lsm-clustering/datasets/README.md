# Datasets (Clustering Variant)

Each dataset lives in its own subdirectory: `datasets/<dataset_name>/`.

The default dataset name used by `main.py` is `lastfm_with_labels`, so you would
create:

```
datasets/
└── lastfm_with_labels/
    ├── sparse_i.txt   # Edge endpoints (i side)
    ├── sparse_j.txt   # Edge endpoints (j side)
    └── labels.txt     # Ground-truth community labels (one integer per line per node)
```

All files are loaded with `numpy.loadtxt` (whitespace-separated integers).

To use a different dataset, edit the `dataset` variable in `main.py`.
