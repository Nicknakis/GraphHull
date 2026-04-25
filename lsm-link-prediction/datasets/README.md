# Datasets

Each dataset lives in its own subdirectory: `datasets/<dataset_name>/`.

The default dataset name used by `main.py` is `hepth`, so you would create:

```
datasets/
└── hepth/
    ├── sparse_i.txt        # Training edge endpoints (i side)
    ├── sparse_j.txt        # Training edge endpoints (j side)
    ├── sparse_i_rem.txt    # Held-out positive edges (i side)
    ├── sparse_j_rem.txt    # Held-out positive edges (j side)
    ├── non_sparse_i.txt    # Held-out non-edges (i side)
    └── non_sparse_j.txt    # Held-out non-edges (j side)
```

Each `.txt` file is loaded with `numpy.loadtxt` and should contain one integer node
index per line (or whitespace-separated), matching the format expected by
`numpy.loadtxt(...).long()`.

To use a different dataset name, change the `dataset` variable in `main.py`.
