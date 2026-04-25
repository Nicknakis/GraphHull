# GraphHull: Latent Space Model (LSM) — Clustering Variant

A PyTorch implementation of an LSM that learns global archetypes and local
convex hulls per community, evaluated on a **node-clustering** task using NMI
and ARI against ground-truth labels.

> This is the *clustering* variant. A separate (related but distinct) variant
> for **link prediction** lives alongside this project — the two are kept as
> independent codebases because the training schedules, priors, evaluation
> metrics, and initialization strategies differ.

## Project Structure

```
lsm-clustering/
├── README.md
├── requirements.txt
├── main.py                      # Entry point: trains LSM and evaluates clustering
├── lsm/
│   ├── __init__.py
│   ├── model.py                 # LSM model class (with clustering methods)
│   ├── geometry.py              # Simplex intersection utilities
│   ├── seed_utils.py            # Deterministic seeding helpers
│   ├── initializers.py          # All init_* helpers (incl. init_phase1_stable)
│   ├── data.py                  # Dataset loader (edges + ground-truth labels)
│   └── spectral_clustering.py   # Place your existing Spectral_clustering_init here
└── datasets/
    └── lastfm_with_labels/
        ├── sparse_i.txt
        ├── sparse_j.txt
        └── labels.txt
```

## Setup

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   Note: `torch_sparse` requires a separate installation matching your PyTorch /
   CUDA version. See https://github.com/rusty1s/pytorch_sparse.

2. **Spectral clustering module**

   `lsm/spectral_clustering.py` is bundled. It contains `Spectral_clustering_init`
   with `Adjacency`, `Normalized_sym`, `Normalized`, and `MDS` methods.
   (One small fix vs. the original: `scipy.errstate` → `np.errstate`, which
   `scipy >= 1.x` no longer exposes.)

3. **`torch_sparse` is optional**

   It is only used inside `LSM.sample_network`, which neither the training loop
   nor the clustering evaluators ever call. The import is now lazy (inside the
   method), so you can run training without installing `torch_sparse`.

4. **Add your dataset**

   The default dataset is `lastfm_with_labels`. Place the three required text
   files under `datasets/lastfm_with_labels/`. To use a different dataset, edit
   the `dataset` variable in `main.py`.

## Usage

From the project root:

```bash
python main.py
```

The script:

- Sets all seeds for reproducibility (deterministic CUDA kernels enabled).
- Trains an `LSM` for 10,000 epochs in two phases (scaling → phase 1 → phase 2
  with anchor-dominant local hulls).
- Reports NMI and ARI against ground-truth labels every 100 epochs.

