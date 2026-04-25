# Latent Space Model (LSM) with Anchor-Dominant Convex Hulls

A PyTorch implementation of a Latent Space Model for graph link prediction that learns
global archetypes and local convex hulls per community.

## Project Structure

```
lsm-project/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── main.py                      # Entry point: runs full training + visualization
├── lsm/                         # Core package
│   ├── __init__.py
│   ├── model.py                 # LSM model class
│   ├── geometry.py              # Simplex / convex-hull intersection utilities
│   ├── ema.py                   # Exponential Moving Average helper
│   ├── initializers.py          # Parameter initialization utilities
│   ├── data.py                  # Dataset loading
│   ├── visualization.py         # Plotting utilities
│   └── spectral_clustering.py   # (place your existing Spectral_clustering_init here)
└── datasets/
    └── hepth/                   # Example dataset directory
        ├── sparse_i.txt
        ├── sparse_j.txt
        ├── sparse_i_rem.txt
        ├── sparse_j_rem.txt
        ├── non_sparse_i.txt
        └── non_sparse_j.txt
```

## Setup

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   Note: `torch_sparse` requires a separate installation matching your PyTorch / CUDA
   version. See the [official instructions](https://github.com/rusty1s/pytorch_sparse).

2. **Spectral clustering module**

   `lsm/spectral_clustering.py` is bundled. It contains `Spectral_clustering_init`
   with `Adjacency`, `Normalized_sym`, `Normalized`, and `MDS` methods.
   (One small fix vs. the original: `scipy.errstate` → `np.errstate`, which
   `scipy >= 1.x` no longer exposes.)

3. **`torch_sparse` is optional**

   It is only used inside `LSM.sample_network`, which neither the training loop nor
   `link_prediction` ever calls. The import is now lazy (inside the method), so
   you can run training without installing `torch_sparse`.

4. **Add datasets**

   Each dataset directory under `datasets/` should contain six `.txt` files describing
   the training edges, held-out edges, and held-out non-edges. The default dataset name
   is `hepth`.

## Usage

Run the full pipeline (training + link prediction evaluation + plots) from the project
root:

```bash
python main.py
```

To switch datasets, change the `dataset` variable inside `main.py`, or pass it through
your own runner.

## What the code does

- Trains the `LSM` model in two phases (scaling phase, then phase 1, then a richer
  representation with anchor-dominant local hulls).
- Reports AUC-ROC and AUC-PR on held-out links every 100 epochs.
- After training, generates:
  - PCA projections of the latent space, archetypes, and local hulls.
  - Reordered adjacency spy plots with community block outlines.
  - Per-community circular plots.
  - Pairwise hull intersection / minimum-distance diagnostics.

## Notes

No calculations, libraries, or numerical behavior were changed during refactoring —
only file organization. The `EMA` class is provided but not invoked in the training
loop (matching the original script).
