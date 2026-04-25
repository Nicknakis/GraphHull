# GraphHull

PyTorch implementation of **Archetypal Graph Generative Models: Explainable and Identifiable Communities via Anchor-Dominant Convex Hulls**, a Spotlight paper at the 29th International Conference on Artificial Intelligence and Statistics (**AISTATS 2026**), Tangier, Morocco. PMLR: Volume 300.

> Copyright © 2026 by the author(s).

## Description

Representation learning has been essential for graph machine learning tasks such as link prediction, community detection, and network visualization. Despite recent advances in achieving high performance on these downstream tasks, little progress has been made toward self-explainable models. Understanding the patterns behind predictions is equally important, motivating recent interest in explainable machine learning.

In this paper, we present **GraphHull**, an explainable generative model that represents networks using two levels of convex hulls:

- **Global level** — the vertices of a convex hull are treated as *archetypes*, each corresponding to a pure community in the network.
- **Local level** — each community is refined by a prototypical hull whose vertices act as representative profiles, capturing community-specific variation.

This two-level construction yields clear multi-scale explanations: a node's position relative to global archetypes and its local prototypes directly accounts for its edges. The geometry is well-behaved by design, while local hulls are kept disjoint by construction. To further encourage diversity and stability, we place principled priors — including determinantal point processes — and fit the model under MAP estimation with scalable subsampling.

Experiments on real networks demonstrate the ability of GraphHull to recover multi-level community structure and to achieve competitive or superior performance in link prediction and community detection, while naturally providing interpretable predictions.

## Repository structure

GraphHull is shipped as two self-contained projects, one per downstream task:

```
.
├── lsm-link-prediction/   # Held-out edge / non-edge prediction (AUC-ROC, AUC-PR)
└── lsm-clustering/        # Node clustering vs. ground-truth labels (NMI, ARI)
```

Each subfolder contains its own `README.md`, `requirements.txt`, `main.py`, `lsm/` package, and `lsm/spectral_clustering.py`. They are kept apart because the training schedules, priors, evaluation metrics, and initialization strategies differ.

| | `lsm-link-prediction` | `lsm-clustering` |
|---|---|---|
| **Evaluation** | AUC-ROC / AUC-PR | NMI / ARI |
| **Determinism** | Not enforced | Full deterministic seeding |
| **`A_svd_boxed`** | Used with `K = D` in experiments | Used with `K = D` in experiments |
| **`σ` box** | `[0.5, 2.5]` | `[0.1, 1.5]` |
| **Phase transitions** | scaling→phase1 @ 1000, phase1→phase2 @ 2000 | scaling→phase1 @ 200, phase1→phase2 @ 2000 |
| **LR schedule** | drops to 0.01 @ 5000 | drops to 0.01 @ 2000, then 0.001 @ 4000 |
| **Grad clip** | 5.0 | 2.0 |
| **Initialization** | `init_biases_from_degrees` + spectral + centroid SVD | `init_phase1_stable` (canonicalize spectral, farthest-first anchors, entropy-calibrated logits, density-calibrated `b0`) |
| **Post-training** | Extensive PCA / hull / circular / adjacency plots | Training-only |

## Requirements

- Python 3.8.3 (developed and tested), PyTorch 1.12.1
- See each subproject's `requirements.txt` for the full list

`torch_sparse` is **optional**: it is only used by `LSM.sample_network()`, which neither training loop calls. The import is lazy, so you can run the model without it.

## Quick start

```bash
git clone https://github.com/<your-username>/GraphHull.git
cd GraphHull/lsm-link-prediction        # or: cd GraphHull/lsm-clustering
pip install -r requirements.txt
python main.py
```

Cora is bundled in both projects under `datasets/cora/`. To use it, change the `dataset` variable at the top of the `for psi in psis:` block in `main.py` to `'cora'`. To use your own data, drop a directory under `datasets/<your-name>/` with the expected files (see each project's `datasets/README.md`).

## Smoke-test status

Both projects were tested end-to-end on Cora (2,485 nodes, 7 classes, ~5k edges) for 2,200 epochs each, exercising all three training phases (scaling → phase 1 → phase 2):

| Project | Result | Sample numbers |
|---|---|---|
| `lsm-link-prediction` | All phases run cleanly through epoch 2,200 | (Below convergence at 2,200 epochs) |
| `lsm-clustering` | All phases run cleanly through epoch 2,200 | NMI ≈ 0.43, ARI ≈ 0.35 at epoch ~1,800 |

## Reference

If you use GraphHull, please cite:

> Nikolaos Nakis, Chrysoula Kosma, Panagiotis Promponas, Michail Chatzianastasis, and Giannis Nikolentzos. **Archetypal Graph Generative Models: Explainable and Identifiable Communities via Anchor-Dominant Convex Hulls.** *AISTATS 2026 (Spotlight Paper).* [arXiv:2602.21342](https://arxiv.org/abs/2602.21342)

```bibtex
@inproceedings{nakis2026graphhull,
  title     = {Archetypal Graph Generative Models: Explainable and Identifiable Communities via Anchor-Dominant Convex Hulls},
  author    = {Nakis, Nikolaos and Kosma, Chrysoula and Promponas, Panagiotis and Chatzianastasis, Michail and Nikolentzos, Giannis},
  booktitle = {Proceedings of the 29th International Conference on Artificial Intelligence and Statistics (AISTATS)},
  series    = {Proceedings of Machine Learning Research},
  volume    = {300},
  year      = {2026},
  publisher = {PMLR},
  note      = {Spotlight Paper}
}
```
