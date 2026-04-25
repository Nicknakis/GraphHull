"""Dataset loading utilities for the clustering variant."""

import numpy as np
import torch


def load_dataset(dataset, device, base_dir="./datasets"):
    """
    Load a clustering dataset stored under `<base_dir>/<dataset>/`:
        sparse_i.txt, sparse_j.txt   (training edges)
        labels.txt                   (ground-truth community labels)

    Returns
    -------
    sparse_i, sparse_j : LongTensors of edge endpoints (symmetrized).
    labels_            : 1-D numpy int array of ground-truth labels.
    k_labs             : int, number of distinct ground-truth communities.
    N                  : int, number of nodes.
    """
    sparse_i_ = torch.from_numpy(np.loadtxt(f'{base_dir}/{dataset}/sparse_i.txt')).long().to(device)
    sparse_j_ = torch.from_numpy(np.loadtxt(f'{base_dir}/{dataset}/sparse_j.txt')).long().to(device)

    sparse_i = torch.cat((sparse_i_, sparse_j_))
    sparse_j = torch.cat((sparse_j_, sparse_i_))

    labels_ = np.loadtxt(f'{base_dir}/{dataset}/labels.txt').astype(int)
    k_labs = int(max(labels_) + 1)

    N = int(max(sparse_i.max(), sparse_j.max()) + 1)

    return sparse_i, sparse_j, labels_, k_labs, N
