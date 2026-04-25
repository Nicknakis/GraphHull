"""Dataset loading utilities."""

import numpy as np
import torch


def load_dataset(dataset, device, base_dir="./datasets"):
    """
    Load a dataset stored as six text files under `<base_dir>/<dataset>/`:
        sparse_i.txt, sparse_j.txt          (training edges)
        sparse_i_rem.txt, sparse_j_rem.txt  (held-out positive edges)
        non_sparse_i.txt, non_sparse_j.txt  (held-out negative pairs)

    Returns
    -------
    sparse_i, sparse_j : LongTensors of training edge endpoints (symmetrized).
    rem_i, rem_j       : LongTensors of held-out edge endpoints (positives + negatives).
    target             : FloatTensor with 1 for held-out positives, 0 for negatives.
    N                  : int, number of nodes.
    """
    sparse_i_ = torch.from_numpy(np.loadtxt(f'{base_dir}/{dataset}/sparse_i.txt')).long().to(device)
    sparse_j_ = torch.from_numpy(np.loadtxt(f'{base_dir}/{dataset}/sparse_j.txt')).long().to(device)

    sparse_i = torch.cat((sparse_i_, sparse_j_))
    sparse_j = torch.cat((sparse_j_, sparse_i_))

    sparse_i_rem = torch.from_numpy(np.loadtxt(f'{base_dir}/{dataset}/sparse_i_rem.txt')).long().to(device)
    sparse_j_rem = torch.from_numpy(np.loadtxt(f'{base_dir}/{dataset}/sparse_j_rem.txt')).long().to(device)

    non_sparse_i_rem = torch.from_numpy(np.loadtxt(f'{base_dir}/{dataset}/non_sparse_i.txt')).long().to(device)
    non_sparse_j_rem = torch.from_numpy(np.loadtxt(f'{base_dir}/{dataset}/non_sparse_j.txt')).long().to(device)

    rem_i = torch.cat((sparse_i_rem, non_sparse_i_rem))
    rem_j = torch.cat((sparse_j_rem, non_sparse_j_rem))

    target = torch.cat((torch.ones(sparse_i_rem.shape[0]), torch.zeros(non_sparse_i_rem.shape[0])))

    N = int(max(sparse_i.max(), sparse_j.max()) + 1)

    return sparse_i, sparse_j, rem_i, rem_j, target, N
