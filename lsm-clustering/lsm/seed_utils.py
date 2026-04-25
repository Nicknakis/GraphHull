"""Deterministic seeding utilities (matches the original script)."""

import os
import random

import numpy as np
import torch


def set_seed(seed=9):
    """
    Set all relevant random seeds and configure deterministic behavior.

    Mirrors the original block:
        SEED = 9
        os.environ["PYTHONHASHSEED"] = str(SEED)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
        random.seed(SEED); np.random.seed(SEED)
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark    = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"   # CUDA determinism
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
