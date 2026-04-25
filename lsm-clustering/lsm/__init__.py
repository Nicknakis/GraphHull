"""LSM (clustering variant): Latent Space Model with anchor-dominant local convex hulls."""

from lsm.model import LSM
from lsm.geometry import simplex_intersection, overlapping_pairs

__all__ = [
    "LSM",
    "simplex_intersection",
    "overlapping_pairs",
]
