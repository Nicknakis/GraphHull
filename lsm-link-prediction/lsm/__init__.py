"""LSM: Latent Space Model with anchor-dominant local convex hulls."""

from lsm.model import LSM
from lsm.ema import EMA
from lsm.geometry import (
    simplex_intersection,
    overlapping_pairs,
    hulls_intersect_lp,
    hull_min_distance_qp,
)

__all__ = [
    "LSM",
    "EMA",
    "simplex_intersection",
    "overlapping_pairs",
    "hulls_intersect_lp",
    "hull_min_distance_qp",
]
