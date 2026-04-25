"""Geometric utilities: simplex intersection via LP."""

import numpy as np
from itertools import combinations
from scipy.optimize import linprog


def simplex_intersection(P, Q, tol=1e-9):
    """
    P, Q: (D, D) arrays. Each row is a vertex in R^D.
    Returns (intersects, x, alpha, beta)
      - intersects: bool
      - x: intersection point in R^D (None if disjoint)
      - alpha, beta: barycentric weights over rows of P and Q, respectively
    """
    P = np.asarray(P, dtype=float)
    Q = np.asarray(Q, dtype=float)
    assert P.ndim == 2 and Q.ndim == 2 and P.shape == Q.shape and P.shape[0] == P.shape[1], \
        "Each simplex must be a (D, D) matrix: D vertices in R^D."

    D = P.shape[0]
    # Variables z = [alpha (D), beta (D)]
    Aeq = np.zeros((D + 2, 2 * D))
    beq = np.zeros(D + 2)
    # Vector equality: P^T alpha - Q^T beta = 0  (D rows)
    Aeq[:D, :D] = P.T
    Aeq[:D, D:] = -Q.T
    # Sum-to-one constraints
    Aeq[D, :D] = 1.0
    beq[D] = 1.0
    Aeq[D + 1, D:] = 1.0
    beq[D + 1] = 1.0

    res = linprog(c=np.zeros(2 * D), A_eq=Aeq, b_eq=beq,
                  bounds=[(0, None)] * (2 * D), method="highs")

    if res.status != 0:
        return False, None, None, None

    z = res.x
    alpha, beta = z[:D], z[D:]
    x = P.T @ alpha  # = Q.T @ beta
    # Tighten with a small numerical check
    if (np.all(alpha >= -tol) and np.all(beta >= -tol)
            and abs(alpha.sum() - 1) <= 1e-7
            and abs(beta.sum() - 1) <= 1e-7
            and np.linalg.norm((P.T @ alpha) - (Q.T @ beta), ord=np.inf) <= 1e-7):
        return True, x, alpha, beta
    return False, None, None, None


def overlapping_pairs(polys):
    """
    polys: list/array of K simplices, each shape (D, D)
    Returns:
      - pairs: list of (i, j) indices that intersect
      - witnesses: dict[(i,j)] = {'x': point, 'alpha': alpha, 'beta': beta}
    """
    pairs = []
    witnesses = {}
    for i, j in combinations(range(len(polys)), 2):
        ok, x, a, b = simplex_intersection(polys[i], polys[j])
        if ok:
            pairs.append((i, j))
            witnesses[(i, j)] = {'x': x, 'alpha': a, 'beta': b}

    return pairs
