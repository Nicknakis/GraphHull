"""
Geometric utilities: simplex intersection (LP), convex-hull intersection (LP),
and minimum distance between convex hulls (QP).
"""

import numpy as np
from itertools import combinations
from scipy.optimize import linprog
from cvxopt import matrix, solvers

solvers.options['show_progress'] = False
solvers.options.update({'abstol': 1e-9, 'reltol': 1e-9, 'feastol': 1e-9, 'maxiters': 100})


def _cvx(x):
    return None if x is None else matrix(x)


# -----------------------------------------------------------------------------
# Simplex intersection (square D x D simplices in R^D)
# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
# Convex hull intersection via LP (handles non-square / general simplex sizes)
# -----------------------------------------------------------------------------
def hulls_intersect_lp(V, W, tol=1e-7, svd_tol=1e-10):
    """
    Returns True iff conv(V) ∩ conv(W) ≠ ∅ (within tol).
    V: [Lv, D], W: [Lw, D]. Lv, Lw >= 0. D can be 0.
    """
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)

    Lv, Dv = V.shape if V.ndim == 2 else (0, 0)
    Lw, Dw = W.shape if W.ndim == 2 else (0, 0)
    if Lv == 0 or Lw == 0:
        return False
    if Dv != Dw:
        raise ValueError("V and W must have same dimensionality")
    D = Dv

    # Both single points → simple distance check
    if Lv == 1 and Lw == 1:
        return np.linalg.norm(V[0] - W[0]) <= tol

    # Choose last vertices as anchors (works even if Lv==1 or Lw==1;
    # the corresponding block just has zero columns).
    vL = V[Lv - 1]
    wL = W[Lw - 1]

    # Build equality system:  V^T alpha = W^T beta  with alpha_Lv = 1 - sum(a), beta_Lw = 1 - sum(b)
    # M z = rhs, where z = [a (Lv-1); b (Lw-1)]
    A_V = V[:Lv - 1].T - vL[:, None]    # (D, Lv-1)
    A_W = W[:Lw - 1].T - wL[:, None]    # (D, Lw-1)
    M = np.hstack([A_V, -A_W])          # (D, (Lv-1)+(Lw-1))
    rhs = (wL - vL)                     # (D,)
    n = M.shape[1]

    # Compress to full row rank and check consistency of Mz=rhs
    if D > 0 and n > 0:
        U, s, _ = np.linalg.svd(M, full_matrices=False)  # U: (D, m), m=min(D,n)
        # Scale-aware cutoff
        svd_eps = max(svd_tol, (s.max() if s.size else 0.0) * 1e-12)
        r = int((s > svd_eps).sum())
        m = U.shape[1]

        # If rhs has a component orthogonal to col(M), infeasible → no overlap
        if r < m:
            rhs_perp = U[:, r:m].T @ rhs
            if np.linalg.norm(rhs_perp) > tol:
                return False

        if r > 0:
            U_rT = U[:, :r].T
            Aeq = U_rT @ M            # (r, n)
            beq = U_rT @ rhs          # (r,)
        else:
            Aeq = np.zeros((0, n))
            beq = np.zeros((0,))
    else:
        # No equalities (degenerate); feasibility reduces to inequalities below.
        Aeq = np.zeros((0, n))
        beq = np.zeros((0,))

    # Inequalities: z >= 0; sum(a) <= 1; sum(b) <= 1
    G = np.zeros((n + 2, n), dtype=np.float64)
    h = np.zeros((n + 2,), dtype=np.float64)
    if n > 0:
        G[:n, :n] = -np.eye(n)         # z >= 0  ->  -I z <= 0
    if Lv > 1:
        G[n, :Lv - 1] = 1.0
        h[n] = 1.0
    if Lw > 1:
        G[n + 1, Lv - 1:] = 1.0
        h[n + 1] = 1.0

    c = np.zeros((n,), dtype=np.float64)  # dummy objective for feasibility

    # Solve LP
    A_mat = _cvx(Aeq) if Aeq.shape[0] > 0 else None
    b_mat = _cvx(beq) if beq.shape[0] > 0 else None
    res = solvers.lp(_cvx(c), _cvx(G), _cvx(h), A=A_mat, b=b_mat)

    status = res.get('status', '')
    if status == 'primal infeasible':
        return False
    if status != 'optimal':
        # Inconclusive or numerical issues: be conservative (no overlap)
        return False

    # Reconstruct convex weights
    z = np.array(res['x']).reshape(-1) if n > 0 else np.zeros((0,))
    a = z[:max(Lv - 1, 0)]
    b = z[max(Lv - 1, 0):]

    alpha = np.concatenate([a, [1.0 - a.sum()]]) if Lv > 0 else np.zeros((0,))
    beta = np.concatenate([b, [1.0 - b.sum()]]) if Lw > 0 else np.zeros((0,))

    # Numerical hygiene: small negatives → 0, then renormalize
    if alpha.min() < -tol or beta.min() < -tol:
        return False
    alpha = np.maximum(alpha, 0.0)
    beta = np.maximum(beta, 0.0)
    sa, sb = alpha.sum(), beta.sum()
    if sa <= tol or sb <= tol:
        return False
    alpha /= sa
    beta /= sb

    # Final geometric check in original space
    p = V.T @ alpha
    q = W.T @ beta
    return np.linalg.norm(p - q) <= tol


# -----------------------------------------------------------------------------
# Minimum distance between two convex hulls via QP
# -----------------------------------------------------------------------------
def hull_min_distance_qp(V, W, tol=1e-7, ridge=1e-10):
    """
    Min distance between conv(V) and conv(W).
    V: [L1, D], W: [L2, D]  (numpy float64)
    Returns a nonnegative float (0 means intersect/touch up to tol).
    """
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    L1, D1 = V.shape
    L2, D2 = W.shape
    assert D1 == D2, "V and W must have same dim"

    VV = V @ V.T            # [L1, L1]
    WW = W @ W.T            # [L2, L2]
    VW = V @ W.T            # [L1, L2]

    # x = [alpha; beta] ∈ R^{L1+L2}, objective 0.5 * x^T P x
    P = np.block([[VV, -VW],
                  [-VW.T, WW]]).astype(float)
    # Symmetrize and ridge for PSD
    P = 0.5 * (P + P.T)
    P += ridge * np.eye(L1 + L2)

    q = np.zeros(L1 + L2)

    # Simplex constraints: alpha >= 0, sum alpha = 1 ; beta >= 0, sum beta = 1
    G = -np.eye(L1 + L2)
    h = np.zeros(L1 + L2)

    Aeq = np.zeros((2, L1 + L2))
    Aeq[0, :L1] = 1.0
    Aeq[1, L1:] = 1.0
    beq = np.ones(2)

    # Solve QP
    res = solvers.qp(matrix(P), matrix(q),
                     matrix(G), matrix(h),
                     matrix(Aeq), matrix(beq))
    if res['status'] != 'optimal':
        return np.nan

    x = np.array(res['x']).reshape(-1)
    alpha = x[:L1]
    beta = x[L1:]

    # Project to simplices defensively
    alpha = np.clip(alpha, 0, None)
    s = alpha.sum()
    alpha = alpha / s if s > 0 else np.full(L1, 1.0 / L1)
    beta = np.clip(beta, 0, None)
    s = beta.sum()
    beta = beta / s if s > 0 else np.full(L2, 1.0 / L2)

    # Witness points and distance
    p = V.T @ alpha   # [D]
    q = W.T @ beta    # [D]
    d = float(np.linalg.norm(p - q))
    return 0.0 if d <= tol else d
