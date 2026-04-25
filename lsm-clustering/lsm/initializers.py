"""
Parameter initialization utilities for the clustering-variant LSM.

Includes the full `init_phase1_stable` pipeline plus standalone helpers:
deterministic spectral canonicalization, farthest-first anchor selection,
entropy-calibrated logits, and a bisection-based density calibration for `b0`.
"""

import math

import torch
import torch.nn.functional as F


@torch.no_grad()
def init_biases_from_degrees(model, sparse_i, sparse_j, undirected=True, eps=1.0):
    N = model.input_size
    deg = torch.zeros(N, device=sparse_i.device)
    deg.index_add_(0, sparse_i, torch.ones_like(sparse_i, dtype=deg.dtype))
    deg.index_add_(0, sparse_j, torch.ones_like(sparse_j, dtype=deg.dtype))
    if undirected:
        deg = deg / 2  # your edges are doubled (i->j and j->i)
    p = (deg + eps) / (N - 1 + 2 * eps)
    logit = torch.log(p) - torch.log1p(-p)
    model.g.data.copy_(logit - logit.mean())


@torch.no_grad()
def init_B_from_assignments(model, sparse_i, sparse_j):
    K = model.K
    m = torch.softmax(model.latent_z1, dim=1)
    c = m.argmax(1)
    # counts of edges between communities (undirected: symmetrize)
    C = torch.zeros(K, K, device=c.device)
    ci, cj = c[sparse_i], c[sparse_j]
    for a in range(K):
        for b in range(K):
            C[a, b] = ((ci == a) & (cj == b)).sum()
    C = 0.5 * (C + C.T)

    # expected pairs if random pairing inside sample
    n = torch.bincount(c, minlength=K).float()
    P = n[:, None] * n[None, :]
    P[torch.arange(K), torch.arange(K)] = n * (n - 1)

    eps = 1.0
    M = torch.log((C + eps) / (P - C + eps))  # coarse log-odds
    # solve M ≈ b 1^T + 1 b^T in least squares → b = (M 1)/K - mean(M)
    one = torch.ones(K, device=C.device)
    b = (M @ one) / K
    b = b - b.mean()
    model.B_free.data.copy_(b)


@torch.no_grad()
def init_memberships_from_spectral(model, scale=6.0):
    # spectral_data is already N x K; amplify differences to get near one-hot logits
    Zs = model.spectral_data  # [N,K]
    # normalize columns to comparable scale
    Zs = (Zs - Zs.mean(0, keepdim=True)) / (Zs.std(0, keepdim=True).clamp_min(1e-6))
    # sharpen logits
    model.latent_z1.data.copy_(scale * Zs)


@torch.no_grad()
def init_A_from_centroids(model):
    Z0 = torch.softmax(model.latent_z1, dim=1)  # [N,K]
    C = (Z0.T @ Z0) + 1e-6 * torch.eye(model.K, device=Z0.device)
    U, _, Vt = torch.linalg.svd(C)
    # map to boxed singulars
    s = 0.3 + (2.0 - 0.3) * torch.sigmoid(model.sig_free)
    A0 = (U * s.unsqueeze(0)) @ Vt
    model.Hu.data.copy_(U)
    model.Hv.data.copy_(Vt.T)
    # Put some structure: encourage near-identity early
    model.sig_free.data.copy_(torch.logit((s - 0.3) / (2.0 - 0.3)))


@torch.no_grad()
def init_A_by_least_squares(model):
    # Target “coordinates” from top-K spectral components (recentered)
    Z0 = torch.softmax(model.latent_z1, dim=1)             # [N,K]
    X = model.spectral_data                                 # [N,K] already K comps
    X = X - X.mean(0, keepdim=True)

    # Solve min_A || Z0 A - X ||_F ; A = argmin least squares
    # torch.linalg.lstsq is stable (PyTorch 1.9+)
    A_ls = torch.linalg.lstsq(Z0, X).solution              # [K,K]

    # Feed through your boxed-SVD parametrization to keep constraints
    Qu, _ = torch.linalg.qr(A_ls, mode='reduced')
    Qv, _ = torch.linalg.qr(A_ls.T, mode='reduced')
    model.Hu.data.copy_(Qu)
    model.Hv.data.copy_(Qv)
    # Choose singulars close to A_ls but within box
    s = torch.diag(Qu.T @ A_ls @ Qv)                       # crude diag capture
    s = s.clamp(0.3, 2.0)
    # Map s back to sig_free
    sigma_min, sigma_max = 0.3, 2.0
    sig = (s - sigma_min) / (sigma_max - sigma_min)
    sig = sig.clamp(1e-3, 1 - 1e-3)
    model.sig_free.data.copy_(torch.logit(sig))


@torch.no_grad()
def init_anchor_dominant_rows(model, target_s=0.05):
    # s = eps * sigmoid(t) -> choose t to make s ≈ target_s
    # eps = model.anchor_eps (e.g., 0.49). Solve for t:
    # sigmoid(t) = target_s / eps
    eps = float(model.anchor_eps)
    t = torch.logit(torch.tensor(min(0.99, max(0.01, target_s / eps)),
                                 device=model.local_shrink.device))
    model.local_shrink.data.fill_(float(t))   # small initial non-anchor mass


@torch.no_grad()
def init_biases_from_degrees_shrunk(model, sparse_i, sparse_j, undirected=True, alpha=4.0):
    """
    EB shrinkage:  p_i = (deg_i + alpha * p0) / ((N-1) + alpha)
    g_i = logit(p_i) - mean(logit(p))
    alpha controls shrinkage toward global density (alpha~2-8 works well).
    """
    N = model.input_size
    dev = sparse_i.device
    deg = torch.zeros(N, device=dev, dtype=torch.float)
    deg.index_add_(0, sparse_i, torch.ones_like(sparse_i, dtype=deg.dtype))
    deg.index_add_(0, sparse_j, torch.ones_like(sparse_j, dtype=deg.dtype))
    if undirected:
        deg = deg / 2.0  # if edges are doubled

    p0 = (deg.sum() / (N * (N - 1))).clamp(1e-8, 1 - 1e-8)
    p_i = (deg + alpha * p0) / ((N - 1) + alpha)
    p_i = p_i.clamp(1e-8, 1 - 1e-8)
    logit = torch.log(p_i) - torch.log1p(-p_i)
    model.g.data.copy_(logit - logit.mean())


@torch.no_grad()
def init_memberships_from_spectral_kpp_margin(model, subN=40000, scale=6.0, seed=0):
    """
    Build latent_z1 (logits) from distances to K k-means++ centers in spectral space.
    Produces near one-hot softmax with margin (controlled by 'scale').
    """
    torch.manual_seed(seed)
    Zs = model.spectral_data.clone()  # [N,K]
    # whiten per column for balanced geometry
    Zs = (Zs - Zs.mean(0, keepdim=True)) / (Zs.std(0, keepdim=True).clamp_min(1e-6))

    N, K = Zs.shape
    dev = Zs.device

    # pick a subset for center seeding (memory friendly)
    idx = torch.randperm(N, device=dev)[:min(subN, N)]
    Xsub = Zs[idx]  # [M,K]

    # k-means++ on subset
    centers = []
    c0 = torch.argmax((Xsub ** 2).sum(dim=1)).item()
    centers.append(Xsub[c0:c0 + 1])
    d2 = torch.cdist(Xsub, centers[0]) ** 2  # [M,1]
    for _ in range(1, K):
        probs = (d2.min(dim=1).values + 1e-12)
        probs = probs / probs.sum()
        new_idx = torch.multinomial(probs, 1).item()
        centers.append(Xsub[new_idx:new_idx + 1])
        d2 = torch.minimum(d2, torch.cdist(Xsub, centers[-1]) ** 2)
    C = torch.cat(centers, dim=0)  # [K,K]

    # distances of all nodes to centers
    D = torch.cdist(Zs, C)  # [N,K]
    # convert to logits: closer -> larger logit
    # normalize D by robust scale to make 'scale' portable across graphs
    rob = torch.quantile(D, 0.9).clamp_min(1e-6)
    logits = - D / (rob / scale)
    model.latent_z1.data.copy_(logits)


@torch.no_grad()
def calibrate_b0_to_density(model, samples=200_000):
    N = model.input_size
    dev = model.g.device
    # observed density (doubled undirected)
    p_obs = model.sparse_i_idx.numel() / float(N * (N - 1))
    # sample pairs i != j
    i = torch.randint(0, N, (samples,), device=dev)
    j = torch.randint(0, N - 1, (samples,), device=dev)
    j = j + (j >= i)
    gi_gj = (model.g[i] + model.g[j]).detach()
    lo, hi = -10.0, 10.0          # bisection; monotone in b0
    for _ in range(35):
        mid = 0.5 * (lo + hi)
        m = torch.sigmoid(mid + gi_gj).mean().item()
        if m > p_obs:
            hi = mid
        else:
            lo = mid
    model.b0.data.fill_(0.5 * (lo + hi))


@torch.no_grad()
def canonicalize_spectral(X):
    """
    X: [N,K] spectral embedding (float tensor).
    Steps:
      - z-score columns (columnwise mean/std)
      - fix column signs so sum >= 0
      - order columns by descending variance
      - L2-normalize rows (optional but helps distances)
    Returns X_can: [N,K]
    """
    X = X.clone()
    mu = X.mean(0, keepdim=True)
    sd = X.std(0, keepdim=True).clamp_min(1e-6)
    X = (X - mu) / sd
    # fix signs
    sgn = torch.sign(X.sum(0, keepdim=True)).clamp(min=-1, max=1)
    sgn[sgn == 0] = 1
    X = X * sgn
    # order columns by variance
    var = X.var(0, unbiased=False)
    order = torch.argsort(var, descending=True)
    X = X[:, order]
    # row normalize (keeps distances numerically stable)
    X = F.normalize(X, dim=1)
    return X


@torch.no_grad()
def farthest_first_anchors(X, K):
    """
    Deterministic farthest-first anchors in the space of X (rows are nodes).
    Returns indices of K anchors.
    """
    # start with the farthest from the mean
    c = X.mean(0, keepdim=True)               # [1,K]
    d2 = ((X - c) ** 2).sum(1)                # [N]
    idx0 = torch.argmax(d2).item()
    anchors = [idx0]
    # iteratively add farthest from current set
    # maintain min distance to any selected anchor
    min_d2 = ((X - X[idx0]) ** 2).sum(1)      # [N]
    for _ in range(1, K):
        # exclude already picked
        min_d2[anchors] = -1.0
        nxt = torch.argmax(min_d2).item()
        anchors.append(nxt)
        # update min distances
        d2_new = ((X - X[nxt]) ** 2).sum(1)
        min_d2 = torch.minimum(min_d2, d2_new)
    return torch.tensor(anchors, device=X.device, dtype=torch.long)


@torch.no_grad()
def logits_from_anchors(X, anchor_idx, target_entropy_frac=0.60):
    """
    Build logits L_{i,a} = -||X_i - A_a||^2 / tau with tau chosen
    so that mean softmax entropy is target_entropy_frac * log(K).
    Deterministic bisection over tau.
    """
    A = X[anchor_idx]                           # [K,K] (same dim as columns)
    D2 = torch.cdist(X, A).pow(2)               # [N,K]
    K = D2.size(1)
    # choose tau by entropy target
    Ht = target_entropy_frac * math.log(K)
    lo, hi = 1e-3, 100.0
    for _ in range(35):
        mid = 0.5 * (lo + hi)
        L = - D2 / mid
        P = F.softmax(L, dim=1)
        H = -(P.clamp_min(1e-9).log() * P).sum(1).mean()
        if H > Ht:       # too diffuse -> decrease tau (sharpen)
            hi = mid
        else:            # too sharp -> increase tau (soften)
            lo = mid
    tau = 0.5 * (lo + hi)
    L = - D2 / tau
    return L


@torch.no_grad()
def init_phase1_stable(model):
    # (a) degree bias (you already have EB shrink—use it; else your simpler variant)
    init_biases_from_degrees(model, model.sparse_i_idx, model.sparse_j_idx, undirected=True)

    # (b) canonicalize the spectral embedding you already computed
    X = canonicalize_spectral(model.spectral_data)   # [N,K] (float32 is fine)

    # (c) deterministic anchors and entropy-calibrated logits
    anchor_idx = farthest_first_anchors(X, model.K)
    L = logits_from_anchors(X, anchor_idx, target_entropy_frac=0.2)  # 0.55–0.65 is a good range
    model.latent_z1.data.copy_(L)

    # (d) map A to fit the embedding (least squares → your boxed SVD)
    # init_A_by_least_squares(model)  # you already defined this helper
    calibrate_b0_to_density(model)

    # (f) optional: initialize community intercepts from assignments (no randomness)
    init_B_from_assignments(model, model.sparse_i_idx, model.sparse_j_idx)

    # (g) locals: tight, non-overlapping by construction (keeps phase-1 stable)
    init_anchor_dominant_rows(model, target_s=0.05)  # small spread around anchor
    model.anchor_eps = 0.4                           # < 0.5, gives a bit of room later
