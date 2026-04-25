"""
Visualization utilities for the LSM model:
- PCA projections of latent space, archetypes, and local hulls
- Reordered adjacency spy plots with community block outlines
- Per-community circular plots
- Diagnostic plots (per-cluster PCA, LDA projection)
"""

import math

import numpy as np
import matplotlib.pyplot as plt
import torch
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Rectangle
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


def make_distinct_palette(K):
    """Build a large qualitative palette of K visually distinct colors."""
    cmaps = [
        plt.cm.tab10, plt.cm.Dark2, plt.cm.Set1, plt.cm.Set2, plt.cm.Set3,
        plt.cm.Paired, plt.cm.tab20, plt.cm.tab20b, plt.cm.tab20c
    ]
    cols = []
    for cm in cmaps:
        n = getattr(cm, 'N', 256)
        # Sample evenly across each qualitative map
        take = 10 if cm in (plt.cm.tab10, plt.cm.Dark2, plt.cm.Set1, plt.cm.Set2,
                            plt.cm.Set3, plt.cm.Paired) else 20
        idxs = np.linspace(0, n - 1, take, dtype=int)
        cols.extend([cm(i) for i in idxs])
        if len(cols) >= K:
            break
    # Fallback if still short: evenly sample HSV
    if len(cols) < K:
        hsv = plt.cm.hsv(np.linspace(0, 1, K - len(cols), endpoint=False))
        cols.extend([tuple(c) for c in hsv])
    return cols[:K]


def plot_reordered_adjacency_simple(model, N, sparse_i, sparse_j, K, save_path=None):
    """Spy-like reordered adjacency with one rectangle per community block."""
    adj = torch.zeros(N, N)
    adj[sparse_i, sparse_j] = 1

    mask = torch.arange(0, model.K ** 2, model.K)
    idx = model.latent_z_.argmax(1)
    idx = (mask[idx] + model.latent_z_loc.argmax(1)).argsort()

    plt.spy(adj[:, idx][idx, :], markersize=0.1)
    plt.show()


def plot_pca_latent_space(model, save_path=None):
    """PCA projection of latent_z, A, and A_all."""
    pca = PCA(n_components=2)

    X = model.latent_z.detach().cpu().numpy()
    A_ = model.A.detach().cpu().numpy()
    all_A = model.A_all.view(-1, model.K).detach().cpu().numpy()

    pca.fit(np.concatenate((all_A, A_, X)))

    A_proj = pca.transform(A_)
    X_proj = pca.transform(all_A)
    latent_proj = pca.transform(X)

    print("Explained variance ratio:", pca.explained_variance_ratio_)
    print("Singular values:", pca.singular_values_)

    plt.figure(figsize=(10, 8))

    scatter1 = plt.scatter(
        latent_proj[:, 0], latent_proj[:, 1],
        c=model.m.argmax(1).detach().cpu().numpy(),
        cmap='tab10', alpha=0.7, s=40, label="Latent Z"
    )

    plt.scatter(
        X_proj[:, 0], X_proj[:, 1],
        c='green', alpha=0.6, s=25, label="A_all"
    )

    plt.scatter(
        A_proj[:, 0], A_proj[:, 1],
        c='black', marker='X', s=100, label="A"
    )

    plt.colorbar(scatter1, label="Cluster assignment (argmax of m)")
    plt.xlabel("PCA Component 1", fontsize=12)
    plt.ylabel("PCA Component 2", fontsize=12)
    plt.title("PCA Projection of Latent Space and A Matrices", fontsize=14, fontweight='bold')
    plt.legend(frameon=True)
    plt.grid(True, linestyle="--", alpha=0.5)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.show()

    return pca, A_proj, X_proj, latent_proj


def report_hull_diagnostics(model):
    """Print per-hull log-volume proxy and minimum pairwise vertex distances."""
    with torch.no_grad():
        # per-hull log-volume proxy
        vols = []
        for c in range(model.A_all.size(0)):
            B = model.A_all[c]
            a = B[-1]
            E = B[:-1] - a.unsqueeze(0)
            G = E @ E.T + 1e-6 * torch.eye(E.size(0), device=B.device)
            vols.append(float(0.5 * torch.logdet(G).cpu()))
        print("log-volume per hull:", vols)

        # check for duplicate local vertices
        for c in range(model.A_all.size(0)):
            Dm = torch.cdist(model.A_all[c], model.A_all[c])
            Dm = Dm + 1e9 * torch.eye(Dm.size(0), device=Dm.device)
            print(f"min pairwise dist in hull {c}:", float(Dm.min().cpu()))


def plot_hulls_pca(model):
    """Project A_all to 2D via global PCA and draw each hull."""
    A_all = model.A_all.detach().cpu().numpy()   # [K, L, D]

    pca = PCA(n_components=2)
    A_all_2d = pca.fit_transform(A_all.reshape(-1, A_all.shape[-1]))
    A_all_2d = A_all_2d.reshape(A_all.shape[0], A_all.shape[1], 2)

    plt.figure(figsize=(8, 8))
    colors = ["red", "blue", "green", "orange", "purple"]

    for k in range(A_all.shape[0]):
        P = A_all_2d[k]
        hull = ConvexHull(P)
        plt.fill(P[hull.vertices, 0], P[hull.vertices, 1], alpha=0.3,
                 color=colors[k % len(colors)])
        plt.scatter(P[:, 0], P[:, 1], color=colors[k % len(colors)], s=30, edgecolor="k")

    plt.title("Convex hulls from model.A_all")
    plt.axis("equal")
    plt.show()


def plot_hulls_lda(model):
    """Project A_all via LDA and draw each hull."""
    A = model.A_all.detach().cpu().numpy()
    K, L, D = A.shape

    X = A.reshape(-1, D)
    y = np.repeat(np.arange(K), L)

    n_comp = min(2, max(1, K - 1))
    proj = LDA(n_components=n_comp).fit_transform(X, y)
    P2 = proj.reshape(K, L, n_comp)

    plt.figure(figsize=(8, 8))
    colors = ["red", "blue", "green", "orange", "purple"]

    for k in range(K):
        P = P2[k]
        if n_comp == 1:
            P = np.c_[P[:, 0], np.zeros_like(P[:, 0])]

        hull = ConvexHull(P)
        plt.fill(P[hull.vertices, 0], P[hull.vertices, 1],
                 alpha=0.25, color=colors[k % len(colors)])
        plt.scatter(P[:, 0], P[:, 1], s=30, c=colors[k % len(colors)],
                    edgecolors="black", linewidths=0.5)

        plt.title("Convex hulls via LDA projection")
        plt.axis("equal")
        plt.tight_layout()
        plt.show()


def plot_hulls_per_cluster_pca(model):
    """Each hull projected with its own PCA in a side-by-side layout."""
    A = model.A_all.detach().cpu().numpy()
    K, L, D = A.shape
    colors = ["red", "blue", "green", "orange", "purple"]

    fig, axes = plt.subplots(1, K, figsize=(4 * K, 4))
    if K == 1:
        axes = [axes]

    for k in range(K):
        pca_k = PCA(n_components=2).fit(A[k])
        P = pca_k.transform(A[k])
        hull = ConvexHull(P)
        ax = axes[k]
        ax.fill(P[hull.vertices, 0], P[hull.vertices, 1],
                alpha=0.25, color=colors[k % len(colors)])
        ax.scatter(P[:, 0], P[:, 1], s=30, c=colors[k % len(colors)],
                   edgecolors="black", linewidths=0.5)
        ax.set_title(f"Cluster {k} (own PCA)")
        ax.axis("equal")

    plt.tight_layout()
    plt.show()


def plot_full_latent_with_hulls(model, A_proj, X_proj, latent_proj, K, save_path=None):
    """Full annotated plot: nodes, local hulls colored, global archetypes, global hull."""
    labels = model.m.argmax(1).detach().cpu().numpy()
    K_detected = int(labels.max() + 1)
    palette = make_distinct_palette(K_detected)
    cmap_cat = ListedColormap(palette)
    norm_cat = BoundaryNorm(np.arange(K_detected + 1) - 0.5, K_detected)

    K_hulls, L = model.A_all.shape[:2]
    B_proj = X_proj.reshape(K_hulls, L, 2)

    plt.figure(figsize=(10, 8))

    plt.scatter(
        latent_proj[:, 0], latent_proj[:, 1],
        c=labels, cmap=cmap_cat, norm=norm_cat,
        alpha=0.75, s=40, linewidths=0.2, edgecolors='k',
        label="Node embeddings $z_i$"
    )

    for c in range(K_hulls):
        Pc = B_proj[c]
        if Pc.shape[0] >= 3:
            hc = ConvexHull(Pc)
            polyc = Pc[hc.vertices]
            plt.fill(
                polyc[:, 0], polyc[:, 1],
                alpha=0.20, zorder=0,
                facecolor=palette[c % K_detected]
            )
            plt.plot(
                np.r_[polyc[:, 0], polyc[0, 0]],
                np.r_[polyc[:, 1], polyc[0, 1]],
                color=palette[c % K_detected], lw=1.0, zorder=1
            )

    plt.scatter(
        X_proj[:, 0], X_proj[:, 1],
        c='black', edgecolor='black', linewidths=0.6,
        alpha=0.7, s=100, label="$B_k$ prototypes", zorder=100
    )

    plt.scatter(
        A_proj[:, 0], A_proj[:, 1],
        c='red', marker='X', s=150, label="Global archetypes $A$", zorder=150
    )

    pts = A_proj[:, :2]
    if pts.shape[0] >= 3:
        hull = ConvexHull(pts)
        poly = pts[hull.vertices]
        plt.fill(poly[:, 0], poly[:, 1], facecolor='black', alpha=0.10, zorder=0,
                 label="Global hull ($A$)")
        plt.plot(np.r_[poly[:, 0], poly[0, 0]], np.r_[poly[:, 1], poly[0, 1]],
                 color='black', lw=1.2, zorder=1)
    elif pts.shape[0] == 2:
        plt.plot(pts[:, 0], pts[:, 1], color='black', lw=1.2, zorder=1)

    plt.xlabel("PCA Component 1", fontsize=20)
    plt.ylabel("PCA Component 2", fontsize=20)

    plt.axis('equal')
    plt.legend(frameon=True, fontsize=15, loc='best')
    plt.tight_layout()
    plt.tick_params(axis='x', labelsize=18)
    plt.tick_params(axis='y', labelsize=18)

    if save_path is None:
        save_path = f"latent_hulls_{K}.png"
    plt.savefig(save_path, bbox_inches="tight", dpi=200)

    plt.show()


def plot_block_adjacency(model, N, sparse_i, sparse_j, K, save_path=None):
    """Reordered spy adjacency with off-diagonal block outlines colored from Tab20."""
    adj = torch.zeros(N, N, device=sparse_i.device, dtype=torch.float32)
    adj[sparse_i, sparse_j] = 1.0
    adj[sparse_j, sparse_i] = 1.0
    adj.fill_diagonal_(0)

    mask = torch.arange(0, model.K ** 2, model.K, device=adj.device)
    idx_comm = model.latent_z_.argmax(1)
    idx_loc = model.latent_z_loc.argmax(1)
    key = mask[idx_comm] + idx_loc
    idx = torch.argsort(key)

    A = adj[idx][:, idx].detach().cpu().numpy()
    comm_ord = idx_comm[idx].detach().cpu().numpy()

    ii, jj = np.nonzero(A)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(jj, ii, s=1.0, c='#1f77b4', marker='s', linewidths=0)

    ax.invert_yaxis()
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-0.5, A.shape[1] - 0.5)
    ax.set_ylim(A.shape[0] - 0.5, -0.5)
    ax.set_xticks([])
    ax.set_yticks([])

    if comm_ord.size > 0:
        change = np.where(np.diff(comm_ord) != 0)[0] + 1
        starts = np.r_[0, change]
        ends = np.r_[change, comm_ord.size]

        TAB20_PAIR = "dark"
        pair_offset = 0 if TAB20_PAIR == "dark" else 1

        palette = list(plt.cm.tab20.colors)

        blocks = [(int(s), int(e), int(comm_ord[s])) for s, e in zip(starts, ends)]

        MIN_ONES = 1
        A_bin = (A > 0)

        for i, (rs, re, ci) in enumerate(blocks):
            for j, (cs, ce, cj) in enumerate(blocks):
                sub = A_bin[rs:re, cs:ce]
                if np.count_nonzero(sub) >= MIN_ONES:
                    on_diag = (i == j)

                    color_idx = (2 * ci + pair_offset) % len(palette)
                    color = palette[color_idx]

                    rect = Rectangle(
                        (cs - 0.5, rs - 0.5),
                        ce - cs, re - rs,
                        fill=False,
                        edgecolor=color,
                        linewidth=5 if on_diag else 2.0,
                        alpha=0.9 if on_diag else 0.75,
                        linestyle='-' if on_diag else '--',
                        joinstyle="miter",
                        capstyle="butt",
                        clip_on=False
                    )
                    ax.add_patch(rect)

    plt.tight_layout()
    if save_path is None:
        save_path = f"adj_{K}.png"
    plt.savefig(save_path, bbox_inches="tight", dpi=600)
    plt.show()


def plot_simple_block_adjacency(model, N, sparse_i, sparse_j, K, save_path=None):
    """Simpler reordered spy adjacency with one rectangle per diagonal block."""
    adj = torch.zeros(N, N, device=sparse_i.device, dtype=torch.float32)
    adj[sparse_i, sparse_j] = 1.0
    adj[sparse_j, sparse_i] = 1.0
    adj.fill_diagonal_(0)

    mask = torch.arange(0, model.K ** 2, model.K, device=adj.device)
    idx_comm = model.latent_z_.argmax(1)
    idx_loc = model.latent_z_loc.argmax(1)
    key = mask[idx_comm] + idx_loc
    idx = torch.argsort(key)

    A = adj[idx][:, idx].detach().cpu().numpy()
    comm_ord = idx_comm[idx].detach().cpu().numpy()

    ii, jj = np.nonzero(A)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(jj, ii, s=1., c='blue', marker='s', linewidths=0)

    ax.invert_yaxis()
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-0.5, A.shape[1] - 0.5)
    ax.set_ylim(A.shape[0] - 0.5, -0.5)
    ax.set_xticks([])
    ax.set_yticks([])

    if comm_ord.size > 0:
        change = np.where(np.diff(comm_ord) != 0)[0] + 1
        starts = np.r_[0, change]
        ends = np.r_[change, comm_ord.size]

        palette = list(plt.cm.tab20.colors)

        for s, e in zip(starts, ends):
            cval = int(comm_ord[s])
            rect = Rectangle(
                (s - 0.5, s - 0.5),
                e - s, e - s,
                fill=False,
                edgecolor=palette[cval % len(palette)],
                linewidth=5,
                alpha=0.8
            )
            ax.add_patch(rect)

    plt.tight_layout()
    if save_path is None:
        save_path = f"adj_{K}.png"
    plt.savefig(save_path, bbox_inches="tight", dpi=600)
    plt.show()


def plot_circular_per_community(model, sparse_i, sparse_j, dataset):
    """Per-community circular plot (one figure per community)."""
    labels = model.m.argmax(1).detach().cpu().numpy()
    K_detected = int(labels.max() + 1)
    palette = make_distinct_palette(K_detected)

    for k in range(model.K):

        col = palette[k % K_detected]

        idx = model.m.argmax(1) == k

        nodes = torch.where(model.m.argmax(1) == k)[0]
        print(nodes.shape[0])

        sparse_i_ = []
        sparse_j_ = []

        for i, j in zip(sparse_i, sparse_j):
            if i in nodes:
                if j in nodes:
                    sparse_i_.append(i)
                    sparse_j_.append(j)

        sparse_i__ = torch.stack(sparse_i_).long()
        sparse_j__ = torch.stack(sparse_j_).long()

        mask = torch.zeros(int(nodes.max() + 1)).long()

        mask[nodes] = torch.arange(nodes.unique().shape[0]).long()

        sparse_i_ = mask[sparse_i__]
        sparse_j_ = mask[sparse_j__]

        Z = model.latent_z_loc[idx].detach().cpu().numpy()

        pca = PCA(n_components=2)

        X_ = pca.fit_transform(Z)

        arg_max = Z.argmax(1)

        comp = pca.components_.transpose()
        inv = np.arctan2(comp[:, 1], comp[:, 0])
        degree = np.mod(np.degrees(inv), 360)

        idxs = np.argsort(degree)

        step = (2 * math.pi) / model.K
        radius = 10
        points = np.zeros((model.K, 2))
        for i in range(model.K):
            points[i, 0] = (radius * math.cos(i * step))
            points[i, 1] = (radius * math.sin(i * step))

        points = points[idxs]

        # plt.scatter(points[:,0],points[:,1])
        _X = Z @ points

        print('CREATING and SAVING circular plots!!! \n')
        plt.figure(figsize=(7, 7), dpi=300)

        for i, j in zip(sparse_i_.cpu().numpy(), sparse_j_.cpu().numpy()):
            plt.plot([_X[i, 0], _X[j, 0]], [_X[i, 1], _X[j, 1]],
                     color=col, lw=0.7, alpha=0.35, zorder=2)
        plt.scatter(_X[:, 0], _X[:, 1], c=col, s=50, zorder=500,
                    edgecolors="black", linewidths=0.5)
        plt.scatter(points[:, 0], points[:, 1], c='black', s=150, alpha=0.8, zorder=1000)
        plt.set_cmap("tab10")
        plt.axis('off')
        plt.savefig(f"cir_{dataset}_{k}.png", dpi=200)
        plt.show()


def plot_full_latent_with_hulls_no_global(model, A_proj, X_proj, latent_proj, save_path_template=None):
    """
    Same as plot_full_latent_with_hulls but without the global hull shading,
    saved per-c (matches the second per-c plot block in the original script).
    """
    labels = model.m.argmax(1).detach().cpu().numpy()
    K_detected = int(labels.max() + 1)
    palette = make_distinct_palette(K_detected)
    cmap_cat = ListedColormap(palette)
    norm_cat = BoundaryNorm(np.arange(K_detected + 1) - 0.5, K_detected)

    K_hulls, L = model.A_all.shape[:2]
    B_proj = X_proj.reshape(K_hulls, L, 2)

    plt.figure(figsize=(10, 8))

    plt.scatter(
        latent_proj[:, 0], latent_proj[:, 1],
        c=labels, cmap=cmap_cat, norm=norm_cat,
        alpha=0.75, s=40, linewidths=0.2, edgecolors='k',
        label="Node embeddings $z_i$"
    )

    for c in range(K_hulls):
        Pc = B_proj[c]
        if Pc.shape[0] >= 3:
            hc = ConvexHull(Pc)
            polyc = Pc[hc.vertices]
            plt.fill(
                polyc[:, 0], polyc[:, 1],
                alpha=0.20, zorder=0,
                facecolor=palette[c % K_detected]
            )
            plt.plot(
                np.r_[polyc[:, 0], polyc[0, 0]],
                np.r_[polyc[:, 1], polyc[0, 1]],
                color=palette[c % K_detected], lw=1.0, zorder=1
            )

    plt.scatter(
        X_proj[:, 0], X_proj[:, 1],
        c='black', edgecolor='black', linewidths=0.6,
        alpha=0.7, s=100, label="$B_k$ prototypes", zorder=100
    )

    plt.scatter(
        A_proj[:, 0], A_proj[:, 1],
        c='red', marker='X', s=150, label="Global archetypes $A$", zorder=150
    )

    plt.axis('equal')
    plt.tight_layout()
    plt.tick_params(axis='x', labelsize=18)
    plt.tick_params(axis='y', labelsize=18)

    # Mirrors the original loop's saving inside plot_full_latent_with_hulls_no_global
    if save_path_template is None:
        save_path_template = "latent_hulls_{c}_.png"
    # The original code saves once with the last c value; preserve that behavior.
    save_path = save_path_template.format(c=K_hulls - 1)
    plt.savefig(save_path, bbox_inches="tight", dpi=200)

    plt.show()
