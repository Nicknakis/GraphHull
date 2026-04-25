"""
Entry point: train an LSM model and produce all post-training visualizations.

Run from the project root:

    python main.py
"""

import math
import time

import numpy as np
import torch
import torch.optim as optim

# `torch.distributions.dirichlet.Dirichlet` is imported in the original script before
# the main block; preserve that here even though it is not directly referenced.
from torch.distributions.dirichlet import Dirichlet  # noqa: F401

from lsm.model import LSM, device
from lsm.ema import EMA
from lsm.geometry import overlapping_pairs, hulls_intersect_lp, hull_min_distance_qp
from lsm.data import load_dataset
from lsm.initializers import (
    init_biases_from_degrees,
    init_memberships_from_spectral,
    init_A_from_centroids,
)
from lsm.visualization import (
    plot_reordered_adjacency_simple,
    plot_pca_latent_space,
    report_hull_diagnostics,
    plot_hulls_pca,
    plot_hulls_lda,
    plot_hulls_per_cluster_pca,
    plot_full_latent_with_hulls,
    plot_block_adjacency,
    plot_circular_per_community,
    plot_full_latent_with_hulls_no_global,
)

start = time.perf_counter()

print(device)


def main():
    psis = [1.]

    for psi in psis:
        dataset = 'hepth'

        sparse_i, sparse_j, rem_i, rem_j, target, N = load_dataset(dataset, device)

        priors = []
        s = []

        min_loss_RE = 1000000

        epoch_num = 10000
        runs = 1

        for run in range(runs):

            print("RUN number:", run)

            sample_size = int(0.3 * N)

            # Missing_data should be set to False for link_prediction since we do
            # not consider these interactions as missing but as zeros.
            model = LSM(number_of_K=8, latent_dim=8, k_extra=None,
                        sparse_i=sparse_i, sparse_j=sparse_j,
                        input_size1=N, input_size2=N,
                        sample_size=sample_size).to(device)
            init_biases_from_degrees(model, sparse_i, sparse_j, undirected=True)
            # init_B_from_assignments(model, sparse_i, sparse_j)

            init_memberships_from_spectral(model)
            init_A_from_centroids(model)
            # init_anchor_dominant_rows(model)

            nu = model.D + 2

            d = model.D
            sigma_star = 0.7                # your desired marginal std per dim
            Psi = (nu + d + 1) * (sigma_star ** 2) * torch.eye(d)

            optimizer1 = optim.Adam(model.parameters(), 0.05)

            # optimizer1 = torch.optim.AdamW(model.parameters(), lr=5e-2, weight_decay=1e-4)

            dyads = (N * (N - 1))

            num_pairs = model.K * (model.K - 1) // 2

            colors = np.array(["green", "blue", "red"])
            losses = []

            ema = EMA(model, decay=0.999)

            for epoch in range(epoch_num):
                if epoch == 8000:
                    model.sample_size = int(0.3 * N)

                if model.scaling:

                    loss1 = -model.LSM_likelihood_bias_cs(epoch=epoch)

                else:

                    if model.phase_1:
                        loss1 = (-model.LSM_likelihood_bias_cs(epoch=epoch)
                                 + 0.5 * (model.s ** 2)
                                 - model.log_p_z
                                 - model.log_prior_bias())
                    if not model.phase_1:

                        def sched_linear(epoch, start, end, hi, lo):
                            if epoch <= start:
                                return hi
                            if epoch >= end:
                                return lo
                            t = (epoch - start) / max(1, end - start)
                            return hi + (lo - hi) * t

                        N_full = model.input_size            # total nodes
                        s_curr = model.sample_size           # sampled nodes this step
                        dyads_full = N_full * (N_full - 1)
                        dyads_batch = s_curr * (s_curr - 1)
                        scale = dyads_full / max(1, dyads_batch)

                        like_term = -model.LSM_likelihood_bias_cs(epoch=epoch)
                        # sep_energy_per_pair = model.sep_loss / max(1, num_pairs)

                        # sep_prior_term =  scale * sep_energy_per_pair

                        priors = (0.5 * (model.s ** 2)
                                  - model.log_p_z
                                  - model.log_prior_bias()
                                  - model.log_p_z_loc
                                  - model.log_p_b_loc  # -model.b_hull_loc
                                  )  # +model.lambda_neglogprior)   # sigma already encodes strength → MAP-valid

                        loss1 = like_term + priors

                # model_pks.append(1*model.p_k.detach())

                optimizer1.zero_grad()
                loss1.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)   # (see §4)
                optimizer1.step()

                if epoch % 100 == 0:
                    print(loss1)

                    if not model.scaling:
                        AUC_ROC, AUC_PR = model.link_prediction(rem_i, rem_j, target)

                        print(f'epoch ={epoch} ---- AUC_ROC = {AUC_ROC} ---- AUC_PR = {AUC_PR}')

                        # AUC_ROC,AUC_PR=model.link_prediction_(rem_i, rem_j, target)
                        # print(f'epoch ={epoch} ---- GLASS AUC_ROC = {AUC_ROC} ---- GLASS AUC_PR = {AUC_PR}')

                    # if epoch>6000:
                    #     # Example:
                    #     polys = list(model.A_all.detach().cpu().numpy())
                    #     print(overlapping_pairs(polys))

                current_lr = optimizer1.param_groups[0]['lr']
                # print(f"Epoch {epoch}: Current LR = {current_lr:.5f}")
                min_lr_threshold = 1e-07
                # Early stop if learning rate is below threshold
                if current_lr < min_lr_threshold:
                    print(f"Stopping early at epoch {epoch} as learning rate dropped below {min_lr_threshold}")
                    break
                if epoch == 5000:
                    # scheduler.step(loss1)
                    optimizer1.param_groups[0]['lr'] = 0.01

                losses.append(loss1.item())

        # `model.A_all` only exists once phase 2 has been entered (epoch >= 2000).
        # Skip these checks if the run stopped earlier (e.g. small `epoch_num`).
        if hasattr(model, "A_all"):
            polys = list(model.A_all.detach().numpy())
            print(overlapping_pairs(polys))

    # `model.link_prediction` requires phase 1 or phase 2 state (it reads
    # `self.latent_z_`, which is only created once scaling is turned off).
    if not model.scaling:
        AUC_ROC, AUC_PR = model.link_prediction(rem_i, rem_j, target)
        print(f'epoch ={epoch} ---- AUC_ROC = {AUC_ROC} ---- AUC_PR = {AUC_PR}')
    else:
        print(f'Training stopped during scaling phase (epoch={epoch}); '
              f'skipping final link_prediction.')

    # ---- Post-training visualizations & diagnostics ----
    # All of the visualizations below assume phase 2 has been reached and so
    # `model.A_all` exists. Skip them otherwise.
    if not hasattr(model, "A_all"):
        print("Training stopped before phase 2 (epoch < 2000); skipping post-training plots.")
        return

    K = model.K

    # Reordered adjacency (simple spy)
    plot_reordered_adjacency_simple(model, N, sparse_i, sparse_j, K)

    # Cluster sizes diagnostic
    with torch.no_grad():
        z_now = torch.softmax(model.latent_z1 / 0.3, dim=1)   # low T for eval only
        labels = torch.argmax(z_now, 1)
        print(torch.bincount(labels, minlength=model.K))      # class sizes

    # PCA projection
    pca, A_proj, X_proj, latent_proj = plot_pca_latent_space(model)

    # Hull diagnostics
    report_hull_diagnostics(model)

    # Pairwise hull intersection (LP) and minimum distance (QP)
    K_, L_, D_ = model.A_all.shape
    A_all_np = model.A_all.detach().cpu().numpy()

    overlap = np.zeros((K_, K_), dtype=bool)
    for i in range(K_):
        for j in range(i + 1, K_):
            overlap[i, j] = hulls_intersect_lp(A_all_np[i], A_all_np[j])
            overlap[j, i] = overlap[i, j]
    print("pairwise hull intersection (True = overlap):\n", overlap)

    min_d = np.full((K_, K_), np.nan)
    for i in range(K_):
        for j in range(i + 1, K_):
            d_val = hull_min_distance_qp(A_all_np[i], A_all_np[j])
            min_d[i, j] = min_d[j, i] = d_val
    print("pairwise hull margins:\n", min_d)
    print("global min margin:", np.nanmin(min_d[np.triu_indices(K_, 1)]))

    # Hull plots
    plot_hulls_pca(model)
    plot_hulls_lda(model)
    plot_hulls_per_cluster_pca(model)

    # Full latent + hulls (with global hull)
    plot_full_latent_with_hulls(model, A_proj, X_proj, latent_proj, K)

    # Block adjacency with off-diagonal outlines
    plot_block_adjacency(model, N, sparse_i, sparse_j, K)

    # Reproduce the warm-up `fracs` block from the original (kept for parity)
    fracs = []
    epochs = 10000
    for e in range(epochs):
        frac = min(1.0, (e / epochs)) ** 0.5
        fracs.append(frac)

    # Final overlapping-pair check (matches the duplicate at the end of the original)
    polys = list(model.A_all.detach().numpy())
    print(overlapping_pairs(polys))

    # Per-community circular plots
    plot_circular_per_community(model, sparse_i, sparse_j, dataset)

    # Per-c latent + hulls plot (no global hull)
    plot_full_latent_with_hulls_no_global(model, A_proj, X_proj, latent_proj)


if __name__ == "__main__":
    main()
