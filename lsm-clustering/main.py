"""
Entry point for the clustering variant: trains an LSM and reports NMI/ARI
against ground-truth labels.

Run from the project root:

    python main.py
"""

import time

import numpy as np
import torch

# `torch.distributions.dirichlet.Dirichlet` is imported in the original script
# before the main block; preserve that here even though it is not directly
# referenced.
from torch.distributions.dirichlet import Dirichlet  # noqa: F401

from lsm.seed_utils import set_seed
from lsm.model import LSM, device
from lsm.data import load_dataset
from lsm.initializers import init_phase1_stable

# Reproducibility (matches the original SEED=9 block)
SEED = 9   # 42, 0, 1993, 28, 9
set_seed(SEED)

start = time.perf_counter()

print(device)


def main():
    psis = [1.]

    for psi in psis:
        dataset = 'lastfm_with_labels'

        sparse_i, sparse_j, labels_, k_labs, N = load_dataset(dataset, device)

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
            model = LSM(number_of_K=k_labs, latent_dim=k_labs, k_extra=None,
                        sparse_i=sparse_i, sparse_j=sparse_j,
                        input_size1=N, input_size2=N,
                        sample_size=sample_size).to(device)
            init_phase1_stable(model)

            nu = model.D + 2

            d = model.D
            sigma_star = 0.7                # your desired marginal std per dim
            Psi = (nu + d + 1) * (sigma_star ** 2) * torch.eye(d)

            optimizer1 = torch.optim.Adam(model.parameters(), lr=5e-2,
                                          betas=(0.9, 0.99), weight_decay=1e-4)

            # optimizer1 = torch.optim.AdamW(model.parameters(), lr=0.05, weight_decay=1e-4)

            dyads = (N * (N - 1))

            num_pairs = model.K * (model.K - 1) // 2

            colors = np.array(["green", "blue", "red"])
            losses = []

            for epoch in range(epoch_num):
                if epoch == 8000:
                    model.sample_size = int(0.3 * N)

                if model.scaling:

                    loss1 = -model.LSM_likelihood_bias_cs(epoch=epoch) - model.log_prior_bias()

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

                        priors = (0.5 * ((model.s ** 2) / (1 ** 2))
                                  - model.log_p_z
                                  - model.log_prior_bias()
                                  - model.log_p_z_loc
                                  - model.log_p_b_loc  # -model.b_hull_loc
                                  )

                        loss1 = like_term + priors

                optimizer1.zero_grad()
                loss1.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)   # (see §4)
                optimizer1.step()

                if epoch % 100 == 0:
                    if not model.scaling:
                        print(loss1)

                        if model.K == k_labs:
                            NMI, ARI = model.clustering(labels_)

                            print(f'epoch ={epoch} ---- NMI = {NMI} ---- ARI = {ARI}')
                        if model.K != k_labs:

                            NMI, ARI = model.clustering_spherical(labels_)

                            print(f'epoch ={epoch} ---- NMI_kmeans = {NMI} ---- ARI_kmeans = {ARI}')

                current_lr = optimizer1.param_groups[0]['lr']
                # print(f"Epoch {epoch}: Current LR = {current_lr:.5f}")
                min_lr_threshold = 1e-07
                # Early stop if learning rate is below threshold
                if current_lr < min_lr_threshold:
                    print(f"Stopping early at epoch {epoch} as learning rate dropped below {min_lr_threshold}")
                    break
                if epoch == 2000:
                    # scheduler.step(loss1)
                    optimizer1.param_groups[0]['lr'] = 1e-02
                    # current_lr = optimizer1.param_groups[0]['lr']
                    # print(current_lr)
                if epoch == 4000:
                    # scheduler.step(loss1)
                    optimizer1.param_groups[0]['lr'] = 1e-03

                losses.append(loss1.item())


if __name__ == "__main__":
    main()
