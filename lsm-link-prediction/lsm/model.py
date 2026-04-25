"""
Latent Space Model (LSM) with anchor-dominant local convex hulls.

Defines the main `LSM` class, including the spectral initialization
mixin from `lsm.spectral_clustering.Spectral_clustering_init`.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics

from lsm.spectral_clustering import Spectral_clustering_init


# Device setup (matches original script)
CUDA = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if CUDA:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')


class LSM(nn.Module, Spectral_clustering_init):
    def __init__(self, number_of_K, latent_dim, k_extra, sparse_i, sparse_j,
                 input_size1, input_size2, sample_size, scaling=None):
        super(LSM, self).__init__()
        # initialization
        Spectral_clustering_init.__init__(self, num_of_eig=latent_dim, device=device)

        self.D = latent_dim
        self.K = number_of_K

        self.input_size = input_size1

        self.softmax_loc = nn.Softmax(dim=-1)

        self.log_s = None

        self.b0 = nn.Parameter(torch.zeros(1, device=device))
        self.g = nn.Parameter(torch.randn(input_size1, device=device))
        self.d = nn.Parameter(torch.randn(input_size2, device=device))

        self.logit_sigma = nn.Parameter(torch.zeros(self.K))

        self.scaling = 1
        # create indices to index properly the receiver and senders variable
        self.sparse_i_idx = sparse_i
        self.sparse_j_idx = sparse_j

        self.spectral_data = self.spectral_clustering()  # +1e-1

        self.Softmax = nn.Softmax(1)

        self.memberships_loc = nn.Parameter(torch.rand(input_size1, self.K, device=device))

        self.k_extra = k_extra
        self.softplus = nn.Softplus()
        self.A_uncon = nn.Parameter(torch.rand(self.K, self.D, device=device))

        self.s_dim_ = nn.Parameter(torch.rand(self.D, device=device))

        self.torch_pi = torch.tensor(math.pi)

        # init_pre_softplus = torch.log(torch.exp(torch.tensor(target_value)) - 1)

        self.s_ = nn.Parameter(2 * torch.ones(1))

        self.log_sigma = torch.nn.Parameter(torch.zeros(self.K))

        # self.latent_z1=nn.Parameter(torch.log(init_z+0.1))
        # self.latent_w1=nn.Parameter(torch.log(init_w+0.1))

        # self.temperature_=nn.Parameter(torch.rand(1))

        # self.latent_z1=nn.Parameter(torch.rand(input_size1,self.K,device=device))
        self.latent_z1 = nn.Parameter(self.spectral_data)

        self.local_epoch = 0

        self.latent_w1 = nn.Parameter(torch.rand(input_size2, self.K, device=device))

        self.A_free = nn.Parameter(torch.randn(self.K, self.D, device=device))

        self.local_hull_con = nn.Parameter(torch.randn(self.K, (self.K - 1), self.K, device=device))

        self.sampling_weights = torch.ones(input_size1)

        self.sample_size = sample_size

        self.s_logits = torch.nn.Parameter(torch.zeros(self.K))  # learns effective rank

        self.Hu = torch.nn.Parameter(torch.rand(self.K, self.K))
        self.Hv = torch.nn.Parameter(torch.rand(self.K, self.K))
        self.sig_free = torch.nn.Parameter(torch.zeros(self.K))
        self.nu_free = torch.nn.Parameter(torch.zeros(1))

        self.pi_logits = torch.nn.Parameter(torch.zeros(self.K))   # shape [r]

        self.s_free = torch.nn.Parameter(torch.zeros(self.K))
        self.phase_1 = True

        self.w_free = nn.Parameter(torch.zeros(self.K, self.K, device=device))  # [hull, dim]

        self.B_free = nn.Parameter(torch.rand(self.K, device=device))  # community-pair logits
        # Cholesky of Psi: L is lower-triangular with positive diagonal

        # __init__
        self.H = nn.Parameter(torch.randn(self.D, self.D))       # for eigenvectors
        self.logit_eigs = nn.Parameter(torch.zeros(self.D))      # for eigenvalues

        self.C_free = nn.Parameter(torch.zeros(self.K, self.D, self.D))   # unconstrained
        self.logsig = nn.Parameter(torch.zeros(self.K, self.D))      # per-dim log-scales

        # --- Separation prior hyperparams & params (REPLACE THIS BLOCK) ---
        # Separator directions u_{kℓ} live in R^D, not R^K
        self.sep_u = nn.Parameter(torch.randn(self.K, self.K, self.D))  # [k, ℓ, D]
        self.sep_b = nn.Parameter(torch.zeros(self.K, self.K))          # [k, ℓ]

        # Prior scales (treat as fixed hyperparams for MAP)
        self.sep_tau = 0.03   # soft max/min temperature (sharper = 0.02–0.05)
        self.sep_rho = 15     # target margin in whitened latent units
        self.sep_sigma = 0.01  # logistic "noise" temperature in the prior (smaller = stronger)
        self.sep_beta = 1.0    # softplus slope (keep 1.0 if using sigma)

        # per-pair separation strengths (only k<ℓ used)
        self.log_lambda_sep = nn.Parameter(torch.full((self.K, self.K), -2.0))  # exp(-2) ≈ 0.14 start
        self.lambda_prior_rate = 1.0   # b in Gamma(a,b); see below
        self.lambda_prior_shape = 1.5  # a > 1 encourages moderate growth

        # after self.local_hull_con = nn.Parameter(torch.randn(self.K,(self.K-1),self.K, device=device))
        self.local_shrink = nn.Parameter(torch.zeros(self.K, self.K - 1, device=device))  # t -> s = eps * sigmoid(t)
        self.anchor_eps = .49  # < 0.5 to guarantee disjoint hulls by construction; you can anneal later

    def build_anchor_dominant_B(self, eps=None):
        """
        Returns B_all: [K, K, K] where for each hull k:
          - K-1 rows live in the slab {w_k >= 1 - eps}
          - last row is the pure anchor e_k
        """
        if eps is None:
            eps = float(self.anchor_eps)
        K = self.K
        logits = self.local_hull_con            # [K, K-1, K]
        shrink_t = self.local_shrink            # [K, K-1]

        eyeK = torch.eye(K, device=logits.device, dtype=logits.dtype)
        out = []

        for k in range(K):
            L = logits[k]                       # [K-1, K]
            t = shrink_t[k]                     # [K-1]
            # mask out the anchor column
            mask = torch.ones(K, device=logits.device, dtype=torch.bool)
            mask[k] = False
            # distribution over non-anchor coordinates
            Q = F.softmax(L[:, mask], dim=-1)   # [K-1, K-1]
            # shrink s in (0, eps)
            s = eps * torch.sigmoid(t)          # [K-1]

            W = torch.zeros(K - 1, K, device=logits.device, dtype=logits.dtype)
            W[:, mask] = s.unsqueeze(-1) * Q    # non-anchor mass
            W[:, k] = 1.0 - s                   # anchor mass

            # append exact anchor vertex e_k
            Bk = torch.cat([W, eyeK[k].unsqueeze(0)], dim=0)  # [K, K]
            out.append(Bk)

        return torch.stack(out, dim=0)          # [K, K, K]

    def A_svd_boxed(self, Hu, Hv, sig_free, sigma_min=.5, sigma_max=2.5, eps=1e-8):
        # Orthonormal columns (retraction via QR)
        Qu = torch.linalg.qr(Hu, mode='reduced')[0]  # [K,K]
        Qv = torch.linalg.qr(Hv, mode='reduced')[0]  # [K,K]
        # Box singular values
        s = sigma_min + (sigma_max - sigma_min) * torch.sigmoid(sig_free)  # [K]
        A = (Qu * s.unsqueeze(0)) @ Qv.T
        return A

    def dpp_across_hulls_set_rbf(self,
                                 A_all,                  # [H,K,D]
                                 pin_masks=None,         # [H,K] booleans; True=pinned (exclude)
                                 center_rows=True,       # subtract per-hull mean (on normalized rows)
                                 tau=None,               # if None, use median heuristic
                                 eps=1e-6, alpha=1.):
        H, K, D = A_all.shape
        if H <= 1:
            return A_all.new_tensor(0.0)

        # 1) Normalize rows (directional geometry)
        A_norm = 1 * A_all  # F.normalize(A_all, dim=-1)

        # 2) Optionally center each hull (reduce mean-shift dominance)
        if center_rows:
            means = A_norm.mean(dim=1, keepdim=True)
            A_norm = F.normalize(A_norm - means, dim=-1)

        # 3) Build set embeddings via kernel mean embedding with RBF
        # L_{pq} = mean_{x in p, y in q} exp(-||x-y||^2 / tau)
        # Auto τ by median of cross-hull distances (on a subsample if needed)
        if tau is None:
            # collect a few cross-hull distances
            dists = []
            for p in range(H):
                Xp = A_norm[p] if pin_masks is None else A_norm[p][~pin_masks[p]]
                if Xp.shape[0] == 0:
                    continue
                for q in range(p + 1, H):
                    Xq = A_norm[q] if pin_masks is None else A_norm[q][~pin_masks[q]]
                    if Xq.shape[0] == 0:
                        continue
                    D2 = self._pairwise_sq_dists(Xp, Xq)
                    dists.append(D2.reshape(-1))
            if len(dists) == 0:
                tau_eff = A_all.new_tensor(1.0)
            else:
                all_d = torch.cat(dists)
                tau_eff = all_d.median().clamp_min(1e-6)
        else:
            tau_eff = torch.as_tensor(tau, dtype=A_all.dtype, device=A_all.device)

        L = A_all.new_zeros(H, H, dtype=A_all.dtype)
        for p in range(H):
            Xp = A_norm[p] if pin_masks is None else A_norm[p][~pin_masks[p]]
            if Xp.shape[0] == 0:
                continue
            for q in range(p, H):
                Xq = A_norm[q] if pin_masks is None else A_norm[q][~pin_masks[q]]
                if Xq.shape[0] == 0:
                    continue
                D2 = self._pairwise_sq_dists(Xp, Xq)
                Kpq = torch.exp(-D2 / tau_eff)           # [|Xp|, |Xq|]
                val = Kpq.mean()
                L[p, q] = L[q, p] = val

        I = torch.eye(H, device=L.device, dtype=L.dtype)
        logdetL = torch.slogdet(alpha * L + eps * I).logabsdet
        logdetIp = torch.slogdet(I + alpha * L).logabsdet
        return (logdetL - logdetIp)

    def sample_pos_edges(self, sparse_i, sparse_j, M_pos):
        E = sparse_i.numel()
        idx = torch.randint(0, E, (M_pos,), device=sparse_i.device)
        return sparse_i[idx], sparse_j[idx]                     # pos_i, pos_j

    def sample_uniform_pairs(self, N, M_neg, device, return_q=False, symmetric=True):
        """
        Uniform negative sampling over ordered pairs (i != j).
        If symmetric=True, also returns the mirrored pairs (j,i) so that
        negatives match the undirected edge convention (stored both ways).
        """
        # sample M_neg ordered pairs with i != j
        i = torch.randint(0, N, (M_neg,), device=device)
        j = torch.randint(0, N - 1, (M_neg,), device=device)
        j = j + (j >= i)  # skip self-pairs by shifting >= i

        if symmetric:
            i_all = torch.cat([i, j], dim=0)
            j_all = torch.cat([j, i], dim=0)
        else:
            i_all, j_all = i, j

        if return_q:
            # uniform over all ordered pairs i != j
            q = torch.full((i_all.numel(),), 1.0 / (N * (N - 1)), device=device)
            return i_all, j_all, q
        return i_all, j_all

    def sample_network(self):
        # Lazy import: torch_sparse is only needed if sample_network() is actually called.
        from torch_sparse import spspmm

        # sample for undirected network
        sample_idx = torch.multinomial(self.sampling_weights, self.sample_size, replacement=False)
        # translate sampled indices w.r.t. to the full matrix, it is just a diagonal matrix
        indices_translator = torch.cat([sample_idx.unsqueeze(0), sample_idx.unsqueeze(0)], 0)
        # adjacency matrix in edges format
        edges = torch.cat([self.sparse_i_idx.unsqueeze(0), self.sparse_j_idx.unsqueeze(0)], 0)
        # matrix multiplication B = Adjacency x Indices translator
        # see spspmm function, it give a multiplication between two matrices
        # indexC is the indices where we have non-zero values and valueC the actual values (in this case ones)
        indexC, valueC = spspmm(edges, torch.ones(edges.shape[1]), indices_translator,
                                torch.ones(indices_translator.shape[1]),
                                self.input_size, self.input_size, self.input_size, coalesced=True)
        # second matrix multiplication C = Indices translator x B, indexC returns where we have edges inside the sample
        indexC, valueC = spspmm(indices_translator, torch.ones(indices_translator.shape[1]), indexC, valueC,
                                self.input_size, self.input_size, self.input_size, coalesced=True)

        # edge row position
        sparse_i_sample = indexC[0, :]
        # edge column position
        sparse_j_sample = indexC[1, :]

        return sample_idx, sparse_i_sample, sparse_j_sample

    def log_prior_bias(self, tau_g=5.0):
        # g_i ~ N(0, tau_g^2)  (choose tau_g based on degree dispersion)
        return -0.5 * (self.g ** 2).sum() / (tau_g ** 2)

    def dpp_prior_within_hull(self, A, tau=1., eps=1e-6):
        """
        Proper L-ensemble DPP log-prior on the K rows of A (shape [K, D]),
        using cosine (Gram) features on unit-normalized rows.
        Returns log det(L) - log det(I+L), where L = tau * (U U^T).
        """
        A = A.to(torch.float64)
        U = A / (A.norm(dim=1, keepdim=True).clamp_min(1e-30))      # [K, D]
        G = U @ U.T
        G = 0.5 * (G + G.T)                                         # symmetrize
        L = tau * G
        I = torch.eye(L.size(0), dtype=L.dtype, device=L.device)
        logdetL = torch.slogdet(L + eps * I).logabsdet              # jitter for PD-ness
        logdetIp = torch.slogdet(I + L).logabsdet
        return (logdetL - logdetIp).to(A.dtype)

    def _pairwise_sq_dists(self, X, Y=None):
        # X:[n,d], Y:[m,d] or None -> [n,n]
        if Y is None:
            Y = X
        x2 = (X * X).sum(-1, keepdim=True)
        y2 = (Y * Y).sum(-1, keepdim=True)
        D2 = (x2 + y2.T - 2 * (X @ Y.T)).clamp_min(0)
        return D2

    def dpp_prior_within_hull_rbf(self,
                                  A,
                                  tau=None,               # if None -> median heuristic on off-diagonal distances
                                  eps=1e-6,
                                  normalize=False,        # normalize rows -> direction/shape only
                                  angular=False,          # if normalize=True, use angular distance (unit-sphere RBF)
                                  center=True,            # subtract per-hull mean (after normalize) then re-normalize
                                  mask=None               # optional boolean mask over rows to EXCLUDE (e.g., pinned)
                                  ):
        """
        Proper L-ensemble DPP on the K rows of A (K x D) using an RBF kernel.
        Returns: log det(L) - log det(I + L)

        - normalize=True: work on directions (scale-invariant).
        - angular=True: use angular distance D^2 = 2 - 2*cos; otherwise Euclidean on (possibly normalized) rows.
        - center=True: subtract mean before distance (focus on shape, not location), then renormalize.
        - mask: exclude rows from the kernel (e.g., exact pure-corner rows).
        """
        A = A.to(torch.float64)
        if mask is not None:
            X = A[~mask]
        else:
            X = A
        K = X.shape[0]
        if K <= 1:
            return A.new_tensor(0.0, dtype=A.dtype)

        # Normalize / center if requested
        if normalize:
            X = F.normalize(X, dim=-1)
        if center:
            mu = X.mean(dim=0, keepdim=True)
            X = X - mu
            # keep on unit sphere if normalize=True
            if normalize:
                X = F.normalize(X, dim=-1)

        # Distances
        if normalize and angular:
            # angular (on unit sphere): D^2 = 2 - 2 cos
            C = (X @ X.T).clamp(-1.0, 1.0)
            D2 = (2.0 - 2.0 * C).clamp_min(0)
        else:
            D2 = self._pairwise_sq_dists(X)

        # Median heuristic for tau if not provided
        if tau is None:
            off = D2[~torch.eye(K, dtype=torch.bool, device=D2.device)]
            if off.numel() == 0:
                tau_eff = A.new_tensor(1.0, dtype=A.dtype)
            else:
                tau_eff = off.median().clamp_min(1e-6)
        else:
            tau_eff = torch.as_tensor(tau, dtype=A.dtype, device=A.device).clamp_min(1e-12)

        # RBF kernel (PSD)
        L = torch.exp(-D2 / tau_eff)
        # L-ensemble DPP log-likelihood
        I = torch.eye(K, dtype=L.dtype, device=L.device)
        logdetL = torch.slogdet(L + eps * I).logabsdet
        logdetIp = torch.slogdet(I + L).logabsdet
        return (logdetL - logdetIp).to(A.dtype)

    def dpp_across_hulls_centroid_rbf(self, A_all, whiten=True, tau=None, alpha=0.5, eps=1e-6):
        H, K, D = A_all.shape
        if H <= 1:
            return A_all.new_tensor(0.0)
        Z = A_all
        if whiten:
            X = Z.reshape(-1, D)
            mu = X.mean(0, keepdim=True)
            std = X.std(0, keepdim=True).clamp_min(1e-6)
            Z = (Z - mu) / std
        C = Z.mean(dim=1)                        # [H, D]
        D2 = self._pairwise_sq_dists(C)
        if tau is None:
            off = D2[~torch.eye(H, dtype=torch.bool, device=C.device)]
            tau_eff = (off.median() if off.numel() else D2.new_tensor(1.0)).clamp_min(1e-6)
        else:
            tau_eff = torch.as_tensor(tau, dtype=C.dtype, device=C.device)
        L = torch.exp(-D2 / tau_eff)             # PSD
        I = torch.eye(H, device=C.device, dtype=C.dtype)
        return torch.slogdet(alpha * L + eps * I).logabsdet - torch.slogdet(I + alpha * L).logabsdet

    def log_prior_shrink(self, a=1.8, b=6.0, eps=1e-8):
        """
        Beta(a,b) prior on s/eps ∈ (0,1). Returns log prior (constants dropped for MAP).
        Using t = sigmoid(local_shrink), s = eps * t; density ∝ t^{a-1} (1-t)^{b-1}.
        """
        t = torch.sigmoid(self.local_shrink)  # in (0,1)
        return ((a - 1.0) * torch.log(t + eps) + (b - 1.0) * torch.log(1.0 - t + eps)).sum()

    # introduce the likelihood function containing the two extra biases gamma_i and alpha_j
    def LSM_likelihood_bias_cs(self, epoch, temperature=1, r=1, Poisson=False):
        '''
        Poisson log-likelihood ignoring the log(k!) constant
        '''
        self.epoch = epoch

        # self.latent_w=self.Softmax(self.latent_w1/temperature)
        M_neg = 10 * self.sparse_i_idx.shape[0]

        neg_i, neg_j = self.sample_uniform_pairs(self.input_size, M_neg, device=device)

        scale = (self.input_size * (self.input_size - 1)) / neg_i.shape[0]
        self.s = self.softplus(self.s_)

        if self.epoch == 2000:
            # self.gamma.data=0.5*self.bias+self.gamma.data
            self.phase_1 = False
            self.pre_epochs = 1 * self.epoch

        if self.epoch == 1000:
            # self.gamma.data=0.5*self.bias+self.gamma.data
            self.scaling = 0

        if self.phase_1:
            # self.scaling=0
            if self.scaling:
                # self.gamma=self.gammas[layer]
                # z_pdist1=0.5*torch.mm(torch.exp(self.gammas[layer].unsqueeze(0)),(torch.mm((mat-torch.diag(torch.diagonal(mat))),torch.exp(self.gammas[layer]).unsqueeze(-1))))

                if Poisson:
                    mat = torch.exp(self.g[neg_i] + self.g[neg_j])
                else:
                    mat = self.softplus(self.g[neg_i] + self.g[neg_j])

                # z_pdist1=0.5*torch.mm(torch.exp(self.gammas[layer].unsqueeze(0)),(torch.mm((mat-torch.diag(torch.diagonal(mat))),torch.exp(self.gammas[layer]).unsqueeze(-1))))

                z_pdist1 = scale * mat.sum()  # (mat-torch.diag(torch.diagonal(mat))).sum()

                # take the sampled matrix indices in order to index gamma_i and alpha_j correctly and in agreement with the previous
                # z_pdist2=(self.gammas[layer][sparse_i_]+self.gammas[layer][sparse_j_]).sum()
                z_pdist2 = ((self.g[self.sparse_i_idx] + self.g[self.sparse_j_idx])).sum()

                log_likelihood_sparse = z_pdist2 - z_pdist1

                # self.L_.data=(self.gamma.view(-1)+self.delta.view(-1)).abs().max()
                # self.latent_z.data=self.latent_z.data*self.scaling_factor.data
                return log_likelihood_sparse

            else:
                self.latent_z_ = self.Softmax(self.latent_z1)

                # self.latent_z, self.log_prior_A,self.A = self.project_A_archetypes_roworth(self.latent_z_,self.A_free,self.tau_free)

                self.A = self.A_svd_boxed(self.Hu, self.Hv, self.sig_free)
                self.latent_z = self.latent_z_ @ self.A

                alpha_node = .5  # good default for blocky communities
                self.log_p_z = (((alpha_node - 1.0) *
                                 torch.log(self.latent_z_.clamp_min(1e-6))
                                 ).sum(dim=1).sum()) + self.dpp_prior_within_hull(self.A)  # average over nodes

                mat_0 = self.s * (((self.latent_z[neg_i]) * (self.latent_z[neg_j] + 1e-06)).sum(-1)) + (
                    self.g[neg_i] + self.g[neg_j])  # +self.b0

                mat_1 = self.s * (((self.latent_z[self.sparse_i_idx]) * (self.latent_z[self.sparse_j_idx] + 1e-06)).sum(-1)) + (
                    (self.g[self.sparse_i_idx] + self.g[self.sparse_j_idx]))  # +self.b0

                if Poisson:
                    mat = torch.exp(mat_0)
                else:
                    mat = self.softplus(mat_0)
                z_pdist2 = mat_1.sum()
                z_pdist1 = scale * mat.sum()

                log_likelihood_sparse = z_pdist2 - z_pdist1

                return log_likelihood_sparse

        else:
            self.log_p_z = 0

            self.A = self.A_svd_boxed(self.Hu, self.Hv, self.sig_free)

            self.b_hull = F.softmax(self.local_hull_con, -1)

            alpha_node = 1.
            self.b_hull_loc = (((alpha_node - 1.0) *
                                torch.log(self.b_hull.clamp_min(1e-6))
                                ).sum(dim=-1).sum())

            # --- Anchor-dominant barycentric rows (guaranteed disjoint if eps<0.5) ---
            self.b_hull_total = self.build_anchor_dominant_B(self.anchor_eps)  # [K, K, K]
            # Note: no stabilized_weights() here; anchor-dominance already constrains rows

            # propagate to local vertices in latent space
            self.A_all = self.b_hull_total @ self.A      # [K, K, D]

            self.A_all = self.b_hull_total @ self.A

            # logits -> centered & normalized for cosine Gram
            # [K, K-1, K]
            # z_n = logits_centered / (logits_centered.norm(dim=-1, keepdim=True) + 1e-8)    # [K, K-1, K]

            slogdets = []

            for k in range(self.K):
                slogdets.append(self.dpp_prior_within_hull(self.A_all[k]))

            dpp_logprior_within_hull = torch.stack(slogdets).sum()   # sum over hulls (no mean)

            log_p_shrink = self.log_prior_shrink(a=1., b=1.)

            self.log_p_b_loc = log_p_shrink + dpp_logprior_within_hull + self.dpp_prior_within_hull(self.A)

            def T_sched(epoch, start=2000, end=5500, Thi=2.0, Tlo=0.35):
                if epoch <= start:
                    return Thi
                if epoch >= end:
                    return Tlo
                t = (epoch - start) / (end - start)
                # cosine ramp
                return Tlo + 0.5 * (Thi - Tlo) * (1 + math.cos(math.pi * t))
            # replace tt with:
            tt = T_sched(epoch)

            if epoch < 9000:
                self.m = F.gumbel_softmax(self.latent_z1, tt, hard=False, dim=1)
            else:
                self.m = F.gumbel_softmax(self.latent_z1, tt, hard=True, dim=1)

            # self.m=self.Softmax(self.latent_z1)         # [N, C]

            alloc = self.m.argmax(1)

            self.latent_z_loc = self.Softmax(self.memberships_loc)

            if epoch < 6000:
                alpha_node = 1.
            elif epoch < 7000:
                alpha_node = 1.
            else:
                alpha_node = 1

            self.log_p_z_loc = (((alpha_node - 1.0) *
                                 torch.log(self.latent_z_loc.clamp_min(1e-6))
                                 ).sum(dim=1).sum())

            # self.loc_b=self.B_free[self.m.argmax(1)]

            # self.latent_z = torch.einsum('nk,ked,nd->ne', self.m, self.A_all, self.latent_z_loc)
            self.latent_z = torch.einsum('nk,kld,nl->nd', self.m, self.A_all, self.latent_z_loc)

            mat_0 = self.s * (((self.latent_z[neg_i]) * (self.latent_z[neg_j] + 1e-06)).sum(-1)) + (
                self.g[neg_i] + self.g[neg_j])  # +self.b0#+(self.B_free[alloc[neg_i]]+self.B_free[alloc[neg_j]])#+(self.loc_b[sample_idx].unsqueeze(-1)+self.loc_b[sample_idx])
            # pol_1=(1/4)*((((((self.latent_z[neg_i])+(self.latent_z[neg_j]+1e-06))**2).sum(-1)))-(((((self.latent_z[neg_i])-(self.latent_z[neg_j]+1e-06))**2).sum(-1))))
            # mat_0=self.s* pol_1+(self.g[neg_i]+self.g[neg_j])#+(self.B_free[alloc[neg_i]]+self.B_free[alloc[neg_j]])#+(self.loc_b[sample_idx].unsqueeze(-1)+self.loc_b[sample_idx])

            # pol_2=(1/4)*((((((self.latent_z[self.sparse_i_idx])+(self.latent_z[self.sparse_j_idx]+1e-06))**2).sum(-1)))-(((((self.latent_z[self.sparse_i_idx])-(self.latent_z[self.sparse_j_idx]+1e-06))**2).sum(-1))))
            # mat_1=self.s*pol_2+((self.g[self.sparse_i_idx]+self.g[self.sparse_j_idx]))#+(self.B_free[alloc[self.sparse_i_idx]]+self.B_free[alloc[self.sparse_j_idx]])#+((self.loc_b[sparse_i_sample]+self.loc_b[sparse_j_sample]))

            mat_1 = self.s * (((self.latent_z[self.sparse_i_idx]) * (self.latent_z[self.sparse_j_idx] + 1e-06)).sum(-1)) + (
                (self.g[self.sparse_i_idx] + self.g[self.sparse_j_idx]))  # +self.b0#+(self.B_free[alloc[self.sparse_i_idx]]+self.B_free[alloc[self.sparse_j_idx]])#+((self.loc_b[sparse_i_sample]+self.loc_b[sparse_j_sample]))

            if Poisson:
                mat = torch.exp(mat_0)
            else:
                mat = self.softplus(mat_0)
            z_pdist2 = mat_1.sum()
            z_pdist1 = scale * mat.sum()

            log_likelihood_sparse = z_pdist2 - z_pdist1  # + log_p_z#-(0.001*deg_[layer]*((self.latent_raa_z[:,layer]-1)**2)).sum()

            return log_likelihood_sparse

    def link_prediction(self, rem_i, rem_j, target):

        with torch.no_grad():

            self.s = self.softplus(self.s_)

            if self.phase_1:

                # self.latent_z, _,_ = self.project_A_archetypes_roworth(self.latent_z_,self.A_free, self.tau_free)
                self.A = self.A_svd_boxed(self.Hu, self.Hv, self.sig_free)
                self.latent_z = self.latent_z_ @ self.A

                # Z_norm = self._center_and_normalize(self.latent_z)
                # cos_edges = (Z_norm[rem_i] * Z_norm[rem_j]).sum(dim=-1)

                rates = self.s * (((self.latent_z[rem_i]) * (self.latent_z[rem_j] + 1e-06)).sum(-1)) + (
                    self.g[rem_i] + self.g[rem_j])  # +self.b0

            else:

                self.m = F.one_hot(self.latent_z1.argmax(1), num_classes=self.K).float()

                # self.m = self.Softmax(self.latent_z1).float()

                alloc = self.m.argmax(1)

                self.latent_z_loc = self.Softmax(self.memberships_loc)

                self.latent_z = torch.einsum('nk,ked,ne->nd', self.m, self.A_all, self.latent_z_loc)

                # dot_edges = self._metric_dot_per_hull(
                #     self.latent_z[rem_i], self.latent_z[rem_j],
                #     self.m[rem_i],        self.m[rem_j],
                #     combine="prod"
                # )
                # rates = self.s * dot_edges + (self.g[rem_i] + self.g[rem_j])

                rates = self.s * (((self.latent_z[rem_i]) * (self.latent_z[rem_j] + 1e-06)).sum(-1)) + (
                    self.g[rem_i] + self.g[rem_j])  # +self.b0#+(self.B_free[alloc[rem_i]]+self.B_free[alloc[rem_j]])#+(self.loc_b[rem_i]+self.loc_b[rem_j])

            # rates=-self.s*(((((self.latent_z[rem_i])-(self.latent_z[rem_j]+1e-06))**2).sum(-1))**0.5)+(self.g[rem_i]+self.g[rem_j])

        precision, tpr, thresholds = metrics.precision_recall_curve(target.cpu().data.numpy(), rates.cpu().data.numpy())

        return metrics.roc_auc_score(target.cpu().data.numpy(), rates.cpu().data.numpy()), metrics.auc(tpr, precision)
