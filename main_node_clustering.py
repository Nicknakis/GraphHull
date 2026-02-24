import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.distributions.dirichlet import Dirichlet
from sklearn.decomposition import PCA
import argparse
import torch.functional as f
from spectral_clustering import Spectral_clustering_init

import numpy as np
from itertools import combinations
from scipy.optimize import linprog
from torch.distributions.lkj_cholesky import LKJCholesky
from torch.distributions import HalfCauchy, Dirichlet
from sklearn.cluster import KMeans

from sklearn.metrics import normalized_mutual_info_score,adjusted_rand_score
from copy import deepcopy
# Import all the packages
import torch
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.nn.functional as f# create a dummy data 
import matplotlib.pyplot as plt
import networkx as nx
import timeit
from sklearn import metrics
#from blobs import *
from sklearn.decomposition import PCA
# import sparse 
import scipy.sparse as sparse
# import stats
import math
import scipy.stats as stats
from sklearn import metrics
from scipy.io import loadmat  # this is the SciPy module that loads mat-files
import pandas as pd
#from torch_sparse import spspmm
from scipy import special
from cvxopt import matrix, solvers
solvers.options['show_progress'] = False
from numpy.matlib import repmat
from sklearn.preprocessing import StandardScaler
from scipy.stats import fisher_exact
from torch_sparse import spspmm


start = timeit.default_timer()
CUDA = torch.cuda.is_available()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
if CUDA:        
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

print(device)


from sklearn.metrics import mutual_info_score, normalized_mutual_info_score
from scipy.stats import entropy


import os, random, numpy as np, torch
#42,0,1993,28,9
SEED = 9
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"   # CUDA determinism
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark    = False
torch.use_deterministic_algorithms(True, warn_only=True)






#H,W=calculate_nmf_for_layer(pruned_layers[1],latent_dim)


from copy import deepcopy
# Import all the packages
import torch
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.nn.functional as f# create a dummy data 
import matplotlib.pyplot as plt
import networkx as nx
import timeit
#from spectral_clustering import Spectral_clustering_init
from sklearn import metrics
#from blobs import *
from sklearn.decomposition import PCA
# import sparse 
import scipy.sparse as sparse
# import stats
import math
import scipy.stats as stats
from sklearn import metrics
from scipy.io import loadmat  # this is the SciPy module that loads mat-files
import pandas as pd
from scipy import special
from cvxopt import matrix, solvers
solvers.options['show_progress'] = False
from numpy.matlib import repmat
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_mutual_info_score
import torch.nn.functional as F
from sklearn.preprocessing import normalize

from torch.distributions import MultivariateNormal


start = timeit.default_timer()
CUDA = torch.cuda.is_available()


from sklearn.model_selection import KFold

undirected=1


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
    Aeq = np.zeros((D + 2, 2*D))
    beq = np.zeros(D + 2)
    # Vector equality: P^T alpha - Q^T beta = 0  (D rows)
    Aeq[:D, :D] = P.T
    Aeq[:D, D:] = -Q.T
    # Sum-to-one constraints
    Aeq[D, :D]  = 1.0; beq[D]   = 1.0
    Aeq[D+1, D:] = 1.0; beq[D+1] = 1.0

    res = linprog(c=np.zeros(2*D), A_eq=Aeq, b_eq=beq,
                  bounds=[(0, None)]*(2*D), method="highs")

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





class LSM(nn.Module,Spectral_clustering_init):
    def __init__(self,number_of_K,latent_dim,k_extra,sparse_i,sparse_j, input_size1,input_size2,sample_size,scaling=None):
        super(LSM, self).__init__()
        # initialization
        Spectral_clustering_init.__init__(self,method='Normalized',num_of_eig=number_of_K,device=device)

        
        self.D=latent_dim
        self.K=number_of_K


        self.input_size=input_size1
        
        

        self.softmax_loc=nn.Softmax(dim=-1)
       
        self.log_s=None
       

        self.g=nn.Parameter(torch.randn(input_size1,device=device))


        self.scaling=1
        #create indices to index properly the receiver and senders variable
        self.sparse_i_idx=sparse_i
        self.sparse_j_idx=sparse_j
        
        self.spectral_data=self.spectral_clustering()

        
        self.Softmax=nn.Softmax(1)

        
        self.memberships_loc=nn.Parameter(torch.rand(input_size1,self.K,device=device))

        self.k_extra=k_extra
        self.softplus=nn.Softplus()
        self.A_uncon=nn.Parameter(torch.rand(self.K,self.D, device=device))
        
        self.s_dim_=nn.Parameter(torch.rand(self.D, device=device))
        
        self.torch_pi=torch.tensor(math.pi)
        
        #init_pre_softplus = torch.log(torch.exp(torch.tensor(target_value)) - 1)

        self.s_=nn.Parameter(torch.ones(1))
        
        self.s_k_=nn.Parameter(torch.ones(self.K))

        
        self.log_sigma = torch.nn.Parameter(torch.zeros(self.K)) 
        
        
        #self.latent_z1=nn.Parameter(torch.log(init_z+0.1))
        #self.latent_w1=nn.Parameter(torch.log(init_w+0.1))

       # self.temperature_=nn.Parameter(torch.rand(1))

        #self.latent_z1=nn.Parameter(torch.rand(input_size1,self.K,device=device))
        self.latent_z1=nn.Parameter(self.spectral_data)

        self.local_epoch=0

        self.latent_w1=nn.Parameter(torch.rand(input_size2,self.K,device=device))
        
        self.A_free=nn.Parameter(torch.randn(self.K,self.D, device=device))
        
        
        self.local_hull_con=nn.Parameter(torch.randn(self.K,(self.K-1),self.K, device=device))

        
       # 
        
        self.sampling_weights=torch.ones(input_size1)
        
        self.sample_size=sample_size
        
       
        self.s_logits = torch.nn.Parameter(torch.zeros(self.K))  # learns effective rank
        r = min(self.K, self.D)
        self.Hu = nn.Parameter(0.1*torch.randn(self.K, r, device=device))  # left subspace
        self.Hv = nn.Parameter(0.1*torch.randn(self.D, r, device=device))  # right subspace
        self.sig_free = nn.Parameter(torch.zeros(r, device=device)) 
        
       
                
        # self.Hu = torch.nn.Parameter(torch.rand(self.K, self.K))
        # self.Hv = torch.nn.Parameter(torch.rand(self.K, self.K) )
        # self.sig_free = torch.nn.Parameter(torch.zeros(self.K))
        self.nu_free = torch.nn.Parameter(torch.zeros(1))
        
        self.pi_logits = torch.nn.Parameter(torch.zeros(self.K))   # shape [r]
        
        self.s_free = torch.nn.Parameter(torch.zeros(self.K))
        self.phase_1=True
        
        self.w_free = nn.Parameter(torch.zeros(self.K, self.K, device=device))  # [hull, dim]
        
        self.b0 = nn.Parameter(torch.tensor(0.0, device=device))

        
        self.B_free = nn.Parameter(torch.rand(self.K, device=device))  # community-pair logits
        # Cholesky of Psi: L is lower-triangular with positive diagonal
        
        # __init__
        self.H = nn.Parameter(torch.randn(self.D, self.D))       # for eigenvectors
        self.logit_eigs = nn.Parameter(torch.zeros(self.D)) # for eigenvalues
        
        self.C_free = nn.Parameter(torch.zeros(self.K, self.D, self.D))   # unconstrained
        self.logsig = nn.Parameter(torch.zeros(self.K, self.D))      # per-dim log-scales
        
        # --- Separation prior hyperparams & params (REPLACE THIS BLOCK) ---
        # Separator directions u_{kℓ} live in R^D, not R^K
        self.sep_u = nn.Parameter(torch.randn(self.K, self.K, self.D))  # [k, ℓ, D]
        self.sep_b = nn.Parameter(torch.zeros(self.K, self.K))          # [k, ℓ]
        
        # Prior scales (treat as fixed hyperparams for MAP)
        self.sep_tau   = 0.03   # soft max/min temperature (sharper = 0.02–0.05)
        self.sep_rho   = 15  # target margin in whitened latent units
        self.sep_sigma = 0.01   # logistic "noise" temperature in the prior (smaller = stronger)
        self.sep_beta  = 1.0    # softplus slope (keep 1.0 if using sigma)
        
        # per-pair separation strengths (only k<ℓ used)
        self.log_lambda_sep = nn.Parameter(torch.full((self.K, self.K), -2.0))  # exp(-2) ≈ 0.14 start
        self.lambda_prior_rate = 1.0  # b in Gamma(a,b); see below
        self.lambda_prior_shape = 1.5 # a > 1 encourages moderate growth
        
        # after self.local_hull_con = nn.Parameter(torch.randn(self.K,(self.K-1),self.K, device=device))
        self.local_shrink = nn.Parameter(torch.zeros(self.K, self.K-1, device=device))  # t -> s = eps * sigmoid(t)
        self.anchor_eps = .5 # < 0.5 to guarantee disjoint hulls by construction; you can anneal later
        
        
    def build_anchor_dominant_B(self, eps=None):
        """
        Returns B_all: [K, K, K] where for each hull k:
          - K-1 rows live in the slab {w_k >= 1 - eps}
          - last row is the pure anchor e_k
        """
        if eps is None: eps = float(self.anchor_eps)
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
    
            W = torch.zeros(K-1, K, device=logits.device, dtype=logits.dtype)
            W[:, mask] = s.unsqueeze(-1) * Q    # non-anchor mass
            W[:, k]    = 1.0 - s                # anchor mass
    
            # append exact anchor vertex e_k
            Bk = torch.cat([W, eyeK[k].unsqueeze(0)], dim=0)  # [K, K]
            out.append(Bk)
    
        return torch.stack(out, dim=0)          # [K, K, K]


 
 
    # def A_svd_boxed(self,Hu, Hv, sig_free, sigma_min=.3, sigma_max=2., eps=1e-8):
    #     # Orthonormal columns (retraction via QR)
    #     Qu = torch.linalg.qr(Hu, mode='reduced')[0]  # [K,K]
    #     Qv = torch.linalg.qr(Hv, mode='reduced')[0]  # [K,K]
    #     # Box singular values
    #     s = sigma_min + (sigma_max - sigma_min) * torch.sigmoid(sig_free)  # [K]
    #     A = (Qu * s.unsqueeze(0)) @ Qv.T
    #     return A
    
    def A_svd_boxed(self, sigma_min=.1, sigma_max=1.5):
        # Qu: (K, r), Qv: (D, r)
        Qu = torch.linalg.qr(self.Hu, mode='reduced')[0]
        Qv = torch.linalg.qr(self.Hv, mode='reduced')[0]
        s  = sigma_min + (sigma_max - sigma_min) * torch.sigmoid(self.sig_free)  # (r,)
    
        # diag(s) @ Qv^T  →  scale rows of Qv^T
        # Qv.T: (r, D); s[:, None]: (r, 1)
        right = Qv.T * s.unsqueeze(1)   # (r, D)
    
        # Qu @ right  →  (K, r) @ (r, D) = (K, D)
        return Qu @ right


   
   
    
    def dpp_across_hulls_set_rbf(self,
        A_all,                  # [H,K,D]
        pin_masks=None,         # [H,K] booleans; True=pinned (exclude)
        center_rows=True,       # subtract per-hull mean (on normalized rows)
        tau=None,               # if None, use median heuristic
        eps=1e-6,alpha=1.
    ):
        H, K, D = A_all.shape
        if H <= 1:
            return A_all.new_tensor(0.0)
    
        # 1) Normalize rows (directional geometry)
        A_norm = 1*A_all#F.normalize(A_all, dim=-1)
    
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
                if Xp.shape[0] == 0: continue
                for q in range(p+1, H):
                    Xq = A_norm[q] if pin_masks is None else A_norm[q][~pin_masks[q]]
                    if Xq.shape[0] == 0: continue
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
                L[p,q] = L[q,p] = val
                
    
        I = torch.eye(H, device=L.device, dtype=L.dtype)
        logdetL  = torch.slogdet(alpha*L + eps*I).logabsdet
        logdetIp = torch.slogdet(I + alpha*L).logabsdet
        return (logdetL - logdetIp)
    
    
    def sample_pos_edges(self,sparse_i, sparse_j, M_pos):
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
        # USE torch_sparse lib i.e. : from torch_sparse import spspmm

        # sample for undirected network
        sample_idx=torch.multinomial(self.sampling_weights, self.sample_size,replacement=False)
        # translate sampled indices w.r.t. to the full matrix, it is just a diagonal matrix
        indices_translator=torch.cat([sample_idx.unsqueeze(0),sample_idx.unsqueeze(0)],0)
        # adjacency matrix in edges format
        edges=torch.cat([self.sparse_i_idx.unsqueeze(0),self.sparse_j_idx.unsqueeze(0)],0)
        # matrix multiplication B = Adjacency x Indices translator
        # see spspmm function, it give a multiplication between two matrices
        # indexC is the indices where we have non-zero values and valueC the actual values (in this case ones)
        indexC, valueC = spspmm(edges,torch.ones(edges.shape[1]), indices_translator,torch.ones(indices_translator.shape[1]),self.input_size,self.input_size,self.input_size,coalesced=True)
        # second matrix multiplication C = Indices translator x B, indexC returns where we have edges inside the sample
        indexC, valueC=spspmm(indices_translator,torch.ones(indices_translator.shape[1]),indexC,valueC,self.input_size,self.input_size,self.input_size,coalesced=True)
        
        # edge row position
        sparse_i_sample=indexC[0,:]
        # edge column position
        sparse_j_sample=indexC[1,:]
        
        return sample_idx,sparse_i_sample,sparse_j_sample
    
       
        
    def log_prior_bias(self, tau_g=5):
        # g_i ~ N(0, tau_g^2)  (choose tau_g based on degree dispersion)
        return -0.5 * (self.g ** 2).sum() / (tau_g ** 2)
    
    
    
 


    
    
    def dpp_prior_within_hull(self,A, tau=1., eps=1e-6):
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
        logdetL  = torch.slogdet(L + eps*I).logabsdet               # jitter for PD-ness
        logdetIp = torch.slogdet(I + L).logabsdet
        return (logdetL - logdetIp).to(A.dtype)




    def _pairwise_sq_dists(self,X, Y=None):
        # X:[n,d], Y:[m,d] or None -> [n,n]
        if Y is None: Y = X
        x2 = (X*X).sum(-1, keepdim=True)
        y2 = (Y*Y).sum(-1, keepdim=True)
        D2 = (x2 + y2.T - 2 * (X @ Y.T)).clamp_min(0)
        return D2
    
    def dpp_prior_within_hull_rbf(self,
        A, 
        tau=None,               # if None -> median heuristic on off-diagonal distances
        eps=1e-6, 
        normalize=False,         # normalize rows -> direction/shape only
        angular=False,           # if normalize=True, use angular distance (unit-sphere RBF)
        center=True,           # subtract per-hull mean (after normalize) then re-normalize
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
        logdetL  = torch.slogdet(L + eps * I).logabsdet
        logdetIp = torch.slogdet(I + L).logabsdet
        return (logdetL - logdetIp).to(A.dtype)
    
    
    
    def dpp_across_hulls_centroid_rbf(self, A_all, whiten=False, tau=1, alpha=1., eps=1e-6):
        H, K, D = A_all.shape
        if H <= 1: return A_all.new_tensor(0.0)
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
        return torch.slogdet(alpha*L + eps*I).logabsdet - torch.slogdet(I + alpha*L).logabsdet


        
    def dpp_across_hulls_centroid_gram(self,
                                   A_all,             # [H, K, D]  (H hulls, each with K vertices in R^D)
                                   whiten=False,      # optional per-dim standardization across all vertices
                                       normalize=True,    # L2-normalize centroids -> cosine Gram (recommended)
                                       alpha=1.0,         # strength/scale inside the L-ensemble
                                   eps=1e-6):         # jitter for numerical stability
        """
        DPP prior across local hulls using a Gram (linear/cosine) kernel on hull centroids.
        Returns: log det(alpha*L + eps*I) - log det(I + alpha*L)
        """
        H, K, D = A_all.shape
        if H <= 1:
            return A_all.new_tensor(0.0)
    
        Z = A_all
        if whiten:
            X = Z.reshape(-1, D)
            mu = X.mean(0, keepdim=True)
            std = X.std(0, keepdim=True).clamp_min(1e-6)
            Z = (Z - mu) / std
    
        # Centroids of each hull: [H, D]
        C = Z.mean(dim=1).to(torch.float64)
    
        # Cosine Gram (recommended): L = C_norm @ C_norm^T
        if normalize:
            C = C / C.norm(dim=1, keepdim=True).clamp_min(1e-30)
    
        L = C @ C.T  # PSD Gram
        I = torch.eye(H, dtype=L.dtype, device=L.device)
    
        # L-ensemble DPP objective (paper Eq. (10))
        logdetL  = torch.slogdet(alpha * L + eps * I).logabsdet
        logdetIp = torch.slogdet(I + alpha * L).logabsdet
        return (logdetL - logdetIp).to(A_all.dtype)





    def log_prior_shrink(self, a=1.8, b=6.0, eps=1e-8):
        """
        Beta(a,b) prior on s/eps ∈ (0,1). Returns log prior (constants dropped for MAP).
        Using t = sigmoid(local_shrink), s = eps * t; density ∝ t^{a-1} (1-t)^{b-1}.
        """
        t = torch.sigmoid(self.local_shrink)  # in (0,1)
        return ((a - 1.0) * torch.log(t + eps) + (b - 1.0) * torch.log(1.0 - t + eps)).sum()
    
     
 
    
            
        
        
    #introduce the likelihood function containing the two extra biases gamma_i and alpha_j
    def LSM_likelihood_bias_cs(self,epoch,temperature=1,r=1,Poisson=False):
        '''
        Poisson log-likelihood ignoring the log(k!) constant
        
        '''
        self.epoch=epoch
       
        
        #self.latent_w=self.Softmax(self.latent_w1/temperature)
        M_neg=10*self.sparse_i_idx.shape[0]
        
        
        
        neg_i,neg_j=self.sample_uniform_pairs(self.input_size,M_neg, device=device)
        
        scale=(self.input_size*(self.input_size-1))/neg_i.shape[0]
        self.s=self.softplus(self.s_)
        
       
        
       
        
        if self.epoch==2000:
            # self.gamma.data=0.5*self.bias+self.gamma.data
            self.phase_1=False
            self.pre_epochs=1*self.epoch
            
       
        if self.epoch==200:
             # self.gamma.data=0.5*self.bias+self.gamma.data
            self.scaling=0
        
      
        if self.phase_1:
            #self.scaling=0
            if self.scaling:
                
    
               
                
                
    
              
                #self.gamma=self.gammas[layer]
                #z_pdist1=0.5*torch.mm(torch.exp(self.gammas[layer].unsqueeze(0)),(torch.mm((mat-torch.diag(torch.diagonal(mat))),torch.exp(self.gammas[layer]).unsqueeze(-1))))
                
                if Poisson:
                    mat=torch.exp(self.g[neg_i]+self.g[neg_j])
            
                
                else:
                    mat=self.softplus(self.g[neg_i]+self.g[neg_j]+self.b0)
                
    
    
                #z_pdist1=0.5*torch.mm(torch.exp(self.gammas[layer].unsqueeze(0)),(torch.mm((mat-torch.diag(torch.diagonal(mat))),torch.exp(self.gammas[layer]).unsqueeze(-1))))
                
                z_pdist1=scale*mat.sum()#(mat-torch.diag(torch.diagonal(mat))).sum()
    
                #take the sampled matrix indices in order to index gamma_i and alpha_j correctly and in agreement with the previous
                #z_pdist2=(self.gammas[layer][sparse_i_]+self.gammas[layer][sparse_j_]).sum()
                z_pdist2=((self.g[self.sparse_i_idx]+self.g[self.sparse_j_idx])+self.b0).sum()
    
                
    
               
                log_likelihood_sparse=z_pdist2-z_pdist1
           
            
    
        
               
                    #self.L_.data=(self.gamma.view(-1)+self.delta.view(-1)).abs().max()
                    # self.latent_z.data=self.latent_z.data*self.scaling_factor.data
                return log_likelihood_sparse
    
            else:
                self.latent_z_=self.Softmax(self.latent_z1)

                
              
               
                #self.latent_z, self.log_prior_A,self.A = self.project_A_archetypes_roworth(self.latent_z_,self.A_free,self.tau_free)
                            
                self.A=self.A_svd_boxed()
                self.latent_z=self.latent_z_@self.A

                
                alpha_node = .5 # good default for blocky communities
                self.log_p_z = (((alpha_node - 1.0) *
                            torch.log(self.latent_z_.clamp_min(1e-6))
                          ).sum(dim=1).sum())  +  self.dpp_prior_within_hull(self.A) # average over nodes
                      
                # pol_1=(1/4)*((((((self.latent_z[neg_i])+(self.latent_z[neg_j]+1e-06))**2).sum(-1)))-(((((self.latent_z[neg_i])-(self.latent_z[neg_j]+1e-06))**2).sum(-1))))
                # mat_0=self.s* pol_1+(self.g[neg_i]+self.g[neg_j])#+(self.B_free[alloc[neg_i]]+self.B_free[alloc[neg_j]])#+(self.loc_b[sample_idx].unsqueeze(-1)+self.loc_b[sample_idx])

                mat_0=self.s* (((self.latent_z[neg_i])*(self.latent_z[neg_j]+1e-06)).sum(-1))+(self.g[neg_i]+self.g[neg_j])+self.b0

                
                # pol_2=(1/4)*((((((self.latent_z[self.sparse_i_idx])+(self.latent_z[self.sparse_j_idx]+1e-06))**2).sum(-1)))-(((((self.latent_z[self.sparse_i_idx])-(self.latent_z[self.sparse_j_idx]+1e-06))**2).sum(-1))))
                # mat_1=self.s*pol_2+((self.g[self.sparse_i_idx]+self.g[self.sparse_j_idx]))#+(self.B_free[alloc[self.sparse_i_idx]]+self.B_free[alloc[self.sparse_j_idx]])#+((self.loc_b[sparse_i_sample]+self.loc_b[sparse_j_sample]))

                mat_1=self.s*(((self.latent_z[self.sparse_i_idx])*(self.latent_z[self.sparse_j_idx]+1e-06)).sum(-1))+((self.g[self.sparse_i_idx]+self.g[self.sparse_j_idx]))+self.b0
          
    
                        
                if Poisson:
                    mat=torch.exp(mat_0)
                else:
                    mat=self.softplus(mat_0)
                z_pdist2=mat_1.sum()
                z_pdist1=scale*mat.sum()
    
                log_likelihood_sparse=z_pdist2-z_pdist1 
    
    
                return log_likelihood_sparse
    
            
            
            
        else:
            self.log_p_z=0
            
            self.A=self.A_svd_boxed()


            
          

            self.b_hull=F.softmax(self.local_hull_con,-1)
            
            alpha_node=1.
            self.b_hull_loc =(((alpha_node - 1.0) *
                        torch.log(self.b_hull.clamp_min(1e-6))
                      ).sum(dim=-1).sum())  
            
            
            
            # --- Anchor-dominant barycentric rows (guaranteed disjoint if eps<0.5) ---
            self.b_hull_total = self.build_anchor_dominant_B(self.anchor_eps)  # [K, K, K]
            # Note: no stabilized_weights() here; anchor-dominance already constrains rows
            
            # propagate to local vertices in latent space
            self.A_all = self.b_hull_total @ self.A      # [K, K, D]
            
                        
            
            
           
            
            self.A_all=self.b_hull_total@self.A
            
          

            
            # logits -> centered & normalized for cosine Gram
                          # [K, K-1, K]
            #z_n = logits_centered / (logits_centered.norm(dim=-1, keepdim=True) + 1e-8)    # [K, K-1, K]
          
            slogdets = []
            
            for k in range(self.K):
                
                
                slogdets.append(self.dpp_prior_within_hull(self.A_all[k]))
                

            dpp_logprior_within_hull = torch.stack(slogdets).sum()   # sum over hulls (no mean)

           
            log_p_shrink = self.log_prior_shrink(a=1.2, b=7.)


            self.log_p_b_loc = log_p_shrink+dpp_logprior_within_hull+  self.dpp_prior_within_hull(self.A) #+ self.dpp_across_hulls_centroid_gram(self.A_all)
           
                       
                        
            def T_sched(epoch, start=2000, end=6500, Thi=2.0, Tlo=0.3):
                if epoch <= start: return Thi
                if epoch >= end:   return Tlo
                t = (epoch - start) / (end - start)
                # cosine ramp
                return Tlo + 0.5*(Thi - Tlo)*(1 + math.cos(math.pi * t))
            # replace tt with:
            tt = T_sched(epoch)
            
         
           
            if epoch<6500:
            
                self.m=F.gumbel_softmax(self.latent_z1, tt, hard=False,dim=1)   
            else:
                self.m=F.gumbel_softmax(self.latent_z1, tt, hard=True,dim=1)   
                
                
            

            # self.m=self.Softmax(self.latent_z1)         # [N, C]

            
            alloc=self.m.argmax(1)

            self.latent_z_loc=self.Softmax(self.memberships_loc)
            
            if epoch<6000:
                alpha_node=1.
            
            elif epoch<7000:
                alpha_node =1.
            else:
                alpha_node =1.

            self.log_p_z_loc =(((alpha_node - 1.0) *
                        torch.log(self.latent_z_loc.clamp_min(1e-6))
                      ).sum(dim=1).sum())  
            

            #self.loc_b=self.B_free[self.m.argmax(1)]
            
            #self.latent_z = torch.einsum('nk,ked,nd->ne', self.m, self.A_all, self.latent_z_loc)
            self.latent_z = torch.einsum('nk,kld,nl->nd', self.m, self.A_all, self.latent_z_loc)
            
            # pol_1=(1/4)*((((((self.latent_z[neg_i])+(self.latent_z[neg_j]+1e-06))**2).sum(-1)))-(((((self.latent_z[neg_i])-(self.latent_z[neg_j]+1e-06))**2).sum(-1))))
            # mat_0=self.s* pol_1+(self.g[neg_i]+self.g[neg_j])#+(self.B_free[alloc[neg_i]]+self.B_free[alloc[neg_j]])#+(self.loc_b[sample_idx].unsqueeze(-1)+self.loc_b[sample_idx])

            mat_0=self.s* (((self.latent_z[neg_i])*(self.latent_z[neg_j]+1e-06)).sum(-1))+(self.g[neg_i]+self.g[neg_j])+self.b0#+(self.B_free[alloc[neg_i]]+self.B_free[alloc[neg_j]])#+(self.loc_b[sample_idx].unsqueeze(-1)+self.loc_b[sample_idx])
            
            
            # pol_2=(1/4)*((((((self.latent_z[self.sparse_i_idx])+(self.latent_z[self.sparse_j_idx]+1e-06))**2).sum(-1)))-(((((self.latent_z[self.sparse_i_idx])-(self.latent_z[self.sparse_j_idx]+1e-06))**2).sum(-1))))
            # mat_1=self.s*pol_2+((self.g[self.sparse_i_idx]+self.g[self.sparse_j_idx]))#+(self.B_free[alloc[self.sparse_i_idx]]+self.B_free[alloc[self.sparse_j_idx]])#+((self.loc_b[sparse_i_sample]+self.loc_b[sparse_j_sample]))

            mat_1=self.s*(((self.latent_z[self.sparse_i_idx])*(self.latent_z[self.sparse_j_idx]+1e-06)).sum(-1))+((self.g[self.sparse_i_idx]+self.g[self.sparse_j_idx]))+self.b0#+(self.B_free[alloc[self.sparse_i_idx]]+self.B_free[alloc[self.sparse_j_idx]])#+((self.loc_b[sparse_i_sample]+self.loc_b[sparse_j_sample]))
            
            
                    
            if Poisson:
                mat=torch.exp(mat_0)
            else:
                mat=self.softplus(mat_0)
            z_pdist2=mat_1.sum()
            z_pdist1=scale*mat.sum()

            log_likelihood_sparse=z_pdist2-z_pdist1# + log_p_z#-(0.001*deg_[layer]*((self.latent_raa_z[:,layer]-1)**2)).sum()




           
            return log_likelihood_sparse
        
  
    
            
   
    
    def clustering(self,true_labels):
        
        with torch.no_grad():
            pred_labels=F.one_hot(self.latent_z1.argmax(1), num_classes=self.K).argmax(1).cpu().numpy()
        
           
           # tt=0.3
           # pred_labels=F.gumbel_softmax(self.latent_z1, tt, hard=True,dim=1).argmax(1).numpy()
            
        # Compute NMI
        nmi = normalized_mutual_info_score(true_labels, pred_labels)
        
        # Compute ARI
        ari = adjusted_rand_score(true_labels, pred_labels)



            

           
        return nmi,ari

    def clustering_spherical(self, true_labels, k=None, use_prev_centers=None):
        if k is None:
            k = self.K  # or k_labs if you want ground-truth K
        with torch.no_grad():
            Z = self.latent_z.detach().cpu().numpy()
        # L2-normalize rows (spherical geometry)
        Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)

        km = KMeans(
            n_clusters=k,
            init=use_prev_centers if use_prev_centers is not None else "k-means++",
            n_init=1 if use_prev_centers is not None else 50,
            random_state=0,  # deterministic
            algorithm="elkan"  # faster & often more stable for well-separated clusters
        )
        pred = km.fit_predict(Z)
        nmi = normalized_mutual_info_score(true_labels, pred)
        ari = adjusted_rand_score(true_labels, pred)
        return nmi, ari

    
    
    def clustering_(self,true_labels):

    
    
        with torch.no_grad():
            Z = self.latent_z.detach().cpu().numpy()  
            scaler = StandardScaler().fit(Z)
            Z = scaler.transform(Z)
        kmeans = KMeans(n_clusters=k_labs, n_init=50, random_state=SEED,algorithm='lloyd')
        pred_km = kmeans.fit_predict(Z)
        nmi_km = normalized_mutual_info_score(true_labels, pred_km)
        ari_km = adjusted_rand_score(true_labels, pred_km)

        return nmi_km,ari_km
    
    def clustering_kmeans(self, true_labels, use_prototypes_init=True, use_sample_weight=True, standardize=True):
        with torch.no_grad():
            Z = self.latent_z.detach().cpu().numpy()
            A = self.A.detach().cpu().numpy()      # [K, D]
            conf = torch.softmax(self.latent_z1, dim=1).max(1).values.cpu().numpy()

        if standardize:
            scaler = StandardScaler().fit(Z)
            Z = scaler.transform(Z)
            A0 = scaler.transform(A) if use_prototypes_init else None
        else:
            A0 = A if use_prototypes_init else None

        init = A0 if use_prototypes_init else 'k-means++'
        n_init = 1 if isinstance(init, np.ndarray) else 50

        km = KMeans(n_clusters=self.K, init=init, n_init=n_init, random_state=42)
        pred = km.fit_predict(Z, sample_weight=(conf if use_sample_weight else None))
        nmi = normalized_mutual_info_score(true_labels, pred)
        ari = adjusted_rand_score(true_labels, pred)
        return nmi, ari
    
    
    


        
        
    


from torch.distributions.dirichlet import Dirichlet

import sys





if __name__ == "__main__":
    @torch.no_grad()
    def init_biases_from_degrees(model, sparse_i, sparse_j, undirected=True, eps=1.0):
        N = model.input_size
        deg = torch.zeros(N, device=sparse_i.device)
        deg.index_add_(0, sparse_i, torch.ones_like(sparse_i, dtype=deg.dtype))
        deg.index_add_(0, sparse_j, torch.ones_like(sparse_j, dtype=deg.dtype))
        if undirected:
            deg = deg / 2  # your edges are doubled (i->j and j->i)
        p = (deg + eps) / (N - 1 + 2*eps)
        logit = torch.log(p) - torch.log1p(-p)
        model.g.data.copy_(logit - logit.mean())
    
    # call once after model creation:
    
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
                C[a, b] = ((ci==a) & (cj==b)).sum()
        C = 0.5*(C + C.T)
    
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
        C  = (Z0.T @ Z0) + 1e-6*torch.eye(model.K, device=Z0.device)
        U, _, Vt = torch.linalg.svd(C)
        # map to boxed singulars
        s = 0.3 + (2.0 - 0.3) * torch.sigmoid(model.sig_free)
        A0 = (U * s.unsqueeze(0)) @ Vt
        model.Hu.data.copy_(U)
        model.Hv.data.copy_(Vt.T)
        # Put some structure: encourage near-identity early
        model.sig_free.data.copy_(torch.logit((s - 0.3)/(2.0 - 0.3)))
        
    @torch.no_grad()
    def init_A_by_least_squares(model):
        # Target “coordinates” from top-K spectral components (recentered)
        Z0 = torch.softmax(model.latent_z1, dim=1)             # [N,K]
        X  = model.spectral_data                                # [N,K] already K comps
        X  = X - X.mean(0, keepdim=True)
    
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
        sig = sig.clamp(1e-3, 1-1e-3)
        model.sig_free.data.copy_(torch.logit(sig))
        
    @torch.no_grad()
    def init_anchor_dominant_rows(model, target_s=0.05):
        # s = eps * sigmoid(t) -> choose t to make s ≈ target_s
        # eps = model.anchor_eps (e.g., 0.49). Solve for t:
        # sigmoid(t) = target_s / eps
        eps = float(model.anchor_eps)
        t = torch.logit(torch.tensor(min(0.99, max(0.01, target_s/eps)),
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
    
        p0 = (deg.sum() / (N*(N-1))).clamp(1e-8, 1-1e-8)
        p_i = (deg + alpha * p0) / ((N - 1) + alpha)
        p_i = p_i.clamp(1e-8, 1-1e-8)
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
        c0 = torch.argmax((Xsub**2).sum(dim=1)).item()
        centers.append(Xsub[c0:c0+1])
        d2 = torch.cdist(Xsub, centers[0])**2  # [M,1]
        for _ in range(1, K):
            probs = (d2.min(dim=1).values + 1e-12)
            probs = probs / probs.sum()
            new_idx = torch.multinomial(probs, 1).item()
            centers.append(Xsub[new_idx:new_idx+1])
            d2 = torch.minimum(d2, torch.cdist(Xsub, centers[-1])**2)
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
        p_obs = model.sparse_i_idx.numel() / float(N*(N-1))
        # sample pairs i != j
        i = torch.randint(0, N, (samples,), device=dev)
        j = torch.randint(0, N-1, (samples,), device=dev); j = j + (j >= i)
        gi_gj = (model.g[i] + model.g[j]).detach()
        lo, hi = -10.0, 10.0          # bisection; monotone in b0
        for _ in range(35):
            mid = 0.5*(lo+hi)
            m = torch.sigmoid(mid + gi_gj).mean().item()
            if m > p_obs: hi = mid
            else:         lo = mid
        model.b0.data.fill_(0.5*(lo+hi))
        
        
        
    
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
        d2 = ((X - c)**2).sum(1)                   # [N]
        idx0 = torch.argmax(d2).item()
        anchors = [idx0]
        # iteratively add farthest from current set
        # maintain min distance to any selected anchor
        min_d2 = ((X - X[idx0])**2).sum(1)        # [N]
        for _ in range(1, K):
            # exclude already picked
            min_d2[anchors] = -1.0
            nxt = torch.argmax(min_d2).item()
            anchors.append(nxt)
            # update min distances
            d2_new = ((X - X[nxt])**2).sum(1)
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
            mid = 0.5*(lo + hi)
            L = - D2 / mid
            P = F.softmax(L, dim=1)
            H = -(P.clamp_min(1e-9).log() * P).sum(1).mean()
            if H > Ht:       # too diffuse -> decrease tau (sharpen)
                hi = mid
            else:            # too sharp -> increase tau (soften)
                lo = mid
        tau = 0.5*(lo + hi)
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
#        init_A_by_least_squares(model)  # you already defined this helper
        calibrate_b0_to_density(model)
   
        # (f) optional: initialize community intercepts from assignments (no randomness)
        init_B_from_assignments(model, model.sparse_i_idx, model.sparse_j_idx)
    
        # (g) locals: tight, non-overlapping by construction (keeps phase-1 stable)
        init_anchor_dominant_rows(model, target_s=0.05)  # small spread around anchor
        model.anchor_eps = 0.4                      # < 0.5, gives a bit of room later
        
   
            
            
            
            

   



    

   


    psis=[1.]
    
    for psi in psis:
        dataset='lastfm_with_labels'
   

        sparse_i_=torch.from_numpy(np.loadtxt(f'./datasets/{dataset}/sparse_i.txt')).long().to(device)     
        sparse_j_=torch.from_numpy(np.loadtxt(f'./datasets/{dataset}/sparse_j.txt')).long().to(device)     
    
        sparse_i=torch.cat((sparse_i_,sparse_j_))
        sparse_j=torch.cat((sparse_j_,sparse_i_))
        
        labels_=np.loadtxt(f'./datasets/{dataset}/labels.txt').astype(int)
        k_labs=int(max(labels_)+1)

       
    
    
        
        N=int(max(sparse_i.max(),sparse_j.max())+1)
        
        priors=[]
        s=[]
       
        min_loss_RE=1000000
    
        epoch_num=10000
        runs=1
        for run in range(runs):
            
    
    
            print("RUN number:",run)
           
            sample_size=int(0.3*N)
            
            # Missing_data should be set to False for link_prediction since we do not consider these interactions as missing but as zeros.
            model = LSM(number_of_K=k_labs,latent_dim=k_labs ,k_extra=None,sparse_i=sparse_i,sparse_j=sparse_j, input_size1=N,input_size2=N,sample_size=sample_size).to(device)         
            init_phase1_stable(model)

            
            nu=model.D+2
            
            d = model.D
            sigma_star = 0.7                # your desired marginal std per dim
            Psi = (nu + d + 1) * (sigma_star**2) * torch.eye(d)
            
            
            
            

    
            optimizer1 = torch.optim.Adam(model.parameters(), lr=5e-2, betas=(0.9, 0.99), weight_decay=1e-4)

            
            
            #optimizer1 = torch.optim.AdamW(model.parameters(), lr=0.05, weight_decay=1e-4)

    
            dyads=(N*(N-1))
            
            num_pairs = model.K * (model.K - 1) // 2

    
            #for name, param in model.named_parameters():
            #    print(f"{name}: {param.device}")
                
            colors=np.array(["green","blue","red"])
            losses=[]
            #sampling=True    
            

            for epoch in range(epoch_num):
                if epoch==8000:
                    model.sample_size=int(0.3*N)
                    
               
                
                              
                if model.scaling:
    
    
                    loss1=-model.LSM_likelihood_bias_cs(epoch=epoch)-model.log_prior_bias()
    
                else:   
                        
                       
                    if model.phase_1:
                        loss1=-model.LSM_likelihood_bias_cs(epoch=epoch)+0.5*(model.s**2)- model.log_p_z-model.log_prior_bias()
                    if not model.phase_1:
                        
                        
                        
                                                
                        def sched_linear(epoch, start, end, hi, lo):
                            if epoch <= start: return hi
                            if epoch >= end:   return lo
                            t = (epoch - start) / max(1, end - start)
                            return hi + (lo - hi) * t
                        
                      
                        
                        N = model.input_size            # total nodes
                        s = model.sample_size           # sampled nodes this step
                        dyads_full  = N * (N - 1)
                        dyads_batch = s * (s - 1)
                        scale = dyads_full / max(1, dyads_batch)
                        
                                                
                        like_term = -model.LSM_likelihood_bias_cs(epoch=epoch) 
                        
                        

                        priors    = (0.5*((model.s**2)/(1**2)) - model.log_p_z - model.log_prior_bias()
                                     - model.log_p_z_loc - model.log_p_b_loc#-model.b_hull_loc
                                     )
                        
                        loss1 = like_term + priors

    
                
                optimizer1.zero_grad()
                loss1.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)   # (see §4)
                optimizer1.step()

                
                if epoch%100==0:
                    if not model.scaling:
                        print(loss1)
                        
                        if model.K==k_labs:
                            NMI,ARI=model.clustering(labels_)
                            
                            print(f'epoch ={epoch} ---- NMI = {NMI} ---- ARI = {ARI}')
                        if model.K!=k_labs:

                            NMI,ARI=model.clustering_spherical(labels_)
                        
                            print(f'epoch ={epoch} ---- NMI_kmeans = {NMI} ---- ARI_kmeans = {ARI}')
                        
                      
                   



                
               
                      
                
                current_lr = optimizer1.param_groups[0]['lr']
                #print(f"Epoch {epoch}: Current LR = {current_lr:.5f}")
                min_lr_threshold=1e-07
                # Early stop if learning rate is below threshold
                if current_lr < min_lr_threshold:
                    print(f"Stopping early at epoch {epoch} as learning rate dropped below {min_lr_threshold}")
                    break
                if epoch==2000:
                    #scheduler.step(loss1)
                    optimizer1.param_groups[0]['lr']=1e-02
                  # current_lr = optimizer1.param_groups[0]['lr']
                  # print(current_lr)
                if epoch==4000:
                    #scheduler.step(loss1)
                    optimizer1.param_groups[0]['lr']=1e-03

                losses.append(loss1.item())
        


