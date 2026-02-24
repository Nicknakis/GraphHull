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


# --- Drop this helper anywhere (outside the class is fine) ---
class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {n: p.detach().clone()
                       for n, p in model.named_parameters() if p.requires_grad}
        self.backup = {}

    @torch.no_grad()
    def update(self, model):
        for n, p in model.named_parameters():
            if not p.requires_grad: continue
            self.shadow[n].mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)

    @torch.no_grad()
    def apply_shadow(self, model):
        self.backup = {}
        for n, p in model.named_parameters():
            if not p.requires_grad: continue
            self.backup[n] = p.detach().clone()
            p.data.copy_(self.shadow[n])

    @torch.no_grad()
    def restore(self, model):
        for n, p in model.named_parameters():
            if not p.requires_grad: continue
            p.data.copy_(self.backup[n])
        self.backup = {}



  


class LSM(nn.Module,Spectral_clustering_init):
    def __init__(self,number_of_K,latent_dim,k_extra,sparse_i,sparse_j, input_size1,input_size2,sample_size,scaling=None):
        super(LSM, self).__init__()
        # initialization
        Spectral_clustering_init.__init__(self,num_of_eig=latent_dim,device=device)

        
        self.D=latent_dim
        self.K=number_of_K


        self.input_size=input_size1
        
        

        self.softmax_loc=nn.Softmax(dim=-1)
       
        self.log_s=None
          
        self.b0= nn.Parameter(torch.zeros(1,device=device))
        self.g=nn.Parameter(torch.randn(input_size1,device=device))
        self.d=nn.Parameter(torch.randn(input_size2,device=device))

        self.logit_sigma = nn.Parameter(torch.zeros(self.K))

        self.scaling=1
        #create indices to index properly the receiver and senders variable
        self.sparse_i_idx=sparse_i
        self.sparse_j_idx=sparse_j
        
        self.spectral_data=self.spectral_clustering()#+1e-1

        
        self.Softmax=nn.Softmax(1)

        
        self.memberships_loc=nn.Parameter(torch.rand(input_size1,self.K,device=device))

        self.k_extra=k_extra
        self.softplus=nn.Softplus()
        self.A_uncon=nn.Parameter(torch.rand(self.K,self.D, device=device))
        
        self.s_dim_=nn.Parameter(torch.rand(self.D, device=device))
        
        self.torch_pi=torch.tensor(math.pi)
        
        #init_pre_softplus = torch.log(torch.exp(torch.tensor(target_value)) - 1)

        self.s_=nn.Parameter(2*torch.ones(1))
        
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
        
        
        self.Hu = torch.nn.Parameter(torch.rand(self.K, self.K))
        self.Hv = torch.nn.Parameter(torch.rand(self.K, self.K) )
        self.sig_free = torch.nn.Parameter(torch.zeros(self.K))
        self.nu_free = torch.nn.Parameter(torch.zeros(1))
        
        self.pi_logits = torch.nn.Parameter(torch.zeros(self.K))   # shape [r]
        
        self.s_free = torch.nn.Parameter(torch.zeros(self.K))
        self.phase_1=True
        
        self.w_free = nn.Parameter(torch.zeros(self.K, self.K, device=device))  # [hull, dim]
        
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
        self.anchor_eps = .49# < 0.5 to guarantee disjoint hulls by construction; you can anneal later
        
        
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


        
        
    
       

 
    def A_svd_boxed(self,Hu, Hv, sig_free, sigma_min=.5, sigma_max=2.5, eps=1e-8):
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
    
       
        
    def log_prior_bias(self, tau_g=5.0):
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
    
    
    
    def dpp_across_hulls_centroid_rbf(self, A_all, whiten=True, tau=None, alpha=0.5, eps=1e-6):
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
       
        if self.epoch==1000:
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
                    mat=self.softplus(self.g[neg_i]+self.g[neg_j])
                
    
    
                #z_pdist1=0.5*torch.mm(torch.exp(self.gammas[layer].unsqueeze(0)),(torch.mm((mat-torch.diag(torch.diagonal(mat))),torch.exp(self.gammas[layer]).unsqueeze(-1))))
                
                z_pdist1=scale*mat.sum()#(mat-torch.diag(torch.diagonal(mat))).sum()
    
                #take the sampled matrix indices in order to index gamma_i and alpha_j correctly and in agreement with the previous
                #z_pdist2=(self.gammas[layer][sparse_i_]+self.gammas[layer][sparse_j_]).sum()
                z_pdist2=((self.g[self.sparse_i_idx]+self.g[self.sparse_j_idx])).sum()
    
                
    
               
                log_likelihood_sparse=z_pdist2-z_pdist1
           
            
    
        
               
                    #self.L_.data=(self.gamma.view(-1)+self.delta.view(-1)).abs().max()
                    # self.latent_z.data=self.latent_z.data*self.scaling_factor.data
                return log_likelihood_sparse
    
            else:
                self.latent_z_=self.Softmax(self.latent_z1)

                
              
               
                #self.latent_z, self.log_prior_A,self.A = self.project_A_archetypes_roworth(self.latent_z_,self.A_free,self.tau_free)
                            
                self.A=self.A_svd_boxed(self.Hu, self.Hv, self.sig_free)
                self.latent_z=self.latent_z_@self.A

                
                alpha_node = .5 # good default for blocky communities
                self.log_p_z = (((alpha_node - 1.0) *
                            torch.log(self.latent_z_.clamp_min(1e-6))
                          ).sum(dim=1).sum())  +  self.dpp_prior_within_hull(self.A) # average over nodes
                  
            
                mat_0=self.s* (((self.latent_z[neg_i])*(self.latent_z[neg_j]+1e-06)).sum(-1))+(self.g[neg_i]+self.g[neg_j])#+self.b0
               
             
                                
                mat_1=self.s*(((self.latent_z[self.sparse_i_idx])*(self.latent_z[self.sparse_j_idx]+1e-06)).sum(-1))+((self.g[self.sparse_i_idx]+self.g[self.sparse_j_idx]))#+self.b0
          
    
                        
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
            
           
            self.A=self.A_svd_boxed(self.Hu, self.Hv, self.sig_free)


            
          

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

           
            log_p_shrink = self.log_prior_shrink(a=1., b=1.)


            self.log_p_b_loc = log_p_shrink+dpp_logprior_within_hull+  self.dpp_prior_within_hull(self.A)
           
                       
                        
            def T_sched(epoch, start=2000, end=5500, Thi=2.0, Tlo=0.35):
                if epoch <= start: return Thi
                if epoch >= end:   return Tlo
                t = (epoch - start) / (end - start)
                # cosine ramp
                return Tlo + 0.5*(Thi - Tlo)*(1 + math.cos(math.pi * t))
            # replace tt with:
            tt = T_sched(epoch)
            
                                              
                        
           
            if epoch<9000:
            
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
                alpha_node =1

            self.log_p_z_loc =(((alpha_node - 1.0) *
                        torch.log(self.latent_z_loc.clamp_min(1e-6))
                      ).sum(dim=1).sum())  
            
           
            
            #self.loc_b=self.B_free[self.m.argmax(1)]
            
            #self.latent_z = torch.einsum('nk,ked,nd->ne', self.m, self.A_all, self.latent_z_loc)
            self.latent_z = torch.einsum('nk,kld,nl->nd', self.m, self.A_all, self.latent_z_loc)

            
            mat_0=self.s* (((self.latent_z[neg_i])*(self.latent_z[neg_j]+1e-06)).sum(-1))+(self.g[neg_i]+self.g[neg_j])#+self.b0#+(self.B_free[alloc[neg_i]]+self.B_free[alloc[neg_j]])#+(self.loc_b[sample_idx].unsqueeze(-1)+self.loc_b[sample_idx])
            #pol_1=(1/4)*((((((self.latent_z[neg_i])+(self.latent_z[neg_j]+1e-06))**2).sum(-1)))-(((((self.latent_z[neg_i])-(self.latent_z[neg_j]+1e-06))**2).sum(-1))))
            #mat_0=self.s* pol_1+(self.g[neg_i]+self.g[neg_j])#+(self.B_free[alloc[neg_i]]+self.B_free[alloc[neg_j]])#+(self.loc_b[sample_idx].unsqueeze(-1)+self.loc_b[sample_idx])

            
            
            #pol_2=(1/4)*((((((self.latent_z[self.sparse_i_idx])+(self.latent_z[self.sparse_j_idx]+1e-06))**2).sum(-1)))-(((((self.latent_z[self.sparse_i_idx])-(self.latent_z[self.sparse_j_idx]+1e-06))**2).sum(-1))))
            #mat_1=self.s*pol_2+((self.g[self.sparse_i_idx]+self.g[self.sparse_j_idx]))#+(self.B_free[alloc[self.sparse_i_idx]]+self.B_free[alloc[self.sparse_j_idx]])#+((self.loc_b[sparse_i_sample]+self.loc_b[sparse_j_sample]))
            
            
            mat_1=self.s*(((self.latent_z[self.sparse_i_idx])*(self.latent_z[self.sparse_j_idx]+1e-06)).sum(-1))+((self.g[self.sparse_i_idx]+self.g[self.sparse_j_idx]))#+self.b0#+(self.B_free[alloc[self.sparse_i_idx]]+self.B_free[alloc[self.sparse_j_idx]])#+((self.loc_b[sparse_i_sample]+self.loc_b[sparse_j_sample]))
            
            
                    
            if Poisson:
                mat=torch.exp(mat_0)
            else:
                mat=self.softplus(mat_0)
            z_pdist2=mat_1.sum()
            z_pdist1=scale*mat.sum()

            log_likelihood_sparse=z_pdist2-z_pdist1# + log_p_z#-(0.001*deg_[layer]*((self.latent_raa_z[:,layer]-1)**2)).sum()




           
            return log_likelihood_sparse
        
  
    
                
   
    
    def link_prediction(self,rem_i,rem_j,target):
        
        with torch.no_grad():

            self.s=self.softplus(self.s_)

            if self.phase_1:
    
                #self.latent_z, _,_ = self.project_A_archetypes_roworth(self.latent_z_,self.A_free, self.tau_free)
                self.A=self.A_svd_boxed(self.Hu, self.Hv, self.sig_free)
                self.latent_z=self.latent_z_@self.A
                
                # Z_norm = self._center_and_normalize(self.latent_z)
                # cos_edges = (Z_norm[rem_i] * Z_norm[rem_j]).sum(dim=-1) 
                
                rates=self.s*(((self.latent_z[rem_i])*(self.latent_z[rem_j]+1e-06)).sum(-1))+(self.g[rem_i]+self.g[rem_j])#+self.b0

            else:
            
             
                
                self.m = F.one_hot(self.latent_z1.argmax(1), num_classes=self.K).float()
                
                #self.m = self.Softmax(self.latent_z1).float()

                
                alloc=self.m.argmax(1)


               
                    
                

                
    
                self.latent_z_loc=self.Softmax(self.memberships_loc)
                
                self.latent_z = torch.einsum('nk,ked,ne->nd', self.m, self.A_all, self.latent_z_loc)
                
                
                # dot_edges = self._metric_dot_per_hull(
                #     self.latent_z[rem_i], self.latent_z[rem_j],
                #     self.m[rem_i],        self.m[rem_j],
                #     combine="prod"
                # )
                # rates = self.s * dot_edges + (self.g[rem_i] + self.g[rem_j])
                
                            
                    
        
                rates=self.s*(((self.latent_z[rem_i])*(self.latent_z[rem_j]+1e-06)).sum(-1))+(self.g[rem_i]+self.g[rem_j])#+self.b0#+(self.B_free[alloc[rem_i]]+self.B_free[alloc[rem_j]])#+(self.loc_b[rem_i]+self.loc_b[rem_j])


            #rates=-self.s*(((((self.latent_z[rem_i])-(self.latent_z[rem_j]+1e-06))**2).sum(-1))**0.5)+(self.g[rem_i]+self.g[rem_j])
            
            
        precision, tpr, thresholds = metrics.precision_recall_curve(target.cpu().data.numpy(), rates.cpu().data.numpy())

           
        return metrics.roc_auc_score(target.cpu().data.numpy(),rates.cpu().data.numpy()),metrics.auc(tpr,precision)
    


         
    


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
        P.fill_diagonal_(n*(n-1))  # ordered pairs with i!=j, approx
    
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
        
   
  

        


    psis=[1.]
    
    for psi in psis:
        dataset='hepth'
   

        sparse_i_=torch.from_numpy(np.loadtxt(f'./datasets/{dataset}/sparse_i.txt')).long().to(device)     
        sparse_j_=torch.from_numpy(np.loadtxt(f'./datasets/{dataset}/sparse_j.txt')).long().to(device)    
        
    
        sparse_i=torch.cat((sparse_i_,sparse_j_))
        sparse_j=torch.cat((sparse_j_,sparse_i_))
        
        
        sparse_i_rem=torch.from_numpy(np.loadtxt(f'./datasets/{dataset}/sparse_i_rem.txt')).long().to(device)     

    
        sparse_j_rem=torch.from_numpy(np.loadtxt(f'./datasets/{dataset}/sparse_j_rem.txt')).long().to(device)     
        
        non_sparse_i_rem=torch.from_numpy(np.loadtxt(f'./datasets/{dataset}/non_sparse_i.txt')).long().to(device)     
    
        non_sparse_j_rem=torch.from_numpy(np.loadtxt(f'./datasets/{dataset}/non_sparse_j.txt')).long().to(device)     
        
        rem_i=torch.cat((sparse_i_rem,non_sparse_i_rem))
        rem_j=torch.cat((sparse_j_rem,non_sparse_j_rem))
        
        target=torch.cat((torch.ones(sparse_i_rem.shape[0]),torch.zeros(non_sparse_i_rem.shape[0])))
    
    
    
        
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
            model = LSM(number_of_K=8,latent_dim=8,k_extra=None,sparse_i=sparse_i,sparse_j=sparse_j, input_size1=N,input_size2=N,sample_size=sample_size).to(device)         
            init_biases_from_degrees(model, sparse_i, sparse_j, undirected=True)
            #init_B_from_assignments(model, sparse_i, sparse_j)
            
            init_memberships_from_spectral(model)
            init_A_from_centroids(model)
            #init_anchor_dominant_rows(model)
        
            
            nu=model.D+2
            
            d = model.D
            sigma_star = 0.7                # your desired marginal std per dim
            Psi = (nu + d + 1) * (sigma_star**2) * torch.eye(d)
            
            
            
            

    
            optimizer1 = optim.Adam(model.parameters(), 0.05)  
            
            
            #optimizer1 = torch.optim.AdamW(model.parameters(), lr=5e-2, weight_decay=1e-4)

    
            dyads=(N*(N-1))
            
            num_pairs = model.K * (model.K - 1) // 2

    
            #for name, param in model.named_parameters():
            #    print(f"{name}: {param.device}")
                
            colors=np.array(["green","blue","red"])
            losses=[]
            #sampling=True    
            
            ema = EMA(model, decay=0.999)

            for epoch in range(epoch_num):
                if epoch==8000:
                    model.sample_size=int(0.3*N)

    
                if model.scaling:
    
    
                    loss1=-model.LSM_likelihood_bias_cs(epoch=epoch)
    
    
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
                        # sep_energy_per_pair = model.sep_loss / max(1, num_pairs)

                        
                        # sep_prior_term =  scale * sep_energy_per_pair
                        
                       
                        

                        priors    = (0.5*(model.s**2) - model.log_p_z - model.log_prior_bias()
                                     - model.log_p_z_loc - model.log_p_b_loc#-model.b_hull_loc
                                     )#+model.lambda_neglogprior)   # sigma already encodes strength → MAP-valid
                        
                        loss1 = like_term + priors

    
                #model_pks.append(1*model.p_k.detach())
                
                optimizer1.zero_grad()
                loss1.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)   # (see §4)
                optimizer1.step()
            
                           
                if epoch%100==0:
                    print(loss1)
                    
                    if not model.scaling:
                        AUC_ROC,AUC_PR=model.link_prediction(rem_i, rem_j, target)
                    
                        print(f'epoch ={epoch} ---- AUC_ROC = {AUC_ROC} ---- AUC_PR = {AUC_PR}')
                        
                        # AUC_ROC,AUC_PR=model.link_prediction_(rem_i, rem_j, target)
                    
                        # print(f'epoch ={epoch} ---- GLASS AUC_ROC = {AUC_ROC} ---- GLASS AUC_PR = {AUC_PR}')
                    
                    # if epoch>6000:
                    #     # Example:
                    #     polys = list(model.A_all.detach().cpu().numpy())
                    #     print(overlapping_pairs(polys))
       



                
               
                      
                
                current_lr = optimizer1.param_groups[0]['lr']
                #print(f"Epoch {epoch}: Current LR = {current_lr:.5f}")
                min_lr_threshold=1e-07
                # Early stop if learning rate is below threshold
                if current_lr < min_lr_threshold:
                    print(f"Stopping early at epoch {epoch} as learning rate dropped below {min_lr_threshold}")
                    break
                if epoch==5000:
                # #    #scheduler.step(loss1)
                     optimizer1.param_groups[0]['lr']=0.01
                    
               

                losses.append(loss1.item())
                
                 
        polys = list(model.A_all.detach().numpy())
        print(overlapping_pairs(polys))
        

        
           
    AUC_ROC,AUC_PR=model.link_prediction(rem_i, rem_j, target)
    print(f'epoch ={epoch} ---- AUC_ROC = {AUC_ROC} ---- AUC_PR = {AUC_PR}')

    
    adj=torch.zeros(N,N)
    adj[sparse_i,sparse_j]=1
            
    
    
    mask=torch.arange(0,model.K**2,model.K)
    
    idx=model.latent_z_.argmax(1)
    
    idx=(mask[idx]+model.latent_z_loc.argmax(1)).argsort()
    
    plt.spy(adj[:,idx][idx,:],markersize=0.1)
    plt.show()
    
    
    with torch.no_grad():
        z_now = torch.softmax(model.latent_z1/0.3, dim=1)   # low T for eval only
        labels = torch.argmax(z_now, 1)
        print(torch.bincount(labels, minlength=model.K))     # class sizes
    
        
    
    
    
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.decomposition import PCA
    
    # Perform PCA
    pca = PCA(n_components=2)
    
    X = model.latent_z.detach().cpu().numpy()
    A_ = model.A.detach().cpu().numpy()
    all_A = model.A_all.view(-1, model.K).detach().cpu().numpy()
    
    pca.fit(np.concatenate((all_A, A_, X)))
    
    A_proj = pca.transform(A_)
    X_proj = pca.transform(all_A)
    latent_proj = pca.transform(X)
    
    # Print explained variance
    print("Explained variance ratio:", pca.explained_variance_ratio_)
    print("Singular values:", pca.singular_values_)
    
    # Plot
    plt.figure(figsize=(10, 8))
    
    # Latent space projections
    scatter1 = plt.scatter(
        latent_proj[:, 0], latent_proj[:, 1], 
        c=model.m.argmax(1).detach().cpu().numpy(), 
        cmap='tab10', alpha=0.7, s=40, label="Latent Z"
    )
    
    # All A points
    plt.scatter(
        X_proj[:, 0], X_proj[:, 1], 
        c='green', alpha=0.6, s=25, label="A_all"
    )
    
    # Current A
    plt.scatter(
        A_proj[:, 0], A_proj[:, 1], 
        c='black', marker='X', s=100, label="A"
    )
    
    # Add colorbar, labels, and legend
    plt.colorbar(scatter1, label="Cluster assignment (argmax of m)")
    plt.xlabel("PCA Component 1", fontsize=12)
    plt.ylabel("PCA Component 2", fontsize=12)
    plt.title("PCA Projection of Latent Space and A Matrices", fontsize=14, fontweight='bold')
    plt.legend(frameon=True)
    plt.grid(True, linestyle="--", alpha=0.5)
    
    plt.show()
    
        
    
    with torch.no_grad():
        # per-hull log-volume proxy
        vols = []
        for c in range(model.A_all.size(0)):
            B = model.A_all[c]; a = B[-1]
            E = B[:-1] - a.unsqueeze(0)
            G = E @ E.T + 1e-6*torch.eye(E.size(0), device=B.device)
            vols.append(float(0.5*torch.logdet(G).cpu()))
        print("log-volume per hull:", vols)
    
        # check for duplicate local vertices
        for c in range(model.A_all.size(0)):
            Dm = torch.cdist(model.A_all[c], model.A_all[c])
            Dm = Dm + 1e9*torch.eye(Dm.size(0), device=Dm.device)
            print(f"min pairwise dist in hull {c}:", float(Dm.min().cpu()))
    
    
    
    import numpy as np
    from cvxopt import matrix, solvers
    
    solvers.options['show_progress'] = False
    
    import numpy as np
    from cvxopt import matrix, solvers
    
    solvers.options['show_progress'] = False
    
    def _cvx(x):
        return None if x is None else matrix(x)
    
    import numpy as np
    from cvxopt import solvers, matrix
    
    def _cvx(x):
        return None if x is None else matrix(x)
    
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
        vL = V[Lv-1]
        wL = W[Lw-1]
    
        # Build equality system:  V^T alpha = W^T beta  with alpha_Lv = 1 - sum(a), beta_Lw = 1 - sum(b)
        # M z = rhs, where z = [a (Lv-1); b (Lw-1)]
        A_V = V[:Lv-1].T - vL[:, None]     # (D, Lv-1)
        A_W = W[:Lw-1].T - wL[:, None]     # (D, Lw-1)
        M   = np.hstack([A_V, -A_W])       # (D, (Lv-1)+(Lw-1))
        rhs = (wL - vL)                    # (D,)
        n   = M.shape[1]
    
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
            G[n, :Lv-1] = 1.0; h[n] = 1.0
        else:
            # No a-variables; inequality is vacuous
            pass
        if Lw > 1:
            G[n+1, Lv-1:] = 1.0; h[n+1] = 1.0
        else:
            # No b-variables
            pass
    
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
        a = z[:max(Lv-1, 0)]
        b = z[max(Lv-1, 0):]
    
        alpha = np.concatenate([a, [1.0 - a.sum()]]) if Lv > 0 else np.zeros((0,))
        beta  = np.concatenate([b, [1.0 - b.sum()]]) if Lw > 0 else np.zeros((0,))
    
        # Numerical hygiene: small negatives → 0, then renormalize
        if alpha.min() < -tol or beta.min() < -tol:
            return False
        alpha = np.maximum(alpha, 0.0);  beta = np.maximum(beta, 0.0)
        sa, sb = alpha.sum(), beta.sum()
        if sa <= tol or sb <= tol:
            return False
        alpha /= sa; beta /= sb
    
        # Final geometric check in original space
        p = V.T @ alpha
        q = W.T @ beta
        return np.linalg.norm(p - q) <= tol
    
        
    
    K, L, D = model.A_all.shape
    A_all_np = model.A_all.detach().cpu().numpy()
    overlap = np.zeros((K, K), dtype=bool)
    for i in range(K):
        for j in range(i+1, K):
            overlap[i, j] = hulls_intersect_lp(A_all_np[i], A_all_np[j])
            overlap[j, i] = overlap[i, j]
    print("pairwise hull intersection (True = overlap):\n", overlap)
    
    
        
    
    from cvxopt import matrix, solvers
    solvers.options['show_progress'] = False
    
    import numpy as np
    from cvxopt import matrix, solvers
    solvers.options['show_progress'] = False
    solvers.options.update({'abstol':1e-9,'reltol':1e-9,'feastol':1e-9,'maxiters':100})
    
    def hull_min_distance_qp(V, W, tol=1e-7, ridge=1e-10):
        """
        Min distance between conv(V) and conv(W).
        V: [L1, D], W: [L2, D]  (numpy float64)
        Returns a nonnegative float (0 means intersect/touch up to tol).
        """
        V = np.asarray(V, dtype=float); W = np.asarray(W, dtype=float)
        L1, D1 = V.shape; L2, D2 = W.shape
        assert D1 == D2, "V and W must have same dim"
    
        VV = V @ V.T            # [L1, L1]
        WW = W @ W.T            # [L2, L2]
        VW = V @ W.T            # [L1, L2]
    
        # x = [alpha; beta] ∈ R^{L1+L2}, objective 0.5 * x^T P x
        P = np.block([[ VV,    -VW],
                      [-VW.T,   WW]]).astype(float)
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
        beta  = x[L1:]
    
        # Project to simplices defensively
        alpha = np.clip(alpha, 0, None); s = alpha.sum();  alpha = alpha/s if s>0 else np.full(L1, 1.0/L1)
        beta  = np.clip(beta, 0, None);  s = beta.sum();   beta  = beta/s  if s>0 else np.full(L2, 1.0/L2)
    
        # Witness points and distance
        p = V.T @ alpha   # [D]
        q = W.T @ beta    # [D]
        d = float(np.linalg.norm(p - q))
        return 0.0 if d <= tol else d
    
    K, L, D = model.A_all.shape
    A_all_np = model.A_all.detach().cpu().numpy()
    min_d = np.full((K, K), np.nan)
    for i in range(K):
        for j in range(i+1, K):
            d = hull_min_distance_qp(A_all_np[i], A_all_np[j])
            min_d[i, j] = min_d[j, i] = d
    print("pairwise hull margins:\n", min_d)
    print("global min margin:", np.nanmin(min_d[np.triu_indices(K, 1)]))
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.spatial import ConvexHull
    from sklearn.decomposition import PCA
    
    # get A_all from your model
    A_all = model.A_all.detach().cpu().numpy()   # [K, L, D]
    
    # project everything to 2D for visualization
    pca = PCA(n_components=2)
    A_all_2d = pca.fit_transform(A_all.reshape(-1, A_all.shape[-1]))
    A_all_2d = A_all_2d.reshape(A_all.shape[0], A_all.shape[1], 2)  # [K, L, 2]
    
    # plot each hull
    plt.figure(figsize=(8,8))
    colors = ["red", "blue", "green", "orange", "purple"]
    
    for k in range(A_all.shape[0]):
        P = A_all_2d[k]  # [L,2]
        hull = ConvexHull(P)
        # draw filled hull
        plt.fill(P[hull.vertices,0], P[hull.vertices,1], alpha=0.3, color=colors[k % len(colors)])
        # draw vertices
        plt.scatter(P[:,0], P[:,1], color=colors[k % len(colors)], s=30, edgecolor="k")
    
    plt.title("Convex hulls from model.A_all")
    plt.axis("equal")
    plt.show()
    
        
        
        
        
        
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.spatial import ConvexHull
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    
    # A_all: [K, L, D]
    A = model.A_all.detach().cpu().numpy()
    K, L, D = A.shape
    
    # Build data and labels
    X = A.reshape(-1, D)                    # [K*L, D]
    y = np.repeat(np.arange(K), L)          # [K*L]
    
    # LDA to 2D (max components = min(D, K-1); need K>=3 for full 2D)
    n_comp = min(2, max(1, K-1))
    proj = LDA(n_components=n_comp).fit_transform(X, y)
    P2 = proj.reshape(K, L, n_comp)
    
    plt.figure(figsize=(8, 8))
    colors = ["red", "blue", "green", "orange", "purple"]
    
    for k in range(K):
        P = P2[k]  # [L, 2] if n_comp==2; handle 1D case by padding
        if n_comp == 1:
            P = np.c_[P[:, 0], np.zeros_like(P[:, 0])]
    
        hull = ConvexHull(P)
        plt.fill(P[hull.vertices, 0], P[hull.vertices, 1],
                 alpha=0.25, color=colors[k % len(colors)])
        plt.scatter(P[:, 0], P[:, 1],
                    s=30, c=colors[k % len(colors)], edgecolors="black", linewidths=0.5)
        
        plt.title("Convex hulls via LDA projection")
        plt.axis("equal")
        plt.tight_layout()
        plt.show()
    
        
    from sklearn.decomposition import PCA

    A = model.A_all.detach().cpu().numpy()  # [K, L, D]
    K, L, D = A.shape
    colors = ["red", "blue", "green", "orange", "purple"]
    
    fig, axes = plt.subplots(1, K, figsize=(4*K, 4))
    if K == 1:
        axes = [axes]
    
    for k in range(K):
        pca_k = PCA(n_components=2).fit(A[k])
        P = pca_k.transform(A[k])  # [L, 2]
        hull = ConvexHull(P)
        ax = axes[k]
        ax.fill(P[hull.vertices, 0], P[hull.vertices, 1], alpha=0.25, color=colors[k % len(colors)])
        ax.scatter(P[:, 0], P[:, 1], s=30, c=colors[k % len(colors)], edgecolors="black", linewidths=0.5)
        ax.set_title(f"Cluster {k} (own PCA)")
        ax.axis("equal")
    
    plt.tight_layout()
    plt.show()
    
    
       
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.spatial import ConvexHull
    from matplotlib.colors import ListedColormap, BoundaryNorm
    
    # --- Labels and K ---
    labels = model.m.argmax(1).detach().cpu().numpy()
    K_detected = int(labels.max() + 1)
    
    # --- Build a big qualitative palette (distinct categorical colors) ---
    def make_distinct_palette(K):
        # Grab colors from multiple qualitative colormaps
        cmaps = [
            plt.cm.tab10, plt.cm.Dark2, plt.cm.Set1, plt.cm.Set2, plt.cm.Set3,
            plt.cm.Paired, plt.cm.tab20, plt.cm.tab20b, plt.cm.tab20c
        ]
        cols = []
        for cm in cmaps:
            n = getattr(cm, 'N', 256)
            # Sample evenly across each qualitative map
            take = 10 if cm in (plt.cm.tab10, plt.cm.Dark2, plt.cm.Set1, plt.cm.Set2, plt.cm.Set3, plt.cm.Paired) else 20
            idxs = np.linspace(0, n - 1, take, dtype=int)
            cols.extend([cm(i) for i in idxs])
            if len(cols) >= K:
                break
        # Fallback if still short: evenly sample HSV
        if len(cols) < K:
            hsv = plt.cm.hsv(np.linspace(0, 1, K - len(cols), endpoint=False))
            cols.extend([tuple(c) for c in hsv])
        return cols[:K]
    
    
    palette = make_distinct_palette(K_detected)
    cmap_cat = ListedColormap(palette)
    norm_cat = BoundaryNorm(np.arange(K_detected + 1) - 0.5, K_detected)
    
    # --- Reshape local hull vertices (projected) for per-community colors ---
    K_hulls, L = model.A_all.shape[:2]
    B_proj = X_proj.reshape(K_hulls, L, 2)
    
    # --- Plot ---
    plt.figure(figsize=(10, 8))
    
    # Latent Z colored by cluster id with categorical cmap
    scatter1 = plt.scatter(
        latent_proj[:, 0], latent_proj[:, 1],
        c=labels, cmap=cmap_cat, norm=norm_cat,
        alpha=0.75, s=40, linewidths=0.2, edgecolors='k',
        label="Node embeddings $z_i$"
    )
    
    # Shade each local hull with its community color (semi-transparent)
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
            # outline for clarity
            plt.plot(
                np.r_[polyc[:, 0], polyc[0, 0]],
                np.r_[polyc[:, 1], polyc[0, 1]],
                color=palette[c % K_detected], lw=1.0, zorder=1
            )
    
    # All local-hull vertices (A_all)
    plt.scatter(
        X_proj[:, 0], X_proj[:, 1],
        c='black', edgecolor='black', linewidths=0.6,
        alpha=0.7, s=100, label="$B_k$ prototypes", zorder=100
    )
    
    # Current global archetypes A (black Xs)
    plt.scatter(
        A_proj[:, 0], A_proj[:, 1],
        c='red', marker='X', s=150, label="Global archetypes $A$",zorder=150
    )
    
    # Shade global convex hull of A in light gray/black
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
    
    # Categorical colorbar with integer ticks
    #cb = plt.colorbar(scatter1, ticks=np.arange(K_detected))
    #cb.set_label("Cluster assignment (argmax of $m_i$)")
    
    # Axes/labels
    plt.xlabel("PCA Component 1", fontsize=20)
    plt.ylabel("PCA Component 2", fontsize=20)
    #plt.title("Latent space, global archetypes, and local hulls (PCA projection)", fontsize=14, fontweight='bold')
    
    plt.axis('equal')
    #plt.grid(False, linestyle="--", alpha=0.5)
    plt.legend(frameon=True, fontsize=15, loc='best')
    plt.tight_layout()
    plt.tick_params(axis='x', labelsize=18)  # x-axis ticks
    plt.tick_params(axis='y', labelsize=18)  # y-axis ticks
    
    # Save high-res PNG (good for slides) and vector PDF (best for LaTeX)
    plt.savefig(f"latent_hulls_{K}.png", bbox_inches="tight",dpi=200)
    
    plt.show()
    
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    import torch
    from matplotlib.patches import Rectangle
    
    # Build adjacency (as before)
    adj = torch.zeros(N, N, device=sparse_i.device, dtype=torch.float32)
    adj[sparse_i, sparse_j] = 1.0
    adj[sparse_j, sparse_i] = 1.0  # if graph is undirected
    adj.fill_diagonal_(0)
    
    # Your reordering (community, then local index)
    mask = torch.arange(0, model.K**2, model.K, device=adj.device)
    idx_comm = model.latent_z_.argmax(1)
    idx_loc  = model.latent_z_loc.argmax(1)
    key = mask[idx_comm] + idx_loc
    idx = torch.argsort(key)  # use stable=True if available
    
    # Reorder adjacency and labels
    A = adj[idx][:, idx].detach().cpu().numpy()
    comm_ord = idx_comm[idx].detach().cpu().numpy()
    
    # Spy-like scatter
    ii, jj = np.nonzero(A)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(jj, ii, s=1., c='blue', marker='s', linewidths=0)
    
    # Make it look like spy (top-left origin, square pixels)
    ax.invert_yaxis()
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-0.5, A.shape[1]-0.5)
    ax.set_ylim(A.shape[0]-0.5, -0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # --- Draw one rectangle per community block ---
    # Find contiguous blocks of identical community labels after reordering
    if comm_ord.size > 0:
        change = np.where(np.diff(comm_ord) != 0)[0] + 1
        starts = np.r_[0, change]
        ends   = np.r_[change, comm_ord.size]
    
        # choose distinct but unobtrusive colors for edges
        palette = list(plt.cm.tab20.colors)
    
        for s, e in zip(starts, ends):
            cval = int(comm_ord[s])
            rect = Rectangle(
                (s - 0.5, s - 0.5),            # lower-left corner
                e - s, e - s,                  # width, height
                fill=False,
                edgecolor=palette[cval % len(palette)],
                linewidth=5,
                alpha=0.8
            )
            ax.add_patch(rect)
    
    plt.tight_layout()
    plt.savefig(f"adj_{K}.png", bbox_inches="tight", dpi=600)
    
    plt.show()
    
    
    
    fracs=[]
    epochs=10000
    for e in range(epochs):
        frac = min(1.0, (e/epochs))**0.5
        fracs.append(frac)
    import numpy as np
    from itertools import combinations
    from scipy.optimize import linprog
    
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
    
    # Example:
    polys = list(model.A_all.detach().numpy())
    print(overlapping_pairs(polys))














labels = model.m.argmax(1).detach().cpu().numpy()
K_detected = int(labels.max() + 1)
palette = make_distinct_palette(K_detected)

# CIRCULAR PLOTS------------------------------------------------------------------- 


for k in range(model.K):
    
    col=palette[k % K_detected]

    idx=model.m.argmax(1)==k
    
    nodes=torch.where(model.m.argmax(1)==k)[0]
    print(nodes.shape[0])

    
    sparse_i_=[]
    sparse_j_=[]
    
    for i,j in zip(sparse_i,sparse_j):
        if i in nodes:
            if j in nodes:
                sparse_i_.append(i)
                sparse_j_.append(j)
                
    sparse_i__=torch.stack(sparse_i_).long()
    sparse_j__=torch.stack(sparse_j_).long()
    
    
    mask=torch.zeros(int(nodes.max()+1)).long()
    
    mask[nodes]=torch.arange(nodes.unique().shape[0]).long()
    
    
    
    sparse_i_=mask[sparse_i__]
    sparse_j_=mask[sparse_j__]
    
            
        
    
    
    Z=model.latent_z_loc[idx].detach().cpu().numpy()
    
    pca = PCA(n_components=2)
    
    X_=pca.fit_transform(Z)

    arg_max=Z.argmax(1)
    
    comp=pca.components_.transpose()
    inv = np.arctan2(comp[:, 1], comp[:, 0])
    degree = np.mod(np.degrees(inv), 360)
    
    idxs=np.argsort(degree)
    
    step=(2*math.pi)/model.K
    radius=10
    points=np.zeros((model.K,2))
    for i in range(model.K): 
        points[i,0] = (radius * math.cos(i*step))
        points[i,1] = (radius * math.sin(i*step))
        
    points=points[idxs]
    
       
    # plt.scatter(points[:,0],points[:,1])
    _X=Z@points
    
    
    print('CREATING and SAVING circular plots!!! \n')
    plt.figure(figsize=(7,7),dpi=300)
    
    for i,j in zip(sparse_i_.cpu().numpy(),sparse_j_.cpu().numpy()):
        plt.plot([_X[i,0], _X[j,0]], [_X[i,1], _X[j,1]],color=col,lw=0.7,alpha=0.35,zorder=2)
    plt.scatter(_X[:,0],_X[:,1],c=col,s=50,zorder=500,edgecolors="black",   # add black borders
    linewidths=0.5 )
    plt.scatter(points[:,0],points[:,1],c='black',s=150,alpha=0.8,zorder=1000)
    plt.set_cmap("tab10")
    plt.axis('off')
    plt.savefig(f"cir_{dataset}_{k}.png",dpi=200)
    plt.show()
    
    
    
       
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.spatial import ConvexHull
    from matplotlib.colors import ListedColormap, BoundaryNorm
    
    # --- Labels and K ---
    labels = model.m.argmax(1).detach().cpu().numpy()
    K_detected = int(labels.max() + 1)
    
    # --- Build a big qualitative palette (distinct categorical colors) ---
    def make_distinct_palette(K):
        # Grab colors from multiple qualitative colormaps
        cmaps = [
            plt.cm.tab10, plt.cm.Dark2, plt.cm.Set1, plt.cm.Set2, plt.cm.Set3,
            plt.cm.Paired, plt.cm.tab20, plt.cm.tab20b, plt.cm.tab20c
        ]
        cols = []
        for cm in cmaps:
            n = getattr(cm, 'N', 256)
            # Sample evenly across each qualitative map
            take = 10 if cm in (plt.cm.tab10, plt.cm.Dark2, plt.cm.Set1, plt.cm.Set2, plt.cm.Set3, plt.cm.Paired) else 20
            idxs = np.linspace(0, n - 1, take, dtype=int)
            cols.extend([cm(i) for i in idxs])
            if len(cols) >= K:
                break
        # Fallback if still short: evenly sample HSV
        if len(cols) < K:
            hsv = plt.cm.hsv(np.linspace(0, 1, K - len(cols), endpoint=False))
            cols.extend([tuple(c) for c in hsv])
        return cols[:K]
    
    
    palette = make_distinct_palette(K_detected)
    cmap_cat = ListedColormap(palette)
    norm_cat = BoundaryNorm(np.arange(K_detected + 1) - 0.5, K_detected)
    
    # --- Reshape local hull vertices (projected) for per-community colors ---
    K_hulls, L = model.A_all.shape[:2]
    B_proj = X_proj.reshape(K_hulls, L, 2)
    
    # --- Plot ---
    plt.figure(figsize=(10, 8))
    
    # Latent Z colored by cluster id with categorical cmap
    scatter1 = plt.scatter(
        latent_proj[:, 0], latent_proj[:, 1],
        c=labels, cmap=cmap_cat, norm=norm_cat,
        alpha=0.75, s=40, linewidths=0.2, edgecolors='k',
        label="Node embeddings $z_i$"
    )
    
    # Shade each local hull with its community color (semi-transparent)
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
            # outline for clarity
            plt.plot(
                np.r_[polyc[:, 0], polyc[0, 0]],
                np.r_[polyc[:, 1], polyc[0, 1]],
                color=palette[c % K_detected], lw=1.0, zorder=1
            )
    
    # All local-hull vertices (A_all)
    plt.scatter(
        X_proj[:, 0], X_proj[:, 1],
        c='black', edgecolor='black', linewidths=0.6,
        alpha=0.7, s=100, label="$B_k$ prototypes", zorder=100
    )
    
    # Current global archetypes A (black Xs)
    plt.scatter(
        A_proj[:, 0], A_proj[:, 1],
        c='red', marker='X', s=150, label="Global archetypes $A$",zorder=150
    )
   
    
   
    plt.axis('equal')
    #plt.grid(False, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.tick_params(axis='x', labelsize=18)  # x-axis ticks
    plt.tick_params(axis='y', labelsize=18)  # y-axis ticks
    
    # Save high-res PNG (good for slides) and vector PDF (best for LaTeX)
    plt.savefig(f"latent_hulls_{c}_.png", bbox_inches="tight",dpi=200)
    
    plt.show()
    
import numpy as np
import matplotlib.pyplot as plt
import torch
from matplotlib.patches import Rectangle

# --------------------------
# Inputs assumed pre-defined:
# N, sparse_i, sparse_j, model with attributes:
#   - K
#   - latent_z_ (N x K)       -> community logits
#   - latent_z_loc (N x K)    -> local index logits
# --------------------------

# --- Build adjacency (as before) ---
adj = torch.zeros(N, N, device=sparse_i.device, dtype=torch.float32)
adj[sparse_i, sparse_j] = 1.0
adj[sparse_j, sparse_i] = 1.0  # if graph is undirected
adj.fill_diagonal_(0)

# --- Reordering (community, then local index) ---
mask = torch.arange(0, model.K**2, model.K, device=adj.device)
idx_comm = model.latent_z_.argmax(1)          # community id per node
idx_loc  = model.latent_z_loc.argmax(1)       # local id within community
key = mask[idx_comm] + idx_loc
idx = torch.argsort(key)                      # add stable=True if PyTorch>=2.0

# --- Reorder adjacency and labels ---
A = adj[idx][:, idx].detach().cpu().numpy()
comm_ord = idx_comm[idx].detach().cpu().numpy()

# --- Spy-like scatter (your original look) ---
ii, jj = np.nonzero(A)
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(jj, ii, s=1.0,  c='#1f77b4', marker='s', linewidths=0)

# Make it look like spy (top-left origin, square pixels)
ax.invert_yaxis()
ax.set_aspect('equal', adjustable='box')
ax.set_xlim(-0.5, A.shape[1] - 0.5)
ax.set_ylim(A.shape[0] - 0.5, -0.5)
ax.set_xticks([])
ax.set_yticks([])

# -------------------------------
# Block outlines (diagonal + off-diagonal)
# with 1-1 Tab20 mapping matching hulls
# -------------------------------
if comm_ord.size > 0:
    # contiguous row/col blocks after reordering
    change = np.where(np.diff(comm_ord) != 0)[0] + 1
    starts = np.r_[0, change]
    ends   = np.r_[change, comm_ord.size]

    # Tab20 comes in (dark, light) pairs per hue.
    # Set this to whichever half your **hulls** used:
    TAB20_PAIR = "dark"  # or "light"
    pair_offset = 0 if TAB20_PAIR == "dark" else 1

    palette = list(plt.cm.tab20.colors)

    # Collect blocks: (start, end, community label)
    blocks = [(int(s), int(e), int(comm_ord[s])) for s, e in zip(starts, ends)]

    # Draw rectangles for all block pairs that have edges
    MIN_ONES = 1  # increase to reduce clutter, e.g., 50 or 100
    A_bin = (A > 0)

    for i, (rs, re, ci) in enumerate(blocks):        # row-block
        for j, (cs, ce, cj) in enumerate(blocks):    # col-block
            sub = A_bin[rs:re, cs:ce]
            if np.count_nonzero(sub) >= MIN_ONES:
                on_diag = (i == j)

                # 1-1 mapping to Tab20 pair indices:
                # ci=0 -> idx 0/1 (blue), ci=1 -> 2/3 (orange), ci=2 -> 4/5 (green), ...
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

# Save
K = int(getattr(model, "K", 0))
plt.savefig(f"adj_{K}.png", bbox_inches="tight", dpi=600)
plt.show()
