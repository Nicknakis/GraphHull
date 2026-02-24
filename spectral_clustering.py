
import scipy
from scipy import sparse
import numpy as np
import torch
import networkx as nx
from sklearn.manifold import MDS


from sklearn.preprocessing import StandardScaler

class Spectral_clustering_init():
    def __init__(self,num_of_eig=10,method='Normalized_sym',device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        
        self.num_of_eig=num_of_eig
        self.method=method
        self.device=device

    
    def spectral_clustering(self):
        
        sparse_i=self.sparse_i_idx.cpu().numpy()
        sparse_j=self.sparse_j_idx.cpu().numpy()
        idx_shape=sparse_i.shape[0]
        if (sparse_i<sparse_j).sum()==idx_shape:
            sparse_i_new=np.concatenate((sparse_i,sparse_j))
            sparse_j_new=np.concatenate((sparse_j,sparse_i))
            
            sparse_i=sparse_i_new
            sparse_j=sparse_j_new
            
        V=np.ones(sparse_i.shape[0])
   
        Affinity_matrix=sparse.coo_matrix((V,(sparse_i,sparse_j)),shape=(self.input_size,self.input_size))
        
       
        
        if self.method=='Adjacency':
             eig_val, eig_vect = scipy.sparse.linalg.eigsh(Affinity_matrix,self.num_of_eig,which='LM')
             X = eig_vect.real
             rows_norm = np.linalg.norm(X, axis=1, ord=2)
             U_norm = (X.T / rows_norm).T
             
             

            
        elif self.method=='Normalized_sym':
            n, m = Affinity_matrix.shape
            diags = Affinity_matrix.sum(axis=1).flatten()
            D = sparse.spdiags(diags, [0], m, n, format="csr")
            L = D - Affinity_matrix
            with scipy.errstate(divide="ignore"):
                diags_sqrt = 1.0 / np.sqrt(diags)
            diags_sqrt[np.isinf(diags_sqrt)] = 0
            DH = sparse.spdiags(diags_sqrt, [0], m, n, format="csr")
            tem=DH @ (L @ DH)
            #eig_val, eig_vect = scipy.sparse.linalg.eigs(tem,self.num_of_eig,which='SR')
            #X = eig_vect.real
            #self.X=X
            #rows_norm = np.linalg.norm(X, axis=1,ord=2)
            #U_norm =(X.T / rows_norm).T
            
            v0 = np.full(m, 1.0/np.sqrt(m))  # deterministic start vector
            # 'SM' = smallest magnitude; for normalized Laplacian, also consider 'SA'
            eig_val, eig_vect = scipy.sparse.linalg.eigsh(tem, k=self.num_of_eig, which='SM', v0=v0, tol=1e-6, maxiter=5000)
            X = eig_vect.real
            U_norm = (X.T / np.clip(np.linalg.norm(X, axis=1, ord=2), 1e-12, None)).T
                        
                            
                
        elif self.method=='Normalized':
            n, m = Affinity_matrix.shape
            diags = Affinity_matrix.sum(axis=1).flatten()
            D = sparse.spdiags(diags, [0], m, n, format="csr")
            L = D - Affinity_matrix
            with scipy.errstate(divide="ignore"):
                diags_inv = 1.0 /diags
            diags_inv[np.isinf(diags_inv)] = 0
            DH = sparse.spdiags(diags_inv, [0], m, n, format="csr")
            tem=DH @L
            eig_val, eig_vect = scipy.sparse.linalg.eigs(tem,self.num_of_eig,which='SR')
    
            X = eig_vect.real
            self.X=X
            U_norm =X
            
        elif self.method=='MDS':
            n, m = Affinity_matrix.shape

            G= nx.Graph(Affinity_matrix)

            max_l=0
            N = G.number_of_nodes()
            pmat = np.zeros((N, N))+np.inf
            paths = nx.all_pairs_shortest_path_length(G)
            for node_i, node_ij in paths:
                for node_j, length_ij in node_ij.items():
                    pmat[node_i, node_j] = length_ij
                    if length_ij>max_l:
                        max_l=length_ij

            pmat[pmat == np.inf] = max_l+1
            print('shortest path done')
            
            embedding = MDS(n_components=self.num_of_eig,dissimilarity='precomputed')   
            U_norm = embedding.fit_transform(pmat)
            
        else:
            print('Invalid Spectral Clustering Method')
            
        norms = np.linalg.norm(U_norm, axis=1, keepdims=True)   # (N, 1)
        U_norm= U_norm / np.clip(norms,1e-4, None)


        
        return torch.from_numpy(U_norm).float().to(self.device)   
            
        


