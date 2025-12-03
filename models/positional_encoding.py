import torch
import torch.nn as nn
import numpy as np
from torch_geometric.utils import get_laplacian, to_dense_adj
from torch_geometric.utils import to_scipy_sparse_matrix
from scipy.sparse.linalg import eigsh

class LaplacianPE(nn.Module):
    """
    Laplacian Positional Encoding
    Uses eigenvectors of graph Laplacian as positional features
    """
    def __init__(self, pe_dim=16, normalization='sym'):
        super().__init__()
        self.pe_dim = pe_dim
        self.normalization = normalization
        
    def forward(self, edge_index, batch, num_nodes):
        """
        Args:
            edge_index: [2, E] edge indices
            batch: [N] batch assignment
            num_nodes: total number of nodes in batch
        Returns:
            pe: [N, pe_dim] positional encodings
        """
        device = edge_index.device
        
        # Get unique graphs in batch
        unique_batches = torch.unique(batch)
        pe_list = []
        
        for b in unique_batches:
            # Get subgraph for this batch element
            mask = batch == b
            node_ids = torch.where(mask)[0]
            
            # Create subgraph edge_index
            edge_mask = torch.isin(edge_index[0], node_ids) & torch.isin(edge_index[1], node_ids)
            sub_edge_index = edge_index[:, edge_mask]
            
            # Remap node indices to 0...n-1
            n_nodes = mask.sum().item()
            node_map = {node_ids[i].item(): i for i in range(n_nodes)}
            remapped_edges = torch.tensor([
                [node_map[sub_edge_index[0, i].item()], node_map[sub_edge_index[1, i].item()]]
                for i in range(sub_edge_index.shape[1])
            ], device=device).T
            
            # Compute Laplacian eigenvectors
            try:
                L = to_scipy_sparse_matrix(
                    *get_laplacian(remapped_edges, normalization=self.normalization, num_nodes=n_nodes)
                )
                
                # Compute smallest k eigenvectors
                k = min(self.pe_dim, n_nodes - 2)
                eig_vals, eig_vecs = eigsh(L, k=k, which='SM', return_eigenvectors=True)
                
                # Pad if necessary
                if k < self.pe_dim:
                    padding = torch.zeros((n_nodes, self.pe_dim - k), device=device)
                    eig_vecs = torch.cat([
                        torch.from_numpy(eig_vecs).float().to(device),
                        padding
                    ], dim=1)
                else:
                    eig_vecs = torch.from_numpy(eig_vecs).float().to(device)
                    
            except:
                # Fallback: use zeros if Laplacian computation fails
                eig_vecs = torch.zeros((n_nodes, self.pe_dim), device=device)
            
            pe_list.append(eig_vecs)
        
        # Concatenate all positional encodings
        pe = torch.cat(pe_list, dim=0)
        return pe


class RandomWalkPE(nn.Module):
    """
    Random Walk Positional Encoding
    Uses powers of the transition matrix diagonal
    """
    def __init__(self, pe_dim=16):
        super().__init__()
        self.pe_dim = pe_dim
        
    def forward(self, edge_index, batch, num_nodes):
        """
        Compute random walk landing probabilities
        """
        device = edge_index.device
        
        # Get adjacency matrix
        adj = to_dense_adj(edge_index, batch=batch, max_num_nodes=num_nodes // len(torch.unique(batch)))
        
        # Compute degree matrix
        deg = adj.sum(dim=-1, keepdim=True).clamp(min=1)
        
        # Transition matrix
        P = adj / deg
        
        # Compute powers of P and extract diagonal (self-return probabilities)
        pe_list = []
        Pk = torch.eye(P.shape[-1], device=device).unsqueeze(0).repeat(P.shape[0], 1, 1)
        
        for k in range(self.pe_dim):
            # Extract diagonal (self-return probability at step k)
            diag = torch.diagonal(Pk, dim1=-2, dim2=-1)
            pe_list.append(diag)
            Pk = torch.bmm(Pk, P)
        
        pe = torch.stack(pe_list, dim=-1)  # [batch_size, max_nodes, pe_dim]
        
        # Flatten to [total_nodes, pe_dim]
        pe = pe.reshape(-1, self.pe_dim)
        
        return pe