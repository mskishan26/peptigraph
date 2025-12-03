import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, global_mean_pool, global_add_pool
from models.positional_encoding import LaplacianPE, RandomWalkPE

class GraphTransformer(nn.Module):
    """
    Graph Transformer for Peptides-func
    """
    def __init__(
        self,
        num_node_features,
        num_edge_features,
        num_tasks,
        hidden_dim=128,
        num_layers=4,
        num_heads=8,
        dropout=0.1,
        pe_type='laplacian',  # 'laplacian', 'random_walk', or 'none'
        pe_dim=16,
        use_edge_features=True
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.pe_type = pe_type
        self.use_edge_features = use_edge_features
        
        # Positional encoding
        if pe_type == 'laplacian':
            self.pe = LaplacianPE(pe_dim=pe_dim)
        elif pe_type == 'random_walk':
            self.pe = RandomWalkPE(pe_dim=pe_dim)
        else:
            self.pe = None
            pe_dim = 0
        
        # Input projection
        self.node_encoder = nn.Linear(num_node_features + pe_dim, hidden_dim)
        
        if use_edge_features and num_edge_features > 0:
            self.edge_encoder = nn.Linear(num_edge_features, hidden_dim)
        else:
            self.edge_encoder = None
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim // num_heads,
                heads=num_heads,
                dropout=dropout,
                edge_dim=hidden_dim if use_edge_features and num_edge_features > 0 else None,
                beta=True  # Use gating mechanism
            )
            for _ in range(num_layers)
        ])
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Output head
        self.graph_pred_linear = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_tasks)
        )
        
        # Store attention weights for visualization
        self.attention_weights = []
        
    def forward(self, data, return_attention=False):
        """
        Forward pass
        
        Args:
            data: PyG Data object with x, edge_index, edge_attr, batch
            return_attention: if True, store attention weights
            
        Returns:
            out: [batch_size, num_tasks] predictions
        """
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # Compute positional encoding
        if self.pe is not None:
            pe = self.pe(edge_index, batch, x.shape[0])
            x = torch.cat([x, pe], dim=-1)
        
        # Encode node features
        x = self.node_encoder(x)
        x = F.relu(x)
        
        # Encode edge features
        edge_attr_encoded = None
        if self.edge_encoder is not None and edge_attr is not None:
            edge_attr_encoded = self.edge_encoder(edge_attr)
        
        # Clear attention weights
        self.attention_weights = []
        
        # Transformer layers
        for i, (transformer, norm) in enumerate(zip(self.transformer_layers, self.layer_norms)):
            # Store input for residual
            x_residual = x
            
            # Transformer layer
            if return_attention:
                # TransformerConv returns (x, attention_weights) when return_attention_weights=True
                x, (edge_index_att, att_weights) = transformer(
                    x, edge_index, edge_attr_encoded, return_attention_weights=True
                )
                self.attention_weights.append((edge_index_att, att_weights))
            else:
                x = transformer(x, edge_index, edge_attr_encoded)
            
            # Residual connection + LayerNorm
            x = norm(x + x_residual)
            x = self.dropout(x)
        
        # Global pooling
        x = global_mean_pool(x, batch)
        
        # Prediction
        out = self.graph_pred_linear(x)
        
        return out
    
    def get_attention_weights(self):
        """Return stored attention weights"""
        return self.attention_weights