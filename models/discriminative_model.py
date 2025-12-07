import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, BatchNorm1d, ReLU
from torch_geometric.nn import GPSConv, GINEConv, global_add_pool

class Discriminative(nn.Module):
    """
    Discriminative Network
    """
    def __init__(
        self,
        num_node_features,
        num_edge_features,
        num_tasks,
        hidden_dim=128,
        num_layers=4,
        num_heads=4,
        dropout=0.1,
        pe_dim=20,
        use_edge_features=True
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.use_edge_features = use_edge_features

        # Node Encoder
        self.node_encoder = nn.Linear(num_node_features, hidden_dim)
        
        # Edge Encoder
        if use_edge_features and num_edge_features > 0:
            self.edge_encoder = nn.Linear(num_edge_features, hidden_dim)
        else:
            self.edge_encoder = None

        # Positional Encoding (PE) Encoder
        self.pe_encoder = nn.Linear(pe_dim, hidden_dim)
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            local_nn = Sequential(
                Linear(hidden_dim, hidden_dim),
                BatchNorm1d(hidden_dim),
                ReLU(),
                Linear(hidden_dim, hidden_dim),
            )
            local_conv = GINEConv(local_nn)
            
            layer = GPSConv(
                channels=hidden_dim,
                conv=local_conv,
                heads=num_heads,
                attn_type='multihead',
                act='relu',
                norm='batch_norm',
                dropout=dropout,
            )
            self.layers.append(layer)
        self.graph_pred_linear = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_tasks)
        )
        
    def forward(self, data):
        """
        Forward pass
        Args:
            data: PyG Data object with x, edge_index, edge_attr, batch, and pe
        """
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # Encode Features
        x = self.node_encoder(x)
        
        if self.edge_encoder is not None and edge_attr is not None:
            edge_attr_encoded = self.edge_encoder(edge_attr)
        else:
            edge_attr_encoded = None

        # Assume data.pe exists (e.g., RandomWalkPE or LaplacianPE transform)
        if hasattr(data, 'pe') and data.pe is not None:
             pe_feat = self.pe_encoder(data.pe)
             x = x + pe_feat 
        
        # GPS Backbone Loop
        for layer in self.layers:
            x = layer(x, edge_index, batch, edge_attr=edge_attr_encoded)
        
        # Global Pooling
        x_graph = global_add_pool(x, batch)
        
        # Prediction
        out = self.graph_pred_linear(x_graph)
        
        return out