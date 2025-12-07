import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, global_mean_pool, global_add_pool
from models.positional_encoding import LaplacianPE, RandomWalkPE

class GINEStyleGraphTransformer(nn.Module):
    """
    GINE-style Graph Transformer:
    TransformerConv → Node MLP → Edge MLP → Residual → LayerNorm
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
        pe_type='laplacian',
        pe_dim=16,
        use_edge_features=True
    ):
        super().__init__()

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.use_edge_features = use_edge_features
        self.pe_type = pe_type

        # -------- Positional Encoding --------
        if pe_type == 'laplacian':
            self.pe = LaplacianPE(pe_dim=pe_dim)
        elif pe_type == 'random_walk':
            self.pe = RandomWalkPE(pe_dim=pe_dim)
        else:
            self.pe = None
            pe_dim = 0

        # -------- Input Encoders --------
        self.node_encoder = nn.Linear(num_node_features + pe_dim, hidden_dim)

        if use_edge_features and num_edge_features > 0:
            self.raw_edge_encoder = nn.Linear(num_edge_features, hidden_dim)
        else:
            self.raw_edge_encoder = None

        # DEEP EDGE MLP (GINE-style)
        self.edge_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            ) if self.raw_edge_encoder is not None else None
            for _ in range(num_layers)
        ])

        # -------- Transformer Attention Layers --------
        self.transformer_layers = nn.ModuleList([
            TransformerConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim // num_heads,
                heads=num_heads,
                dropout=dropout,
                edge_dim=hidden_dim if self.raw_edge_encoder is not None else None,
                beta=True
            )
            for _ in range(num_layers)
        ])

        # -------- Post-Attention NODE MLP (GINE-style) --------
        self.node_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for _ in range(num_layers)
        ])

        # -------- Norm + Dropout --------
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

        # -------- Output Predictor --------
        self.graph_pred_linear = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_tasks)
        )

        # Used for attention visualization
        self.attention_weights = []


    def forward(self, data, return_attention=False):
        x, edge_index, edge_attr, batch = (
            data.x, data.edge_index, data.edge_attr, data.batch
        )

        # ----- Positional Encoding -----
        if self.pe is not None:
            pe = self.pe(edge_index, batch, x.shape[0])
            x = torch.cat([x, pe], dim=-1)

        # ----- Node encoding -----
        x = self.node_encoder(x)
        x = F.relu(x)

        # ----- Edge encoding (raw) -----
        if self.raw_edge_encoder is not None and edge_attr is not None:
            edge_attr = self.raw_edge_encoder(edge_attr)

        self.attention_weights = []

        # ----- LAYERS -----
        for layer_idx in range(self.num_layers):
            transformer = self.transformer_layers[layer_idx]
            node_mlp = self.node_mlps[layer_idx]
            edge_mlp = self.edge_mlps[layer_idx]
            norm = self.layer_norms[layer_idx]

            # Save for residual
            x_residual = x

            # DEEP EDGE MLP
            if edge_attr is not None and edge_mlp is not None:
                edge_attr_input = edge_attr
                edge_attr = edge_mlp(edge_attr_input)

            # ----- Transformer attention -----
            if return_attention:
                x_att, att_data = transformer(
                    x, edge_index, edge_attr, return_attention_weights=True
                )
                edge_index_att, att_weights = att_data
                self.attention_weights.append((edge_index_att, att_weights))
            else:
                x_att = transformer(x, edge_index, edge_attr)

            # ----- GINE-style NODE MLP -----
            x = node_mlp(x_att)

            # ----- Residual + LayerNorm -----
            x = norm(x + x_residual)
            x = self.dropout(x)

        # ----- Global Pooling -----
        x = global_mean_pool(x, batch)

        # ----- Prediction -----
        return self.graph_pred_linear(x)


    def get_attention_weights(self):
        return self.attention_weights
