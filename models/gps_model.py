import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import (
    GINConv,
    GCNConv,
    TransformerConv,
    global_mean_pool,
)


class GPSLayer(nn.Module):
    """
    One GPS block:
      1) Local GNN message passing (GIN or GCN)
      2) Global Transformer-style attention (TransformerConv)
      3) FFN + LayerNorm + Dropout (Transformer-style)
    """

    def __init__(
        self,
        hidden_dim: int,
        edge_dim: int,
        local_type: str = "gin",
        heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        # ----- Local GNN -----
        if local_type == "gin":
            nn_local = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.local = GINConv(nn_local)

        elif local_type == "gcn":
            self.local = GCNConv(hidden_dim, hidden_dim)
        else:
            raise ValueError(f"Unknown local_type: {local_type}")

        # ----- Global attention (TransformerConv) -----
        # out_channels * heads = hidden_dim
        self.global_attn = TransformerConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim // heads,
            heads=heads,
            edge_dim=edge_dim,   
            dropout=dropout,
        )

        # ----- Feed-forward network -----
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr):
        # Local message passing
        local_out = self.local(x, edge_index)
        x = x + self.dropout(local_out)
        x = self.norm1(x)

        # Global attention
        attn_out = self.global_attn(x, edge_index, edge_attr)
        x = x + self.dropout(attn_out)

        # FFN
        ffn_out = self.ffn(x)
        x = x + self.dropout(ffn_out)
        x = self.norm2(x)

        return x


class GraphGPS(nn.Module):


    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_classes: int = 10,
        local_type: str = "gin",
        heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        # Encode raw node features -> hidden_dim
        self.node_encoder = nn.Linear(node_dim, hidden_dim)
        # Encode raw edge features -> hidden_dim
        self.edge_encoder = nn.Linear(edge_dim, hidden_dim)

        # Stack multiple GPS layers
        self.layers = nn.ModuleList(
            [
                GPSLayer(
                    hidden_dim=hidden_dim,
                    edge_dim=hidden_dim,   
                    local_type=local_type,
                    heads=heads,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.pool = global_mean_pool

        # Graph-level prediction head
        self.mlp_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, data):
        # data: PyG Data or Batch object
        x, edge_index, edge_attr, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )

        x = x.float()
        edge_attr = edge_attr.float()

        # Encode features
        x = self.node_encoder(x)            # [num_nodes, hidden_dim]
        edge_attr = self.edge_encoder(edge_attr)  # [num_edges, hidden_dim]

        # GPS blocks
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)

        # Graph pooling
        graph_emb = self.pool(x, batch)     # [batch_size, hidden_dim]

        # Prediction (logits)
        out = self.mlp_out(graph_emb)       # [batch_size, num_classes]
        return out
