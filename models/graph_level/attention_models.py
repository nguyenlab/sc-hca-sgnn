import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, TransformerConv
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from typing import Optional


class GATEncoder(nn.Module):
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.3,
        concat_heads: bool = True,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.heads = heads
        self.dropout = dropout
        self.concat_heads = concat_heads
        
        # Input projection
        self.input_linear = nn.Linear(input_dim, hidden_dim)
        
        # GAT layers
        self.convs = nn.ModuleList()
        
        for i in range(num_layers):
            # For all layers except the last, concatenate heads
            # For the last layer, average heads to get final hidden_dim
            if i < num_layers - 1:
                conv = GATConv(
                    hidden_dim,
                    hidden_dim // heads,
                    heads=heads,
                    dropout=dropout,
                    concat=True,  # Concatenate multi-head outputs
                )
            else:
                conv = GATConv(
                    hidden_dim,
                    hidden_dim,
                    heads=heads,
                    dropout=dropout,
                    concat=False,  # Average multi-head outputs
                )
            self.convs.append(conv)
        
        # Layer normalization
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.norms.append(nn.LayerNorm(hidden_dim))
    
    def forward(self, x, edge_index):
        # Project input
        x = self.input_linear(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Apply GAT layers with residual connections
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x_res = x
            
            # GAT convolution
            x = conv(x, edge_index)
            x = norm(x)
            x = F.elu(x)  # ELU is commonly used with GAT
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            # Residual connection
            if i < self.num_layers - 1:  # Skip residual on last layer if dims don't match
                x = x + x_res
        
        return x


class TransformerGNNEncoder(nn.Module):
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        heads: int = 1,
        dropout: float = 0.3,
        edge_dim: Optional[int] = None,
        use_positional_encoding: bool = False,  # Disabled by default for stability
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.heads = heads
        self.dropout = dropout
        
        # Input projection
        self.input_linear = nn.Linear(input_dim, hidden_dim)
        
        # Transformer conv layers - simple architecture without FFN
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for _ in range(num_layers):
            # TransformerConv without concat (average heads)
            conv = TransformerConv(
                hidden_dim,
                hidden_dim,
                heads=heads,
                dropout=dropout,
                edge_dim=edge_dim,
                concat=False,  # Average heads instead of concatenate
            )
            self.convs.append(conv)
            self.norms.append(nn.LayerNorm(hidden_dim))
    
    def forward(self, x, edge_index, batch=None):
        # Project input
        x = self.input_linear(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Apply Transformer layers with normalization
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x


class GATClassifier(nn.Module):
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        heads: int = 4,
        num_classes: int = 2,
        dropout: float = 0.3,
        pooling: str = 'mean',
    ):
        super().__init__()
        
        self.encoder = GATEncoder(
            input_dim, hidden_dim, num_layers, heads, dropout
        )
        self.pooling = pooling
        
        # Determine pooling output dimension
        pool_dim = hidden_dim * (2 if pooling == 'both' else 1)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(pool_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )
    
    def forward(self, x, edge_index, edge_type=None, batch=None):
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # Encode nodes
        node_emb = self.encoder(x, edge_index)
        
        # Pool to graph-level
        if self.pooling == 'mean':
            graph_emb = global_mean_pool(node_emb, batch)
        elif self.pooling == 'max':
            graph_emb = global_max_pool(node_emb, batch)
        elif self.pooling == 'add':
            graph_emb = global_add_pool(node_emb, batch)
        elif self.pooling == 'both':
            mean_emb = global_mean_pool(node_emb, batch)
            max_emb = global_max_pool(node_emb, batch)
            graph_emb = torch.cat([mean_emb, max_emb], dim=-1)
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")
        
        # Classify
        logits = self.classifier(graph_emb)
        
        return logits


class TransformerGNNClassifier(nn.Module):
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        heads: int = 4,
        num_classes: int = 2,
        dropout: float = 0.3,
        pooling: str = 'mean',
        use_positional_encoding: bool = False,  # Not used anymore, kept for compatibility
    ):
        super().__init__()
        
        self.encoder = TransformerGNNEncoder(
            input_dim, hidden_dim, num_layers, heads, dropout,
        )
        self.pooling = pooling
        
        # Determine pooling output dimension
        pool_dim = hidden_dim * (2 if pooling == 'both' else 1)
        
        # Simplified classification head (like legacy SubGCN)
        self.classifier = nn.Sequential(
            nn.Linear(pool_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )
    
    def forward(self, x, edge_index, edge_type=None, batch=None):
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # Encode nodes
        node_emb = self.encoder(x, edge_index, batch)
        
        # Pool to graph-level
        if self.pooling == 'mean':
            graph_emb = global_mean_pool(node_emb, batch)
        elif self.pooling == 'max':
            graph_emb = global_max_pool(node_emb, batch)
        elif self.pooling == 'add':
            graph_emb = global_add_pool(node_emb, batch)
        elif self.pooling == 'both':
            mean_emb = global_mean_pool(node_emb, batch)
            max_emb = global_max_pool(node_emb, batch)
            graph_emb = torch.cat([mean_emb, max_emb], dim=-1)
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")
        
        # Classify
        logits = self.classifier(graph_emb)
        
        return logits


def create_gat_model(
    input_dim: int,
    hidden_dim: int = 64,
    num_layers: int = 3,
    heads: int = 4,
    num_classes: int = 2,
    dropout: float = 0.3,
    pooling: str = 'mean',
):
    return GATClassifier(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        heads=heads,
        num_classes=num_classes,
        dropout=dropout,
        pooling=pooling,
    )


def create_transformer_gnn_model(
    input_dim: int,
    hidden_dim: int = 64,
    num_layers: int = 3,
    heads: int = 4,
    num_classes: int = 2,
    dropout: float = 0.3,
    pooling: str = 'mean',
    use_positional_encoding: bool = False,  # Not used anymore, kept for compatibility
):
    return TransformerGNNClassifier(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        heads=heads,
        num_classes=num_classes,
        dropout=dropout,
        pooling=pooling,
        use_positional_encoding=use_positional_encoding,
    )
