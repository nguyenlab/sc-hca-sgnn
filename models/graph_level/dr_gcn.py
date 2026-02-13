import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_mean_pool, global_max_pool
from torch_geometric.utils import add_self_loops, degree
from typing import Optional


class DRGCNConv(MessagePassing):
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        add_self_loop: bool = True,
        bias: bool = True,
        **kwargs
    ):
        super().__init__(aggr='add', **kwargs)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.add_self_loop = add_self_loop
        
        # Learnable weight matrix
        self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels))
        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x, edge_index, edge_weight=None):
        # Add self-loops
        if self.add_self_loop:
            edge_index, edge_weight = add_self_loops(
                edge_index, edge_weight, num_nodes=x.size(0)
            )
        
        # Transform features
        x = torch.matmul(x, self.weight)
        
        # Calculate degree-free normalization
        # Instead of using degree^{-1/2}, we use uniform weight 1/num_neighbors
        row, col = edge_index
        deg = degree(row, x.size(0), dtype=x.dtype)
        deg_inv = 1.0 / deg
        deg_inv[deg_inv == float('inf')] = 0
        
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.size(1), device=x.device)
        
        # Normalize by source degree (degree-free normalization)
        norm = deg_inv[row]
        edge_weight = edge_weight * norm
        
        # Message passing
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight)
        
        if self.bias is not None:
            out = out + self.bias
        
        return out
    
    def message(self, x_j, edge_weight):
        return edge_weight.view(-1, 1) * x_j


class TMPConv(MessagePassing):
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 4,
        dropout: float = 0.3,
        add_self_loop: bool = True,
        **kwargs
    ):
        super().__init__(aggr='add', node_dim=0, **kwargs)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.dropout = dropout
        self.add_self_loop = add_self_loop
        
        # Multi-head attention
        self.head_dim = out_channels // heads
        assert out_channels % heads == 0, "out_channels must be divisible by heads"
        
        # Query, Key, Value transformations
        self.W_q = nn.Linear(in_channels, out_channels)
        self.W_k = nn.Linear(in_channels, out_channels)
        self.W_v = nn.Linear(in_channels, out_channels)
        
        # Temporal encoding
        self.W_temp = nn.Linear(1, heads)  # Maps temporal distance to attention scores
        
        # Gated update
        self.gate = nn.Sequential(
            nn.Linear(in_channels + out_channels, out_channels),
            nn.Sigmoid()
        )
        
        # Output projection
        self.W_out = nn.Linear(out_channels, out_channels)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        for module in [self.W_q, self.W_k, self.W_v, self.W_out]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.W_temp.weight)
        nn.init.zeros_(self.W_temp.bias)
        for module in self.gate:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x, edge_index, temporal_pos=None):
        num_nodes = x.size(0)
        
        # Add self-loops
        if self.add_self_loop:
            edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        
        # Compute temporal distances if positions provided
        if temporal_pos is None:
            temporal_pos = torch.arange(num_nodes, device=x.device, dtype=x.dtype)
        
        row, col = edge_index
        temporal_dist = torch.abs(temporal_pos[row] - temporal_pos[col]).unsqueeze(-1)
        temporal_dist = temporal_dist / (temporal_dist.max() + 1e-6)  # Normalize to [0, 1]
        
        # Transform to Q, K, V
        q = self.W_q(x).view(num_nodes, self.heads, self.head_dim)
        k = self.W_k(x).view(num_nodes, self.heads, self.head_dim)
        v = self.W_v(x).view(num_nodes, self.heads, self.head_dim)
        
        # Message passing with temporal attention
        out = self.propagate(
            edge_index, 
            q=q, k=k, v=v,
            temporal_dist=temporal_dist,
            x=x
        )
        
        out = out.view(num_nodes, -1)
        out = F.dropout(out, p=self.dropout, training=self.training)
        out = self.W_out(out)
        
        # Gated update: blend old and new features
        gate_input = torch.cat([x, out], dim=-1)
        gate_value = self.gate(gate_input)
        out = gate_value * out + (1 - gate_value) * x
        
        return out
    
    def message(self, q_i, k_j, v_j, temporal_dist, index, ptr, size_i):
        # Multi-head attention: [num_edges, heads, head_dim]
        # Q·K / sqrt(d)
        attn_struct = (q_i * k_j).sum(dim=-1) / (self.head_dim ** 0.5)  # [num_edges, heads]
        
        # Temporal attention component
        attn_temp = self.W_temp(temporal_dist)  # [num_edges, heads]
        
        # Combine structural and temporal attention
        attn = attn_struct + attn_temp  # [num_edges, heads]
        attn = F.softmax(attn, dim=0)  # Normalize over all neighbors
        attn = F.dropout(attn, p=self.dropout, training=self.training)
        
        # Apply attention to values
        out = v_j * attn.unsqueeze(-1)  # [num_edges, heads, head_dim]
        
        return out


class DRGCNEncoder(nn.Module):
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Input projection
        self.input_linear = nn.Linear(input_dim, hidden_dim)
        
        # DR-GCN layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(DRGCNConv(hidden_dim, hidden_dim))
        
        # Layer normalization
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.norms.append(nn.LayerNorm(hidden_dim))
    
    def forward(self, x, edge_index):
        # Project input
        x = self.input_linear(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Apply conv layers with residual connections
        for conv, norm in zip(self.convs, self.norms):
            x_res = x
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x + x_res  # Residual connection
        
        return x


class TMPEncoder(nn.Module):
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.heads = heads
        self.dropout = dropout
        
        # Input projection
        self.input_linear = nn.Linear(input_dim, hidden_dim)
        
        # TMP layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(TMPConv(
                hidden_dim, hidden_dim,
                heads=heads, dropout=dropout
            ))
        
        # Layer normalization
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.norms.append(nn.LayerNorm(hidden_dim))
    
    def forward(self, x, edge_index, temporal_pos=None):
        # Project input
        x = self.input_linear(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Apply TMP layers (residual connections handled inside TMPConv)
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index, temporal_pos)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x


class DRGCNClassifier(nn.Module):
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        num_classes: int = 2,
        dropout: float = 0.3,
        pooling: str = 'mean',  # 'mean', 'max', 'both'
    ):
        super().__init__()
        
        self.encoder = DRGCNEncoder(input_dim, hidden_dim, num_layers, dropout)
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
        elif self.pooling == 'both':
            mean_emb = global_mean_pool(node_emb, batch)
            max_emb = global_max_pool(node_emb, batch)
            graph_emb = torch.cat([mean_emb, max_emb], dim=-1)
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")
        
        # Classify
        logits = self.classifier(graph_emb)
        
        return logits


class TMPClassifier(nn.Module):
    
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
        
        self.encoder = TMPEncoder(input_dim, hidden_dim, num_layers, heads, dropout)
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
    
    def forward(self, x, edge_index, edge_type=None, batch=None, temporal_pos=None):
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # Use position features if no temporal_pos provided
        # Position features are at indices 97-99 (last 3 dimensions)
        if temporal_pos is None and x.size(1) >= 100:
            # Use normalized position (feature index 97)
            temporal_pos = x[:, 97]  # Normalized position feature
        
        # Encode nodes with temporal info
        node_emb = self.encoder(x, edge_index, temporal_pos)
        
        # Pool to graph-level
        if self.pooling == 'mean':
            graph_emb = global_mean_pool(node_emb, batch)
        elif self.pooling == 'max':
            graph_emb = global_max_pool(node_emb, batch)
        elif self.pooling == 'both':
            mean_emb = global_mean_pool(node_emb, batch)
            max_emb = global_max_pool(node_emb, batch)
            graph_emb = torch.cat([mean_emb, max_emb], dim=-1)
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")
        
        # Classify
        logits = self.classifier(graph_emb)
        
        return logits


def create_dr_gcn_model(
    input_dim: int,
    hidden_dim: int = 64,
    num_layers: int = 3,
    num_classes: int = 2,
    dropout: float = 0.3,
    pooling: str = 'mean',
):
    return DRGCNClassifier(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout=dropout,
        pooling=pooling,
    )


def create_tmp_model(
    input_dim: int,
    hidden_dim: int = 64,
    num_layers: int = 3,
    heads: int = 4,
    num_classes: int = 2,
    dropout: float = 0.3,
    pooling: str = 'mean',
):
    return TMPClassifier(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        heads=heads,
        num_classes=num_classes,
        dropout=dropout,
        pooling=pooling,
    )
