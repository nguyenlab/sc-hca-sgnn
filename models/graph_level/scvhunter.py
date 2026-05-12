import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool, global_max_pool
from torch_geometric.utils import softmax
from typing import Optional, Tuple


class HeterogeneousAttentionLayer(MessagePassing):
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_edge_types: int = 8,
        heads: int = 4,
        concat: bool = True,
        dropout: float = 0.1,
        negative_slope: float = 0.2,
    ):
        super().__init__(aggr='add', node_dim=0)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_edge_types = num_edge_types
        self.heads = heads
        self.concat = concat
        self.dropout = dropout
        self.negative_slope = negative_slope
        
        # Linear transformations for query, key, value
        self.lin_query = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.lin_key = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.lin_value = nn.Linear(in_channels, heads * out_channels, bias=False)
        
        # Edge-type specific attention bias
        self.edge_type_embedding = nn.Embedding(num_edge_types, heads)
        
        # Attention parameters per head
        self.att_src = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_dst = nn.Parameter(torch.Tensor(1, heads, out_channels))
        
        # Output projection
        if concat:
            self.lin_out = nn.Linear(heads * out_channels, heads * out_channels)
        else:
            self.lin_out = nn.Linear(out_channels, out_channels)
        
        self.dropout_layer = nn.Dropout(dropout)
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.lin_query.weight)
        nn.init.xavier_uniform_(self.lin_key.weight)
        nn.init.xavier_uniform_(self.lin_value.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)
        nn.init.xavier_uniform_(self.lin_out.weight)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        
        N = x.size(0)
        
        # Linear projections
        query = self.lin_query(x).view(N, self.heads, self.out_channels)
        key = self.lin_key(x).view(N, self.heads, self.out_channels)
        value = self.lin_value(x).view(N, self.heads, self.out_channels)
        
        # Get edge type embeddings
        if edge_type is not None:
            edge_type_emb = self.edge_type_embedding(edge_type)  # [E, heads]
        else:
            edge_type_emb = None
        
        # Message passing
        out = self.propagate(
            edge_index,
            query=query,
            key=key,
            value=value,
            edge_type_emb=edge_type_emb,
            size=None,
        )
        
        # Reshape output
        if self.concat:
            out = out.view(N, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)
        
        out = self.lin_out(out)
        
        return out
    
    def message(
        self,
        query_i: torch.Tensor,
        key_j: torch.Tensor,
        value_j: torch.Tensor,
        edge_type_emb: Optional[torch.Tensor],
        index: torch.Tensor,
        ptr: Optional[torch.Tensor],
        size_i: Optional[int],
    ) -> torch.Tensor:
        
        # Compute attention scores
        alpha = (query_i * self.att_src).sum(dim=-1) + (key_j * self.att_dst).sum(dim=-1)
        
        # Add edge type bias
        if edge_type_emb is not None:
            alpha = alpha + edge_type_emb
        
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = self.dropout_layer(alpha)
        
        # Weighted message
        return value_j * alpha.unsqueeze(-1)


class NodeImportanceModule(nn.Module):
    
    def __init__(self, hidden_dim: int, num_node_types: int = 97):
        super().__init__()
        
        # Node type importance embedding
        self.node_type_importance = nn.Embedding(num_node_types, 1)
        
        # Learned importance from features
        self.importance_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
        
    def forward(self, x: torch.Tensor, node_type: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Learned importance from features
        importance = self.importance_net(x)
        
        # If node types provided, combine with type-based importance
        if node_type is not None:
            type_importance = torch.sigmoid(self.node_type_importance(node_type))
            importance = importance * type_importance
        
        return importance


class SCVHUNTEREncoder(nn.Module):
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        num_edge_types: int = 8,
        heads: int = 4,
        dropout: float = 0.3,
        use_node_importance: bool = True,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_node_importance = use_node_importance
        
        # Input projection to hidden_dim (not hidden_dim * heads)
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Heterogeneous attention layers
        self.han_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        
        for i in range(num_layers):
            self.han_layers.append(
                HeterogeneousAttentionLayer(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim // heads,  # Each head outputs this size
                    num_edge_types=num_edge_types,
                    heads=heads,
                    concat=True,  # Concatenate heads to get hidden_dim back
                    dropout=dropout,
                )
            )
            self.layer_norms.append(nn.LayerNorm(hidden_dim))
        
        # Node importance module
        if use_node_importance:
            self.node_importance = NodeImportanceModule(hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.output_dim = hidden_dim
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        # Input projection
        h = self.input_proj(x)
        
        # HAN layers with residual connections
        for i, (han, ln) in enumerate(zip(self.han_layers, self.layer_norms)):
            h_res = h
            h = han(h, edge_index, edge_type)
            h = ln(h)
            h = F.elu(h)
            h = self.dropout(h)
            # Residual connection
            h = h + h_res
        
        # Compute node importance weights
        if self.use_node_importance:
            importance = self.node_importance(h)
            h_weighted = h * importance
        else:
            h_weighted = h
        
        node_emb = h_weighted
        
        # Graph-level readout with importance-weighted pooling
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # Mean and max pooling
        graph_mean = global_mean_pool(node_emb, batch)
        graph_max = global_max_pool(node_emb, batch)
        graph_emb = graph_mean + graph_max  # Combine pooling methods
        
        return node_emb, graph_emb


class SCVHUNTERClassifier(nn.Module):
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        num_edge_types: int = 8,
        heads: int = 4,
        num_classes: int = 2,
        dropout: float = 0.3,
        use_node_importance: bool = False,  # Disabled by default for stability
    ):
        super().__init__()
        
        self.encoder = SCVHUNTEREncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_edge_types=num_edge_types,
            heads=heads,
            dropout=dropout,
            use_node_importance=use_node_importance,
        )
        
        # Simplified classification head (like other working models)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        
        _, graph_emb = self.encoder(x, edge_index, edge_type, batch)
        logits = self.classifier(graph_emb)
        return logits


def create_scvhunter_model(
    input_dim: int = 100,
    hidden_dim: int = 256,
    num_layers: int = 3,
    num_edge_types: int = 8,
    heads: int = 4,
    num_classes: int = 2,
    dropout: float = 0.3,
    use_node_importance: bool = False,  # Disabled by default for stability
) -> SCVHUNTERClassifier:
    
    return SCVHUNTERClassifier(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_edge_types=num_edge_types,
        heads=heads,
        num_classes=num_classes,
        dropout=dropout,
        use_node_importance=use_node_importance,
    )


if __name__ == '__main__':
    # Test SCVHUNTER
    print("Testing SCVHUNTER model...")
    
    batch_size = 4
    num_nodes = 50
    input_dim = 100
    num_classes = 7
    
    # Create dummy data
    x = torch.randn(batch_size * num_nodes, input_dim)
    edge_index = torch.randint(0, num_nodes, (2, num_nodes * 3))
    edge_type = torch.randint(0, 8, (num_nodes * 3,))
    batch = torch.repeat_interleave(torch.arange(batch_size), num_nodes)
    
    # Create model
    model = create_scvhunter_model(
        input_dim=input_dim,
        hidden_dim=128,
        num_layers=3,
        num_classes=num_classes,
    )
    
    # Forward pass
    out = model(x, edge_index, edge_type, batch)
    print(f"Output shape: {out.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("✓ SCVHUNTER test passed!")
