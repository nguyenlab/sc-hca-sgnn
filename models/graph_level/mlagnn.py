import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.utils import softmax, add_self_loops
from typing import Optional, Tuple


class AdaptiveAttentionConv(MessagePassing):
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_edge_types: int = 8,
        heads: int = 4,
        concat: bool = True,
        dropout: float = 0.1,
        add_self_loops: bool = True,
    ):
        super().__init__(aggr='add', node_dim=0)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_edge_types = num_edge_types
        self.heads = heads
        self.concat = concat
        self.dropout = dropout
        self._add_self_loops = add_self_loops
        
        # Feature transformation
        self.lin = nn.Linear(in_channels, heads * out_channels, bias=False)
        
        # Attention parameters (adaptive)
        self.att_l = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_r = nn.Parameter(torch.Tensor(1, heads, out_channels))
        
        # Edge type transformation
        self.edge_lin = nn.Linear(num_edge_types, heads, bias=False)
        
        # Channel aggregation weights
        self.channel_att = nn.Sequential(
            nn.Linear(out_channels, out_channels // 4),
            nn.ReLU(),
            nn.Linear(out_channels // 4, out_channels),
            nn.Sigmoid(),
        )
        
        # Adaptive scaling parameter
        self.scale = nn.Parameter(torch.ones(1))
        
        self.dropout_layer = nn.Dropout(dropout)
        self.bias = nn.Parameter(torch.zeros(heads * out_channels if concat else out_channels))
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att_l)
        nn.init.xavier_uniform_(self.att_r)
        nn.init.xavier_uniform_(self.edge_lin.weight)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        
        N = x.size(0)
        
        # Add self-loops
        if self._add_self_loops:
            edge_index, edge_type = self._add_loops(edge_index, edge_type, N)
        
        # Transform features
        x = self.lin(x).view(N, self.heads, self.out_channels)
        
        # Compute attention
        alpha_l = (x * self.att_l).sum(dim=-1)
        alpha_r = (x * self.att_r).sum(dim=-1)
        
        # Get edge type features
        if edge_type is not None:
            edge_type_onehot = F.one_hot(edge_type, self.num_edge_types).float()
            edge_weight = self.edge_lin(edge_type_onehot)  # [E, heads]
        else:
            edge_weight = None
        
        # Propagate
        out = self.propagate(
            edge_index,
            x=x,
            alpha_l=alpha_l,
            alpha_r=alpha_r,
            edge_weight=edge_weight,
            size=None,
        )
        
        # Channel aggregation
        channel_weights = self.channel_att(out.mean(dim=1))  # [N, out_channels]
        out = out * channel_weights.unsqueeze(1)  # Apply channel attention
        
        # Reshape and apply adaptive scaling
        if self.concat:
            out = out.view(N, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)
        
        out = out * self.scale + self.bias
        
        return out
    
    def _add_loops(
        self,
        edge_index: torch.Tensor,
        edge_type: Optional[torch.Tensor],
        num_nodes: int,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        
        loop_index = torch.arange(num_nodes, device=edge_index.device)
        loop_index = loop_index.unsqueeze(0).repeat(2, 1)
        edge_index = torch.cat([edge_index, loop_index], dim=1)
        
        if edge_type is not None:
            # Use a special edge type for self-loops (last type)
            loop_type = torch.full((num_nodes,), self.num_edge_types - 1, 
                                   dtype=edge_type.dtype, device=edge_type.device)
            edge_type = torch.cat([edge_type, loop_type])
        
        return edge_index, edge_type
    
    def message(
        self,
        x_j: torch.Tensor,
        alpha_l_i: torch.Tensor,
        alpha_r_j: torch.Tensor,
        edge_weight: Optional[torch.Tensor],
        index: torch.Tensor,
        ptr: Optional[torch.Tensor],
        size_i: Optional[int],
    ) -> torch.Tensor:
        
        # Attention scores
        alpha = alpha_l_i + alpha_r_j
        
        # Add edge type bias
        if edge_weight is not None:
            alpha = alpha + edge_weight
        
        alpha = F.leaky_relu(alpha, 0.2)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = self.dropout_layer(alpha)
        
        return x_j * alpha.unsqueeze(-1)


class MultiLevelAttentionBlock(nn.Module):
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_edge_types: int = 8,
        heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # Node-level attention
        self.node_att = AdaptiveAttentionConv(
            in_channels=in_channels,
            out_channels=out_channels,
            num_edge_types=num_edge_types,
            heads=heads,
            concat=True,
            dropout=dropout,
        )
        
        # Graph-level attention (attention pooling)
        self.graph_att = nn.Sequential(
            nn.Linear(heads * out_channels, 1),
            nn.Sigmoid(),
        )
        
        # Layer normalization
        self.norm = nn.LayerNorm(heads * out_channels)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        # Node-level attention
        node_emb = self.node_att(x, edge_index, edge_type)
        node_emb = F.elu(node_emb)
        node_emb = self.norm(node_emb)
        node_emb = self.dropout(node_emb)
        
        # Graph-level attention weights
        att_weights = self.graph_att(node_emb)
        
        return node_emb, att_weights


class MLAGNNEncoder(nn.Module):
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        num_edge_types: int = 8,
        heads: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Input projection - project to hidden_dim * heads for consistent dimensions
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * heads),
            nn.LayerNorm(hidden_dim * heads),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Multi-level attention blocks
        self.attention_blocks = nn.ModuleList()
        self.skip_connections = nn.ModuleList()
        
        # All layers use same dimension for proper skip connections
        for i in range(num_layers):
            self.attention_blocks.append(
                MultiLevelAttentionBlock(
                    in_channels=hidden_dim * heads,
                    out_channels=hidden_dim,
                    num_edge_types=num_edge_types,
                    heads=heads,
                    dropout=dropout,
                )
            )
            # Skip connection - identity since dimensions now match
            self.skip_connections.append(nn.Identity())
        
        # Final projection for graph embedding
        self.final_proj = nn.Linear(hidden_dim * heads, hidden_dim)
        
        # Hierarchical aggregation weights
        self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)
        
        self.output_dim = hidden_dim
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # Input projection
        h = self.input_proj(x)
        
        # Store layer outputs for hierarchical aggregation
        layer_outputs = []
        attention_weights = []
        
        # Multi-level attention blocks with skip connections
        for i, (block, skip) in enumerate(zip(self.attention_blocks, self.skip_connections)):
            h_skip = skip(h)
            h_new, att = block(h, edge_index, edge_type, batch)
            
            # Skip connection for all layers (dimensions now match)
            h = h_new + h_skip
            
            layer_outputs.append(h)
            attention_weights.append(att)
        
        # Hierarchical aggregation of layer outputs
        layer_weights = F.softmax(self.layer_weights, dim=0)
        h_agg = sum(w * lo for w, lo in zip(layer_weights, layer_outputs))
        
        # Final projection
        node_emb = self.final_proj(h_agg)
        
        # Attention-weighted graph pooling
        final_att = attention_weights[-1]  # Use last layer's attention
        h_weighted = node_emb * final_att
        
        # Combine mean and attention-weighted pooling
        graph_mean = global_mean_pool(node_emb, batch)
        graph_att = global_add_pool(h_weighted, batch)
        graph_emb = graph_mean + graph_att
        
        return node_emb, graph_emb


class MLAGNNClassifier(nn.Module):
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        num_edge_types: int = 8,
        heads: int = 4,
        num_classes: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        
        self.encoder = MLAGNNEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_edge_types=num_edge_types,
            heads=heads,
            dropout=dropout,
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
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


def create_mlagnn_model(
    input_dim: int = 100,
    hidden_dim: int = 256,
    num_layers: int = 3,
    num_edge_types: int = 8,
    heads: int = 4,
    num_classes: int = 2,
    dropout: float = 0.3,
) -> MLAGNNClassifier:
    
    return MLAGNNClassifier(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_edge_types=num_edge_types,
        heads=heads,
        num_classes=num_classes,
        dropout=dropout,
    )


if __name__ == '__main__':
    # Test ML-AGNN
    print("Testing ML-AGNN model...")
    
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
    model = create_mlagnn_model(
        input_dim=input_dim,
        hidden_dim=128,
        num_layers=3,
        num_classes=num_classes,
    )
    
    # Forward pass
    out = model(x, edge_index, edge_type, batch)
    print(f"Output shape: {out.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("✓ ML-AGNN test passed!")
