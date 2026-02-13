import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    TransformerConv,
    GATConv,
    SAGEConv,
    global_mean_pool,
    global_max_pool,
)
from torch_geometric.utils import scatter
from typing import Optional, Tuple, List

from models.data import NUM_EDGE_TYPES


class DualPathEncoder(nn.Module):
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.3,
        conv_type: str = 'transformer',  # 'transformer', 'gat', 'sage'
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.conv_type = conv_type
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Full graph convolution layers
        self.full_convs = nn.ModuleList()
        # Subgraph (intra-function) convolution layers
        self.sub_convs = nn.ModuleList()
        
        for i in range(num_layers):
            in_dim = hidden_dim
            out_dim = hidden_dim
            
            if conv_type == 'transformer':
                self.full_convs.append(
                    TransformerConv(in_dim, out_dim // heads, heads=heads, dropout=dropout)
                )
                self.sub_convs.append(
                    TransformerConv(in_dim, out_dim // heads, heads=heads, dropout=dropout)
                )
            elif conv_type == 'gat':
                self.full_convs.append(
                    GATConv(in_dim, out_dim // heads, heads=heads, dropout=dropout)
                )
                self.sub_convs.append(
                    GATConv(in_dim, out_dim // heads, heads=heads, dropout=dropout)
                )
            elif conv_type == 'sage':
                self.full_convs.append(SAGEConv(in_dim, out_dim))
                self.sub_convs.append(SAGEConv(in_dim, out_dim))
            else:
                raise ValueError(f"Unknown conv_type: {conv_type}")
        
        # Layer norms
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        sub_edge_index: torch.Tensor,
    ) -> torch.Tensor:
        
        # Input projection
        h = self.input_proj(x)
        h = F.relu(h)
        h = self.dropout(h)
        
        # Dual-path convolutions
        for i in range(self.num_layers):
            # Full graph path
            h_full = self.full_convs[i](h, edge_index)
            
            # Subgraph path (only if we have subgraph edges)
            if sub_edge_index.size(1) > 0:
                h_sub = self.sub_convs[i](h, sub_edge_index)
                # Combine both paths
                h_combined = h_full + h_sub
            else:
                h_combined = h_full
            
            # Residual connection + normalization
            h = h + h_combined
            h = self.layer_norms[i](h)
            h = F.relu(h)
            h = self.dropout(h)
        
        return h


class FunctionPooling(nn.Module):
    
    def __init__(
        self,
        hidden_dim: int,
        pooling: str = 'mean',
    ):
        super().__init__()
        self.pooling = pooling
        
        if pooling == 'attention':
            self.attention = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.Tanh(),
                nn.Linear(hidden_dim // 2, 1),
            )
    
    def forward(
        self,
        x: torch.Tensor,
        subg: torch.Tensor,
        num_functions: int,
    ) -> torch.Tensor:
        
        # Only consider nodes belonging to a function (subg > 0)
        func_mask = subg > 0
        func_nodes = x[func_mask]
        func_assignment = subg[func_mask] - 1  # 0-indexed
        
        if func_nodes.size(0) == 0:
            # No function nodes, return zeros
            return torch.zeros(num_functions, x.size(1), device=x.device)
        
        if self.pooling == 'mean':
            return scatter(func_nodes, func_assignment, dim=0, dim_size=num_functions, reduce='mean')
        elif self.pooling == 'max':
            return scatter(func_nodes, func_assignment, dim=0, dim_size=num_functions, reduce='max')
        elif self.pooling == 'attention':
            # Compute attention scores
            scores = self.attention(func_nodes).squeeze(-1)  # [num_func_nodes]
            
            # Softmax within each function
            scores_exp = torch.exp(scores)
            scores_sum = scatter(scores_exp, func_assignment, dim=0, dim_size=num_functions, reduce='sum')
            scores_norm = scores_exp / (scores_sum[func_assignment] + 1e-8)
            
            # Weighted sum
            weighted = func_nodes * scores_norm.unsqueeze(-1)
            return scatter(weighted, func_assignment, dim=0, dim_size=num_functions, reduce='sum')
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")


class FunctionLevelClassifier(nn.Module):
    
    def __init__(
        self,
        hidden_dim: int,
        num_classes: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )
    
    def forward(self, func_embeddings: torch.Tensor) -> torch.Tensor:
        return self.classifier(func_embeddings)


class FunctionLevelGNN(nn.Module):
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        heads: int = 4,
        num_classes: int = 2,
        dropout: float = 0.3,
        conv_type: str = 'transformer',
        pooling: str = 'mean',
    ):
        super().__init__()
        
        self.encoder = DualPathEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            heads=heads,
            dropout=dropout,
            conv_type=conv_type,
        )
        
        self.pooling = FunctionPooling(
            hidden_dim=hidden_dim,
            pooling=pooling,
        )
        
        self.classifier = FunctionLevelClassifier(
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
        )
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        sub_edge_index: torch.Tensor,
        subg: torch.Tensor,
        num_functions: int,
        edge_attr: Optional[torch.Tensor] = None,
        sub_edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        
        # Encode nodes
        h = self.encoder(x, edge_index, sub_edge_index)
        
        # Pool by function
        func_embeddings = self.pooling(h, subg, num_functions)
        
        # Classify
        logits = self.classifier(func_embeddings)
        
        return logits
    
    def forward_batch(self, batch) -> torch.Tensor:

        # Get total number of functions in batch
        # We need to handle the subg tensor properly for batched data
        device = batch.x.device
        
        # Encode nodes
        h = self.encoder(batch.x, batch.edge_index, batch.sub_edge_index)
        
        # Pool by function for each graph in batch
        # Need to offset subg for proper batching
        all_func_embeddings = []
        
        ptr = batch.ptr if hasattr(batch, 'ptr') else None
        
        if ptr is not None:
            # Process each graph separately
            for i in range(len(ptr) - 1):
                start, end = ptr[i].item(), ptr[i + 1].item()
                node_h = h[start:end]
                node_subg = batch.subg[start:end]
                num_funcs = batch.num_functions[i] if hasattr(batch.num_functions, '__getitem__') else batch.num_functions
                
                if isinstance(num_funcs, torch.Tensor):
                    num_funcs = num_funcs.item()
                
                func_emb = self.pooling(node_h, node_subg, num_funcs)
                all_func_embeddings.append(func_emb)
            
            func_embeddings = torch.cat(all_func_embeddings, dim=0)
        else:
            # Single graph
            num_funcs = batch.num_functions
            if isinstance(num_funcs, torch.Tensor):
                num_funcs = num_funcs.item()
            func_embeddings = self.pooling(h, batch.subg, num_funcs)
        
        # Classify
        logits = self.classifier(func_embeddings)
        
        return logits


class FunctionLevelGNNWithGraphOutput(nn.Module):
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        heads: int = 4,
        num_classes: int = 2,
        dropout: float = 0.3,
        conv_type: str = 'transformer',
        pooling: str = 'mean',
    ):
        super().__init__()
        
        self.function_gnn = FunctionLevelGNN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            heads=heads,
            num_classes=num_classes,
            dropout=dropout,
            conv_type=conv_type,
            pooling=pooling,
        )
        
        # Graph-level classifier (from function logits)
        self.graph_classifier = nn.Sequential(
            nn.Linear(num_classes, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, num_classes),
        )
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        sub_edge_index: torch.Tensor,
        subg: torch.Tensor,
        num_functions: int,
        edge_attr: Optional[torch.Tensor] = None,
        sub_edge_attr: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        # Function-level predictions
        func_logits = self.function_gnn(
            x, edge_index, sub_edge_index, subg, num_functions,
            edge_attr, sub_edge_attr
        )
        
        # Graph-level prediction (max of function probabilities)
        func_probs = F.softmax(func_logits, dim=-1)
        graph_probs = func_probs.max(dim=0)[0].unsqueeze(0)  # [1, num_classes]
        graph_logits = self.graph_classifier(graph_probs)
        
        return func_logits, graph_logits


# ============================================================================
# Factory Functions
# ============================================================================

def create_function_level_model(
    input_dim: int,
    hidden_dim: int = 128,
    num_layers: int = 3,
    heads: int = 4,
    num_classes: int = 2,
    dropout: float = 0.3,
    conv_type: str = 'transformer',
    pooling: str = 'mean',
) -> FunctionLevelGNN:
    
    return FunctionLevelGNN(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        heads=heads,
        num_classes=num_classes,
        dropout=dropout,
        conv_type=conv_type,
        pooling=pooling,
    )


def create_function_level_model_with_graph(
    input_dim: int,
    hidden_dim: int = 128,
    num_layers: int = 3,
    heads: int = 4,
    num_classes: int = 2,
    dropout: float = 0.3,
    conv_type: str = 'transformer',
    pooling: str = 'mean',
) -> FunctionLevelGNNWithGraphOutput:
    
    return FunctionLevelGNNWithGraphOutput(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        heads=heads,
        num_classes=num_classes,
        dropout=dropout,
        conv_type=conv_type,
        pooling=pooling,
    )
