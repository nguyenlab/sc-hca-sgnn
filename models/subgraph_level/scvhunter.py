import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import scatter
from typing import Optional, Tuple

# Import encoder from graph-level
from models.graph_level.scvhunter import SCVHUNTEREncoder


class FunctionPooling(nn.Module):
    
    def __init__(self, hidden_dim: int, pooling: str = 'mean'):
        super().__init__()
        self.pooling = pooling
        self.hidden_dim = hidden_dim
        
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
        func_mask = subg > 0
        func_nodes = x[func_mask]
        func_assignment = subg[func_mask] - 1
        
        if func_nodes.size(0) == 0:
            out_dim = self.hidden_dim * 2 if self.pooling == 'both' else self.hidden_dim
            return torch.zeros(num_functions, out_dim, device=x.device)
        
        if self.pooling == 'mean':
            return scatter(func_nodes, func_assignment, dim=0, dim_size=num_functions, reduce='mean')
        elif self.pooling == 'max':
            return scatter(func_nodes, func_assignment, dim=0, dim_size=num_functions, reduce='max')
        elif self.pooling == 'both':
            mean_pool = scatter(func_nodes, func_assignment, dim=0, dim_size=num_functions, reduce='mean')
            max_pool = scatter(func_nodes, func_assignment, dim=0, dim_size=num_functions, reduce='max')
            return torch.cat([mean_pool, max_pool], dim=-1)
        elif self.pooling == 'attention':
            scores = self.attention(func_nodes).squeeze(-1)
            scores_exp = torch.exp(scores - scores.max())
            scores_sum = scatter(scores_exp, func_assignment, dim=0, dim_size=num_functions, reduce='sum')
            scores_norm = scores_exp / (scores_sum[func_assignment] + 1e-8)
            weighted = func_nodes * scores_norm.unsqueeze(-1)
            return scatter(weighted, func_assignment, dim=0, dim_size=num_functions, reduce='sum')
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")


class FunctionLevelSCVHUNTER(nn.Module):
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        num_edge_types: int = 8,
        heads: int = 4,
        num_classes: int = 2,
        dropout: float = 0.3,
        pooling: str = 'mean',
        use_node_importance: bool = False,
    ):
        super().__init__()
        
        # Full graph encoder with heterogeneous attention
        self.full_encoder = SCVHUNTEREncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_edge_types=num_edge_types,
            heads=heads,
            dropout=dropout,
            use_node_importance=use_node_importance,
        )
        
        # Subgraph encoder (for intra-function edges)
        self.sub_encoder = SCVHUNTEREncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_edge_types=num_edge_types,
            heads=heads,
            dropout=dropout,
            use_node_importance=use_node_importance,
        )
        
        # Function-level pooling
        pool_dim = hidden_dim * 2 if pooling == 'both' else hidden_dim
        self.pooling_layer = FunctionPooling(hidden_dim, pooling)
        
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
        
        self.hidden_dim = hidden_dim
    
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
        
        # Dual-path encoding with heterogeneous attention
        # Full encoder returns (node_emb, graph_emb) - we only need node_emb
        h_full, _ = self.full_encoder(x, edge_index, edge_attr)
        
        if sub_edge_index.size(1) > 0:
            h_sub, _ = self.sub_encoder(x, sub_edge_index, sub_edge_attr)
            h = h_full + h_sub
        else:
            h = h_full
        
        # Pool by function
        func_embeddings = self.pooling_layer(h, subg, num_functions)
        
        # Classify
        logits = self.classifier(func_embeddings)
        
        return logits


def create_function_level_scvhunter_model(
    input_dim: int = 100,
    hidden_dim: int = 256,
    num_layers: int = 3,
    num_edge_types: int = 8,
    heads: int = 4,
    num_classes: int = 2,
    dropout: float = 0.3,
    pooling: str = 'mean',
    use_node_importance: bool = False,
) -> FunctionLevelSCVHUNTER:
    
    return FunctionLevelSCVHUNTER(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_edge_types=num_edge_types,
        heads=heads,
        num_classes=num_classes,
        dropout=dropout,
        pooling=pooling,
        use_node_importance=use_node_importance,
    )
