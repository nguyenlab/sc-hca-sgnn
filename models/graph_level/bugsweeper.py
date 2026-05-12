import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv
from torch_geometric.nn import global_mean_pool, global_max_pool
from torch_geometric.utils import to_dense_batch
from typing import Optional, Tuple
import math


class CodeGraphPool(nn.Module):
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        # Projection for pooled features
        self.pool_proj = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        cluster: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if cluster is None:
            # Default: pool all nodes in each graph to single supernode
            cluster = batch
        
        num_nodes = x.size(0)
        device = x.device
        
        # Get unique clusters and create mapping
        unique_clusters = torch.unique(cluster)
        num_clusters = unique_clusters.size(0)
        
        # Create cluster to new index mapping
        cluster_map = torch.zeros(cluster.max() + 1, dtype=torch.long, device=device)
        cluster_map[unique_clusters] = torch.arange(num_clusters, device=device)
        new_cluster = cluster_map[cluster]
        
        # Pool node features by cluster (mean pooling within cluster)
        pooled_x = torch.zeros(num_clusters, x.size(1), device=device)
        cluster_counts = torch.zeros(num_clusters, device=device)
        
        pooled_x.scatter_add_(0, new_cluster.unsqueeze(1).expand_as(x), x)
        cluster_counts.scatter_add_(0, new_cluster, torch.ones(num_nodes, device=device))
        pooled_x = pooled_x / cluster_counts.unsqueeze(1).clamp(min=1)
        
        # Project pooled features
        pooled_x = self.pool_proj(pooled_x)
        
        # Pool edges: connect supernodes if their original nodes were connected
        src, dst = edge_index
        new_src = new_cluster[src]
        new_dst = new_cluster[dst]
        
        # Remove self-loops and duplicates
        mask = new_src != new_dst
        pooled_edge_index = torch.stack([new_src[mask], new_dst[mask]], dim=0)
        pooled_edge_index = torch.unique(pooled_edge_index, dim=1)
        
        # Pool batch assignment
        pooled_batch = torch.zeros(num_clusters, dtype=torch.long, device=device)
        for i, c in enumerate(unique_clusters):
            node_idx = (cluster == c).nonzero(as_tuple=True)[0][0]
            pooled_batch[i] = batch[node_idx]
        
        return pooled_x, pooled_edge_index, pooled_batch


class CGNN(nn.Module):
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 1024,
        num_layers: int = 3,
        dropout: float = 0.5,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Input projection with layer norm for stability
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
        )
        
        # GraphSAGE layers with layer norms
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        # First layer: hidden_dim//2 -> hidden_dim
        self.convs.append(SAGEConv(hidden_dim // 2, hidden_dim))
        self.norms.append(nn.LayerNorm(hidden_dim))
        
        # Remaining layers: hidden_dim -> hidden_dim
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))
        
        # CGPool
        self.pool = CodeGraphPool(hidden_dim)
    
    def forward(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor, 
        batch: torch.Tensor,
        cluster: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Input projection
        x = self.input_proj(x)
        
        # GraphSAGE layers with layer norm and residual where possible
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x_in = x
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)  # Always use activation
            x = F.dropout(x, p=self.dropout, training=self.training)
            # Residual connection (skip first layer due to dimension change)
            if i > 0:
                x = x + x_in
        
        # CGPool
        pooled_x, pooled_edge_index, pooled_batch = self.pool(
            x, edge_index, batch, cluster
        )
        
        return pooled_x, pooled_edge_index, pooled_batch


class SecondStageGNN(nn.Module):
    
    def __init__(
        self,
        hidden_dim: int = 1024,
        num_classes: int = 2,
        num_gat_layers: int = 3,
        dropout: float = 0.3,  # Reduced from 0.5
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.dropout = dropout
        
        # GAT layers with layer norms for stability
        self.gat_layers = nn.ModuleList()
        self.gat_norms = nn.ModuleList()
        
        # First GAT: multi-head (4 heads)
        self.gat_layers.append(GATConv(
            hidden_dim, hidden_dim // 4,
            heads=4, concat=True, dropout=dropout
        ))
        self.gat_norms.append(nn.LayerNorm(hidden_dim))
        
        # Remaining GAT: single head
        for _ in range(num_gat_layers - 1):
            self.gat_layers.append(GATConv(
                hidden_dim, hidden_dim,
                heads=1, concat=True, dropout=dropout
            ))
            self.gat_norms.append(nn.LayerNorm(hidden_dim))
        
        # Classifier (3 linear layers)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
    
    def forward(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor, 
        batch: torch.Tensor,
    ) -> torch.Tensor:
        # GAT layers with layer norm and residual connections
        for gat, norm in zip(self.gat_layers, self.gat_norms):
            x_in = x
            x = gat(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x + x_in  # Residual connection
        
        # Global mean pooling
        graph_emb = global_mean_pool(x, batch)
        
        # Classify
        logits = self.classifier(graph_emb)
        
        return logits


class BugSweeper(nn.Module):
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 1024,
        num_classes: int = 2,
        num_sage_layers: int = 3,
        num_gat_layers: int = 3,
        dropout: float = 0.5,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        # Stage 1: CGNN
        self.cgnn = CGNN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_sage_layers,
            dropout=dropout,
        )
        
        # Stage 2: GAT + Classifier
        self.second_stage = SecondStageGNN(
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            num_gat_layers=num_gat_layers,
            dropout=dropout,
        )
    
    def forward(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor, 
        edge_type: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
        cluster: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # Stage 1: CGNN with pooling
        pooled_x, pooled_edge_index, pooled_batch = self.cgnn(
            x, edge_index, batch, cluster
        )
        
        # Stage 2: GAT + Classification
        logits = self.second_stage(pooled_x, pooled_edge_index, pooled_batch)
        
        return logits


class BugSweeperLight(nn.Module):
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_classes: int = 2,
        num_sage_layers: int = 3,
        num_gat_layers: int = 3,
        dropout: float = 0.3,
        pooling: str = 'mean',
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.pooling = pooling
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Stage 1: GraphSAGE layers
        self.sage_layers = nn.ModuleList()
        for _ in range(num_sage_layers):
            self.sage_layers.append(SAGEConv(hidden_dim, hidden_dim))
        self.sage_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_sage_layers)
        ])
        
        # Stage 2: GAT layers (on same graph, simpler pooling)
        self.gat_layers = nn.ModuleList()
        
        # First GAT with multi-head
        self.gat_layers.append(GATConv(
            hidden_dim, hidden_dim // 4,
            heads=4, concat=True, dropout=dropout
        ))
        
        # Remaining GAT layers
        for _ in range(num_gat_layers - 1):
            self.gat_layers.append(GATConv(
                hidden_dim, hidden_dim,
                heads=1, concat=True, dropout=dropout
            ))
        
        self.gat_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_gat_layers)
        ])
        
        # Determine classifier input dim
        pool_dim = hidden_dim * (2 if pooling == 'both' else 1)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(pool_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )
        
        self.dropout = dropout
    
    def forward(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor, 
        edge_type: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass."""
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # Input projection
        x = self.input_proj(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Stage 1: GraphSAGE
        for conv, norm in zip(self.sage_layers, self.sage_norms):
            x_res = x
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x + x_res  # Residual
        
        # Stage 2: GAT
        for conv, norm in zip(self.gat_layers, self.gat_norms):
            x_res = x
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x + x_res  # Residual
        
        # Global pooling
        if self.pooling == 'mean':
            graph_emb = global_mean_pool(x, batch)
        elif self.pooling == 'max':
            graph_emb = global_max_pool(x, batch)
        elif self.pooling == 'both':
            mean_emb = global_mean_pool(x, batch)
            max_emb = global_max_pool(x, batch)
            graph_emb = torch.cat([mean_emb, max_emb], dim=-1)
        else:
            graph_emb = global_mean_pool(x, batch)
        
        # Classify
        logits = self.classifier(graph_emb)
        
        return logits


def create_bugsweeper_model(
    input_dim: int,
    hidden_dim: int = 1024,
    num_classes: int = 2,
    num_sage_layers: int = 3,
    num_gat_layers: int = 3,
    dropout: float = 0.3,  # Reduced from 0.5 for stability
):
    return BugSweeper(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        num_sage_layers=num_sage_layers,
        num_gat_layers=num_gat_layers,
        dropout=dropout,
    )


def create_bugsweeper_light_model(
    input_dim: int,
    hidden_dim: int = 256,
    num_classes: int = 2,
    num_sage_layers: int = 3,
    num_gat_layers: int = 3,
    dropout: float = 0.3,
    pooling: str = 'mean',
):
    return BugSweeperLight(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        num_sage_layers=num_sage_layers,
        num_gat_layers=num_gat_layers,
        dropout=dropout,
        pooling=pooling,
    )
