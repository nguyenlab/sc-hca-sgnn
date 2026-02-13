import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.utils import subgraph
from typing import Optional, Tuple, List


# Edge type constants (from data.py)
EDGE_TYPE_AST = 0       # Structural
EDGE_TYPE_REF = 1       # Reference/data flow
EDGE_TYPE_CFG_NEXT = 2  # Control flow: next
EDGE_TYPE_CFG_TRUE = 3  # Control flow: true branch
EDGE_TYPE_CFG_FALSE = 4 # Control flow: false branch
EDGE_TYPE_CALL = 5      # Function call
EDGE_TYPE_INHERIT = 6   # Inheritance
EDGE_TYPE_GUARD = 7     # Guard condition

# Define which edge types belong to which branch
STRUCTURAL_EDGE_TYPES = [EDGE_TYPE_AST, EDGE_TYPE_INHERIT, EDGE_TYPE_GUARD]
CONTROL_FLOW_EDGE_TYPES = [EDGE_TYPE_REF, EDGE_TYPE_CFG_NEXT, EDGE_TYPE_CFG_TRUE, 
                           EDGE_TYPE_CFG_FALSE, EDGE_TYPE_CALL]


def filter_edges_by_type(
    edge_index: torch.Tensor,
    edge_type: Optional[torch.Tensor],
    allowed_types: List[int],
) -> torch.Tensor:
    if edge_type is None:
        return edge_index
    
    # Create mask for allowed types
    mask = torch.zeros(edge_type.size(0), dtype=torch.bool, device=edge_type.device)
    for t in allowed_types:
        mask = mask | (edge_type == t)
    
    return edge_index[:, mask]


class GATv2Branch(nn.Module):
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        heads: int = 2,
        dropout: float = 0.3,
        use_layer_norm: bool = True,
        use_residual: bool = True,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.heads = heads
        self.use_layer_norm = use_layer_norm
        self.use_residual = use_residual
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # GATv2 layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList() if use_layer_norm else None
        
        for i in range(num_layers):
            if i == 0:
                # First layer: hidden_dim -> hidden_dim (multi-head concat)
                self.convs.append(
                    GATv2Conv(
                        hidden_dim, 
                        hidden_dim // heads, 
                        heads=heads, 
                        concat=True,
                        dropout=dropout,
                        add_self_loops=True,
                    )
                )
            elif i == num_layers - 1:
                # Last layer: hidden_dim -> hidden_dim (single head or averaged)
                self.convs.append(
                    GATv2Conv(
                        hidden_dim,
                        hidden_dim,
                        heads=1,
                        concat=False,
                        dropout=dropout,
                        add_self_loops=True,
                    )
                )
            else:
                # Middle layers: hidden_dim -> hidden_dim (multi-head concat)
                self.convs.append(
                    GATv2Conv(
                        hidden_dim,
                        hidden_dim // heads,
                        heads=heads,
                        concat=True,
                        dropout=dropout,
                        add_self_loops=True,
                    )
                )
            
            if use_layer_norm:
                self.norms.append(nn.LayerNorm(hidden_dim))
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        # Input projection
        x = self.input_proj(x)
        x = F.elu(x)
        x = self.dropout(x)
        
        # GATv2 layers
        for i, conv in enumerate(self.convs):
            x_res = x if self.use_residual else None
            
            x = conv(x, edge_index)
            x = F.elu(x)
            
            if self.use_layer_norm and self.norms is not None:
                x = self.norms[i](x)
            
            # Residual connection
            if self.use_residual and x_res is not None and x.shape == x_res.shape:
                x = x + x_res
            
            x = self.dropout(x)
        
        return x


class BSGVDEncoder(nn.Module):
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        heads: int = 2,
        dropout: float = 0.3,
        use_layer_norm: bool = True,
        use_residual: bool = True,
        pooling: str = 'mean',
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.pooling = pooling
        
        # Branch A: Structural (AST, INHERIT, GUARD edges)
        self.structural_branch = GATv2Branch(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            heads=heads,
            dropout=dropout,
            use_layer_norm=use_layer_norm,
            use_residual=use_residual,
        )
        
        # Branch B: Control Flow (CFG, REF, CALL edges)
        self.control_flow_branch = GATv2Branch(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            heads=heads,
            dropout=dropout,
            use_layer_norm=use_layer_norm,
            use_residual=use_residual,
        )
        
        # Pooling output dimension
        if pooling == 'both':
            self.output_dim = hidden_dim * 4  # 2 branches * 2 pooling methods
        else:
            self.output_dim = hidden_dim * 2  # 2 branches * 1 pooling method
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # Filter edges for each branch
        structural_edges = filter_edges_by_type(edge_index, edge_type, STRUCTURAL_EDGE_TYPES)
        control_flow_edges = filter_edges_by_type(edge_index, edge_type, CONTROL_FLOW_EDGE_TYPES)
        
        # Ensure both branches have some edges (fall back to all edges if empty)
        if structural_edges.size(1) == 0:
            structural_edges = edge_index
        if control_flow_edges.size(1) == 0:
            control_flow_edges = edge_index
        
        # Process through branches
        structural_emb = self.structural_branch(x, structural_edges)  # [N, hidden_dim]
        control_flow_emb = self.control_flow_branch(x, control_flow_edges)  # [N, hidden_dim]
        
        # Concatenate node embeddings
        node_emb = torch.cat([structural_emb, control_flow_emb], dim=1)  # [N, hidden_dim * 2]
        
        # Graph-level pooling for each branch
        if self.pooling == 'mean':
            struct_graph = global_mean_pool(structural_emb, batch)
            cf_graph = global_mean_pool(control_flow_emb, batch)
            graph_emb = torch.cat([struct_graph, cf_graph], dim=1)
        elif self.pooling == 'max':
            struct_graph = global_max_pool(structural_emb, batch)
            cf_graph = global_max_pool(control_flow_emb, batch)
            graph_emb = torch.cat([struct_graph, cf_graph], dim=1)
        elif self.pooling == 'both':
            struct_mean = global_mean_pool(structural_emb, batch)
            struct_max = global_max_pool(structural_emb, batch)
            cf_mean = global_mean_pool(control_flow_emb, batch)
            cf_max = global_max_pool(control_flow_emb, batch)
            graph_emb = torch.cat([struct_mean, struct_max, cf_mean, cf_max], dim=1)
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")
        
        return node_emb, graph_emb


class BSGVDClassifier(nn.Module):
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        heads: int = 2,
        num_classes: int = 2,
        dropout: float = 0.5,
        use_layer_norm: bool = True,
        use_residual: bool = True,
        pooling: str = 'mean',
    ):
        super().__init__()
        
        self.encoder = BSGVDEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            heads=heads,
            dropout=dropout,
            use_layer_norm=use_layer_norm,
            use_residual=use_residual,
            pooling=pooling,
        )
        
        fusion_dim = self.encoder.output_dim
        
        # Fusion and classification head (as in the paper)
        self.fusion_fc = nn.Linear(fusion_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
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
        
        # Fusion
        fused = self.fusion_fc(graph_emb)
        fused = F.relu(fused)
        fused = self.dropout(fused)
        
        # Classification
        logits = self.classifier(fused)
        return logits


class BSGVDFullBimodal(nn.Module):
    
    def __init__(
        self,
        input_dim: int = 100,
        hidden_dim: int = 64,
        num_layers: int = 2,
        heads: int = 2,
        num_classes: int = 2,
        dropout: float = 0.5,
    ):
        super().__init__()
        
        # Branch A: Source Code (ASG)
        self.src_conv1 = GATv2Conv(input_dim, hidden_dim, heads=heads, concat=True, dropout=dropout)
        self.src_conv2 = GATv2Conv(hidden_dim * heads, hidden_dim, heads=1, concat=False, dropout=dropout)
        
        # Branch B: Bytecode (CFG)
        self.byte_conv1 = GATv2Conv(input_dim, hidden_dim, heads=heads, concat=True, dropout=dropout)
        self.byte_conv2 = GATv2Conv(hidden_dim * heads, hidden_dim, heads=1, concat=False, dropout=dropout)
        
        # Fusion & Classification
        self.fusion_fc = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, src_data, byte_data):

        # Source Code Branch
        src_x = F.elu(self.src_conv1(src_data.x, src_data.edge_index))
        src_x = F.elu(self.src_conv2(src_x, src_data.edge_index))
        src_graph_emb = global_mean_pool(src_x, src_data.batch)
        
        # Bytecode Branch
        byte_x = F.elu(self.byte_conv1(byte_data.x, byte_data.edge_index))
        byte_x = F.elu(self.byte_conv2(byte_x, byte_data.edge_index))
        byte_graph_emb = global_mean_pool(byte_x, byte_data.batch)
        
        # Fusion
        combined = torch.cat([src_graph_emb, byte_graph_emb], dim=1)
        fused = F.relu(self.fusion_fc(combined))
        fused = self.dropout(fused)
        
        # Classification
        logits = self.classifier(fused)
        return logits


def create_bsgvd_model(
    input_dim: int = 100,
    hidden_dim: int = 64,
    num_layers: int = 2,
    heads: int = 2,
    num_classes: int = 2,
    dropout: float = 0.3,
    pooling: str = 'mean',
    use_layer_norm: bool = True,
    use_residual: bool = True,
) -> BSGVDClassifier:
    return BSGVDClassifier(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        heads=heads,
        num_classes=num_classes,
        dropout=dropout,
        use_layer_norm=use_layer_norm,
        use_residual=use_residual,
        pooling=pooling,
    )


if __name__ == '__main__':
    # Test BSGVD
    print("Testing BSGVD model...")
    
    batch_size = 4
    num_nodes = 50
    input_dim = 100
    num_classes = 2
    
    # Create dummy data
    x = torch.randn(batch_size * num_nodes, input_dim)
    edge_index = torch.randint(0, num_nodes, (2, num_nodes * 3))
    edge_type = torch.randint(0, 8, (num_nodes * 3,))
    batch = torch.repeat_interleave(torch.arange(batch_size), num_nodes)
    
    # Create model
    model = create_bsgvd_model(
        input_dim=input_dim,
        hidden_dim=128,
        num_layers=2,
        heads=4,
        num_classes=num_classes,
        dropout=0.3,
        pooling='mean',
    )
    
    # Forward pass
    out = model(x, edge_index, edge_type, batch)
    print(f"Output shape: {out.shape}")
    print(f"Expected: [{batch_size}, {num_classes}]")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test with 'both' pooling
    model_both = create_bsgvd_model(
        input_dim=input_dim,
        hidden_dim=128,
        num_layers=3,
        heads=4,
        num_classes=num_classes,
        dropout=0.3,
        pooling='both',
    )
    out_both = model_both(x, edge_index, edge_type, batch)
    print(f"\nWith 'both' pooling - Output shape: {out_both.shape}")
    print(f"Parameters: {sum(p.numel() for p in model_both.parameters()):,}")
    
