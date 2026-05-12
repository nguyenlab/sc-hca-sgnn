import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    MessagePassing,
    TransformerConv,
    GATv2Conv,
    LayerNorm,
    RGCNConv,
)
from torch_geometric.utils import softmax, scatter
from typing import Optional, Tuple, List


# ============================================================================
# Edge-Type-Aware Convolution Layer
# ============================================================================

class EdgeTypeAwareConv(MessagePassing):
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_edge_types: int = 8,
        heads: int = 4,
        dropout: float = 0.1,
        concat: bool = False,
    ):
        super().__init__(aggr='add', node_dim=0)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_edge_types = num_edge_types
        self.heads = heads
        self.head_dim = out_channels // heads
        self.dropout = dropout
        self.concat = concat
        
        # Node transformations
        self.q_proj = nn.Linear(in_channels, out_channels)
        self.k_proj = nn.Linear(in_channels, out_channels)
        self.v_proj = nn.Linear(in_channels, out_channels)
        
        # Edge-type embeddings
        self.edge_type_emb = nn.Embedding(num_edge_types, out_channels)
        
        # Edge-type attention bias
        self.edge_type_attn = nn.Parameter(torch.zeros(num_edge_types, heads))
        
        # Output projection
        self.out_proj = nn.Linear(out_channels, out_channels)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.normal_(self.edge_type_attn, mean=0, std=0.02)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if edge_attr is None:
            edge_attr = torch.zeros(edge_index.size(1), dtype=torch.long, device=x.device)
        
        # Project nodes
        q = self.q_proj(x).view(-1, self.heads, self.head_dim)
        k = self.k_proj(x).view(-1, self.heads, self.head_dim)
        v = self.v_proj(x).view(-1, self.heads, self.head_dim)
        
        # Get edge type embeddings
        edge_emb = self.edge_type_emb(edge_attr).view(-1, self.heads, self.head_dim)
        
        # Message passing
        out = self.propagate(
            edge_index, q=q, k=k, v=v,
            edge_emb=edge_emb, edge_attr=edge_attr,
        )
        
        # Reshape and project
        out = out.view(-1, self.out_channels)
        out = self.out_proj(out)
        
        return out
    
    def message(
        self,
        q_i: torch.Tensor,
        k_j: torch.Tensor,
        v_j: torch.Tensor,
        edge_emb: torch.Tensor,
        edge_attr: torch.Tensor,
        index: torch.Tensor,
        ptr: Optional[torch.Tensor],
        size_i: Optional[int],
    ) -> torch.Tensor:
        # Compute attention scores with edge bias
        # q_i: [E, heads, head_dim], k_j: [E, heads, head_dim]
        attn = (q_i * (k_j + edge_emb)).sum(dim=-1) / (self.head_dim ** 0.5)
        
        # Add edge-type attention bias
        edge_bias = self.edge_type_attn[edge_attr]  # [E, heads]
        attn = attn + edge_bias
        
        # Softmax over incoming edges
        attn = softmax(attn, index, ptr, size_i)
        attn = F.dropout(attn, p=self.dropout, training=self.training)
        
        # Weight values
        out = attn.unsqueeze(-1) * (v_j + edge_emb)
        
        return out


# ============================================================================
# Cross-Attention Module
# ============================================================================

class FunctionContractCrossAttention(nn.Module):
    
    def __init__(
        self,
        hidden_dim: int,
        heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.head_dim = hidden_dim // heads
        
        # Function queries, contract keys/values
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Output projection with gating
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )
        
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNorm(hidden_dim)
    
    def forward(
        self,
        func_emb: torch.Tensor,
        contract_context: torch.Tensor,
        func_batch: torch.Tensor,
    ) -> torch.Tensor:
        # Project
        Q = self.q_proj(func_emb).view(-1, self.heads, self.head_dim)
        K = self.k_proj(contract_context).view(-1, self.heads, self.head_dim)
        V = self.v_proj(contract_context).view(-1, self.heads, self.head_dim)
        
        # Self-attention among functions in same contract
        # For simplicity, we use scaled dot-product attention
        # In practice, should mask to only attend within same contract
        attn_scores = torch.einsum('ihd,jhd->ijh', Q, K) / (self.head_dim ** 0.5)
        
        # Create mask for same-contract functions
        batch_mask = func_batch.unsqueeze(0) == func_batch.unsqueeze(1)  # [F, F]
        attn_scores = attn_scores.masked_fill(~batch_mask.unsqueeze(-1), float('-inf'))
        
        attn_weights = F.softmax(attn_scores, dim=1)
        attn_weights = self.dropout(attn_weights)
        
        # Aggregate
        context = torch.einsum('ijh,jhd->ihd', attn_weights, V)
        context = context.reshape(-1, self.hidden_dim)
        context = self.out_proj(context)
        
        # Gated residual
        gate = self.gate(torch.cat([func_emb, context], dim=-1))
        output = gate * context + (1 - gate) * func_emb
        
        return self.norm(output)


# ============================================================================
# Hierarchical Function Pooling
# ============================================================================

class HierarchicalFunctionPooling(nn.Module):
    
    def __init__(
        self,
        hidden_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Node importance scoring
        self.node_importance = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        
        # Multi-scale aggregators
        self.scale_weights = nn.Parameter(torch.ones(3) / 3)  # mean, max, importance-weighted
        
        # Output transformation
        self.output_transform = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
    
    def forward(
        self,
        x: torch.Tensor,
        subg: torch.Tensor,
        num_functions: int,
    ) -> torch.Tensor:
        # Filter to function nodes only
        func_mask = subg > 0
        func_nodes = x[func_mask]
        func_assignment = subg[func_mask] - 1
        
        if func_nodes.size(0) == 0:
            return torch.zeros(num_functions, self.hidden_dim, device=x.device)
        
        # Compute node importance scores
        importance = self.node_importance(func_nodes).squeeze(-1)
        importance = F.softmax(
            importance - scatter(importance, func_assignment, dim=0, 
                                dim_size=num_functions, reduce='max')[func_assignment],
            dim=0
        )
        
        # Multi-scale aggregation
        # 1. Mean pooling
        mean_pool = scatter(func_nodes, func_assignment, dim=0, 
                           dim_size=num_functions, reduce='mean')
        
        # 2. Max pooling
        max_pool = scatter(func_nodes, func_assignment, dim=0,
                          dim_size=num_functions, reduce='max')
        
        # 3. Importance-weighted pooling
        weighted_nodes = func_nodes * importance.unsqueeze(-1)
        imp_pool = scatter(weighted_nodes, func_assignment, dim=0,
                          dim_size=num_functions, reduce='sum')
        
        # Combine scales with learned weights
        weights = F.softmax(self.scale_weights, dim=0)
        combined = weights[0] * mean_pool + weights[1] * max_pool + weights[2] * imp_pool
        
        return self.output_transform(combined)


# ============================================================================
# Gated Local-Global Fusion
# ============================================================================

class GatedLocalGlobalFusion(nn.Module):
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        
        # Gate network
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        
        # Local/global transformations
        self.local_transform = nn.Linear(hidden_dim, hidden_dim)
        self.global_transform = nn.Linear(hidden_dim, hidden_dim)
        
        self.norm = LayerNorm(hidden_dim)
    
    def forward(
        self,
        local_emb: torch.Tensor,
        global_emb: torch.Tensor,
    ) -> torch.Tensor:
        # Compute adaptive gate
        gate = self.gate(torch.cat([local_emb, global_emb], dim=-1))
        
        # Transform and fuse
        local_feat = self.local_transform(local_emb)
        global_feat = self.global_transform(global_emb)
        
        fused = gate * local_feat + (1 - gate) * global_feat
        
        return self.norm(fused)


# ============================================================================
# Node-Level Attention (from HierarchicalMultiEdgeGCN)
# ============================================================================

class NodeLevelAttention(nn.Module):
    
    def __init__(self, hidden_dim: int):
        super().__init__()
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
        # Compute attention scores
        attn_scores = self.attention(x).squeeze(-1)  # [num_nodes]
        
        # Only apply to function nodes (subg > 0)
        func_mask = subg > 0
        if not func_mask.any():
            return x
        
        func_assignment = subg.clone()
        func_assignment[~func_mask] = 0  # Temporary assignment for non-function nodes
        
        # Softmax within each function
        # Subtract max for numerical stability
        max_scores = scatter(attn_scores, func_assignment, dim=0,
                            dim_size=num_functions + 1, reduce='max')
        scores_normalized = attn_scores - max_scores[func_assignment]
        
        # Exp and normalize
        exp_scores = torch.exp(scores_normalized)
        exp_scores[~func_mask] = 0  # Zero out non-function nodes
        sum_exp = scatter(exp_scores, func_assignment, dim=0,
                         dim_size=num_functions + 1, reduce='sum')
        
        attn_weights = exp_scores / (sum_exp[func_assignment] + 1e-8)
        
        # Apply attention weights
        weighted_x = x * attn_weights.unsqueeze(-1)
        
        return weighted_x


# ============================================================================
# Main Model: Hierarchical Cross-Attention Subgraph GNN
# ============================================================================

class HierarchicalCrossAttentionSGNN(nn.Module):
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_edge_types: int = 8,
        heads: int = 4,
        num_classes: int = 2,
        dropout: float = 0.2,
        use_cross_attention: bool = True,
        pooling: str = 'hierarchical',  # 'hierarchical', 'mean', 'max', 'attention'
        edge_dropout: float = 0.0,  # Edge dropout for data augmentation
        # === Ablation flags ===
        use_rgcn: bool = False,  # Use RGCNConv instead of EdgeTypeAwareConv
        use_dual_channel: bool = True,  # Enable Dual-Channel Message Passing (full + subgraph)
        use_gated_fusion: bool = True,  # Enable gated local-global fusion
        use_node_attention: bool = False,  # Enable node-level attention before pooling
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_cross_attention = use_cross_attention
        self.pooling_type = pooling
        self.edge_dropout = edge_dropout
        
        # Ablation flags
        self.use_rgcn = use_rgcn
        self.use_dual_channel = use_dual_channel
        self.use_gated_fusion = use_gated_fusion
        self.use_node_attention = use_node_attention
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Build convolution layers based on use_rgcn flag
        if use_rgcn:
            # Use RGCNConv (from HierarchicalMultiEdgeGCN style)
            self.full_convs = nn.ModuleList([
                RGCNConv(
                    hidden_dim, hidden_dim,
                    num_relations=num_edge_types,
                    num_bases=min(num_edge_types, 4),  # Basis decomposition
                )
                for _ in range(num_layers)
            ])
            
            if use_dual_channel:
                self.sub_convs = nn.ModuleList([
                    RGCNConv(
                        hidden_dim, hidden_dim,
                        num_relations=num_edge_types,
                        num_bases=min(num_edge_types, 4),
                    )
                    for _ in range(num_layers)
                ])
        else:
            # Use EdgeTypeAwareConv (default HCA-SGNN)
            self.full_convs = nn.ModuleList([
                EdgeTypeAwareConv(
                    hidden_dim, hidden_dim,
                    num_edge_types=num_edge_types,
                    heads=heads,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ])
            
            if use_dual_channel:
                self.sub_convs = nn.ModuleList([
                    EdgeTypeAwareConv(
                        hidden_dim, hidden_dim,
                        num_edge_types=num_edge_types,
                        heads=heads,
                        dropout=dropout,
                    )
                    for _ in range(num_layers)
                ])
        
        # Gated fusion at each layer (only if dual-channel and gated fusion enabled)
        if use_dual_channel and use_gated_fusion:
            self.fusions = nn.ModuleList([
                GatedLocalGlobalFusion(hidden_dim)
                for _ in range(num_layers)
            ])
        
        # Layer norms
        self.norms = nn.ModuleList([
            LayerNorm(hidden_dim)
            for _ in range(num_layers)
        ])
        
        # Node-level attention (from HierarchicalMultiEdgeGCN)
        if use_node_attention:
            self.node_attention = NodeLevelAttention(hidden_dim)
        
        # Pooling layer based on type
        if pooling == 'hierarchical':
            self.pooling = HierarchicalFunctionPooling(hidden_dim, dropout)
        elif pooling == 'attention':
            # Simple attention pooling
            self.pooling_attn = nn.Sequential(
                nn.Linear(hidden_dim, 1),
            )
        # For 'mean' and 'max', we use scatter operations directly
        
        # Cross-attention between functions and contract
        if use_cross_attention:
            self.cross_attention = FunctionContractCrossAttention(
                hidden_dim, heads, dropout
            )
        
        # Classification head with residual
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        sub_edge_index: torch.Tensor,
        subg: torch.Tensor,
        num_functions: int,
        edge_attr: Optional[torch.Tensor] = None,
        sub_edge_attr: Optional[torch.Tensor] = None,
        func_batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Input projection
        h = self.input_proj(x)
        
        # Ensure edge_attr is not None for RGCNConv
        if self.use_rgcn:
            if edge_attr is None:
                edge_attr = torch.zeros(edge_index.size(1), dtype=torch.long, device=x.device)
            if sub_edge_attr is None and sub_edge_index.size(1) > 0:
                sub_edge_attr = torch.zeros(sub_edge_index.size(1), dtype=torch.long, device=x.device)
        
        # Apply edge dropout during training
        if self.training and self.edge_dropout > 0:
            edge_index, edge_attr = self._edge_dropout(edge_index, edge_attr)
            sub_edge_index, sub_edge_attr = self._edge_dropout(sub_edge_index, sub_edge_attr)
        
        # Hierarchical message passing with optional dual-channel and gated fusion
        for i in range(self.num_layers):
            # Full graph convolution (global context)
            h_full = self.full_convs[i](h, edge_index, edge_attr)
            
            if self.use_dual_channel:
                # Subgraph convolution (local function context)
                if sub_edge_index.size(1) > 0:
                    h_sub = self.sub_convs[i](h, sub_edge_index, sub_edge_attr)
                else:
                    h_sub = h_full
                
                if self.use_gated_fusion:
                    # Gated local-global fusion
                    h_fused = self.fusions[i](h_sub, h_full)
                else:
                    # Simple addition fusion (ablation)
                    h_fused = 0.5 * (h_sub + h_full)
            else:
                # Single channel: only use full graph convolution
                h_fused = h_full
            
            # Residual connection + normalization
            h = self.norms[i](h + self.dropout(h_fused))
        
        # Apply node-level attention if enabled (from HierarchicalMultiEdgeGCN)
        if self.use_node_attention:
            h = self.node_attention(h, subg, num_functions)
        
        # Function-level pooling based on type
        func_emb = self._pool_functions(h, subg, num_functions)
        
        # Cross-attention with contract context
        if self.use_cross_attention and func_batch is not None:
            # Use mean of all function embeddings as contract context
            func_emb = self.cross_attention(func_emb, func_emb, func_batch)
        
        # Classification
        logits = self.classifier(func_emb)
        
        return logits
    
    def _edge_dropout(
        self,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        
        if edge_index.size(1) == 0:
            return edge_index, edge_attr
        
        # Keep edges with probability (1 - edge_dropout)
        mask = torch.rand(edge_index.size(1), device=edge_index.device) > self.edge_dropout
        edge_index = edge_index[:, mask]
        if edge_attr is not None:
            edge_attr = edge_attr[mask]
        
        return edge_index, edge_attr
    
    def _pool_functions(
        self,
        h: torch.Tensor,
        subg: torch.Tensor,
        num_functions: int,
    ) -> torch.Tensor:
        
        if self.pooling_type == 'hierarchical':
            return self.pooling(h, subg, num_functions)
        
        # Create mask for function nodes (subg > 0)
        func_mask = subg > 0
        if not func_mask.any():
            return torch.zeros(num_functions, self.hidden_dim, device=h.device)
        
        func_nodes = h[func_mask]
        func_assignment = subg[func_mask] - 1  # 0-indexed
        
        if self.pooling_type == 'mean':
            return scatter(func_nodes, func_assignment, dim=0, 
                          dim_size=num_functions, reduce='mean')
        elif self.pooling_type == 'max':
            return scatter(func_nodes, func_assignment, dim=0,
                          dim_size=num_functions, reduce='max')
        elif self.pooling_type == 'attention':
            # Compute attention scores
            attn_scores = self.pooling_attn(func_nodes).squeeze(-1)  # [N]
            attn_weights = softmax(attn_scores, func_assignment, num_nodes=num_functions)
            
            # Weighted sum
            weighted = func_nodes * attn_weights.unsqueeze(-1)
            return scatter(weighted, func_assignment, dim=0,
                          dim_size=num_functions, reduce='sum')
        else:
            raise ValueError(f"Unknown pooling type: {self.pooling_type}")


# ============================================================================
# Factory Function
# ============================================================================

def create_hca_sgnn_model(
    input_dim: int = 100,
    hidden_dim: int = 256,
    num_layers: int = 4,
    num_edge_types: int = 8,
    heads: int = 4,
    num_classes: int = 2,
    dropout: float = 0.2,
    use_cross_attention: bool = True,
    pooling: str = 'hierarchical',
    edge_dropout: float = 0.0,
    # Ablation flags
    use_rgcn: bool = False,
    use_dual_channel: bool = True,
    use_gated_fusion: bool = True,
    use_node_attention: bool = False,
) -> HierarchicalCrossAttentionSGNN:
    
    return HierarchicalCrossAttentionSGNN(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_edge_types=num_edge_types,
        heads=heads,
        num_classes=num_classes,
        dropout=dropout,
        use_cross_attention=use_cross_attention,
        pooling=pooling,
        edge_dropout=edge_dropout,
        use_rgcn=use_rgcn,
        use_dual_channel=use_dual_channel,
        use_gated_fusion=use_gated_fusion,
        use_node_attention=use_node_attention,
    )


# ============================================================================
# Aliases for convenience
# ============================================================================

HCASGNN = HierarchicalCrossAttentionSGNN
create_function_level_hca_model = create_hca_sgnn_model

__all__ = [
    'HierarchicalCrossAttentionSGNN',
    'HCASGNN',
    'create_hca_sgnn_model',
    'create_function_level_hca_model',
    'EdgeTypeAwareConv',
    'FunctionContractCrossAttention',
    'HierarchicalFunctionPooling',
    'GatedLocalGlobalFusion',
    'NodeLevelAttention',
]
