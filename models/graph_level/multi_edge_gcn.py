import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, RGCNConv, GATConv
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool
from typing import Optional


class MultiEdgeGCNEncoder(nn.Module):
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.3,
        num_edge_types: int = 2,
        mode: str = 'rgcn',  # 'gcn', 'rgcn', 'gat'
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.num_edge_types = num_edge_types
        self.mode = mode
        
        # Input projection
        self.input_linear = nn.Linear(input_dim, hidden_dim)
        
        # Build layers based on mode
        self.convs = nn.ModuleList()
        
        for _ in range(num_layers):
            if mode == 'gcn':
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            elif mode == 'rgcn':
                self.convs.append(RGCNConv(
                    hidden_dim, hidden_dim, 
                    num_relations=num_edge_types,
                    num_bases=min(num_edge_types, 4)  # Basis decomposition
                ))
            elif mode == 'gat':
                self.convs.append(GATConv(
                    hidden_dim, hidden_dim // 4,
                    heads=4, dropout=dropout, concat=True
                ))
            else:
                raise ValueError(f"Unknown mode: {mode}")
        
        # Layer normalization for stability
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.norms.append(nn.LayerNorm(hidden_dim))
    
    def forward(self, x, edge_index, edge_type=None):
        # Project input to hidden dimension
        x = self.input_linear(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Apply conv layers with residual connections
        for conv, norm in zip(self.convs, self.norms):
            x_res = x
            
            if self.mode == 'rgcn' and edge_type is not None:
                x = conv(x, edge_index, edge_type)
            else:
                x = conv(x, edge_index)
            
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x + x_res  # Residual connection
        
        return x


class NodeLevelMultiEdgeGCN(nn.Module):
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.3,
        num_classes: int = 2,
        num_edge_types: int = 2,
        mode: str = 'rgcn',
    ):
        super().__init__()
        
        self.encoder = MultiEdgeGCNEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            num_edge_types=num_edge_types,
            mode=mode,
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )
    
    def forward(self, x, edge_index, edge_type=None, batch=None):
        node_emb = self.encoder(x, edge_index, edge_type)
        logits = self.classifier(node_emb)
        return logits
    
    def predict(self, x, edge_index, edge_type=None, batch=None):
        logits = self.forward(x, edge_index, edge_type, batch)
        probs = F.softmax(logits, dim=-1)
        preds = logits.argmax(dim=-1)
        return preds, probs


class GraphLevelMultiEdgeGCN(nn.Module):
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.3,
        num_classes: int = 8,
        num_edge_types: int = 2,
        mode: str = 'rgcn',
        pooling: str = 'mean',
    ):
        super().__init__()
        
        self.encoder = MultiEdgeGCNEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            num_edge_types=num_edge_types,
            mode=mode,
        )
        
        # Pooling
        self.pooling = pooling
        if pooling == 'mean':
            self.pool = global_mean_pool
        elif pooling == 'add':
            self.pool = global_add_pool
        elif pooling == 'max':
            self.pool = global_max_pool
        else:
            raise ValueError(f"Unknown pooling: {pooling}")
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )
    
    def forward(self, x, edge_index, edge_type=None, batch=None):
        # Encode nodes
        node_emb = self.encoder(x, edge_index, edge_type)
        
        # Pool to graph level
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        graph_emb = self.pool(node_emb, batch)
        
        # Classify
        logits = self.classifier(graph_emb)
        return logits
    
    def predict(self, x, edge_index, edge_type=None, batch=None):
        logits = self.forward(x, edge_index, edge_type, batch)
        probs = F.softmax(logits, dim=-1)
        preds = logits.argmax(dim=-1)
        return preds, probs


class HierarchicalMultiEdgeGCN(nn.Module):
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.3,
        num_classes: int = 8,
        num_edge_types: int = 2,
        mode: str = 'rgcn',
    ):
        super().__init__()
        
        self.encoder = MultiEdgeGCNEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            num_edge_types=num_edge_types,
            mode=mode,
        )
        
        # Attention for pooling
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )
    
    def forward(self, x, edge_index, edge_type=None, batch=None):
        # Encode nodes
        node_emb = self.encoder(x, edge_index, edge_type)
        
        # Compute attention weights
        attn_scores = self.attention(node_emb).squeeze(-1)  # [num_nodes]
        
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # Softmax attention within each graph
        attn_weights = self._softmax_per_graph(attn_scores, batch)
        
        # Weighted sum pooling
        graph_emb = self._weighted_pool(node_emb, attn_weights, batch)
        
        # Classify
        logits = self.classifier(graph_emb)
        return logits
    
    def _softmax_per_graph(self, scores, batch):
        # Subtract max for numerical stability
        max_scores = torch.zeros(batch.max() + 1, device=scores.device)
        max_scores.scatter_reduce_(0, batch, scores, reduce='amax', include_self=False)
        scores = scores - max_scores[batch]
        
        # Exp and normalize
        exp_scores = torch.exp(scores)
        sum_exp = torch.zeros(batch.max() + 1, device=scores.device)
        sum_exp.scatter_add_(0, batch, exp_scores)
        
        return exp_scores / (sum_exp[batch] + 1e-8)
    
    def _weighted_pool(self, node_emb, weights, batch):
        # Weight embeddings
        weighted = node_emb * weights.unsqueeze(-1)
        
        # Sum within each graph
        num_graphs = batch.max() + 1
        graph_emb = torch.zeros(num_graphs, node_emb.size(1), device=node_emb.device)
        graph_emb.scatter_add_(0, batch.unsqueeze(-1).expand_as(weighted), weighted)
        
        return graph_emb
    
    def predict(self, x, edge_index, edge_type=None, batch=None):
        logits = self.forward(x, edge_index, edge_type, batch)
        probs = F.softmax(logits, dim=-1)
        preds = logits.argmax(dim=-1)
        return preds, probs
    
    def get_attention_weights(self, x, edge_index, edge_type=None, batch=None):
        with torch.no_grad():
            node_emb = self.encoder(x, edge_index, edge_type)
            attn_scores = self.attention(node_emb).squeeze(-1)
            if batch is None:
                batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            attn_weights = self._softmax_per_graph(attn_scores, batch)
        return attn_weights


# Factory function
def create_model(
    model_type: str,
    input_dim: int,
    hidden_dim: int = 64,
    num_layers: int = 3,
    dropout: float = 0.3,
    num_classes: int = 8,
    num_edge_types: int = 2,
    mode: str = 'rgcn',
    pooling: str = 'mean',
):
    if model_type == 'node':
        return NodeLevelMultiEdgeGCN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            num_classes=num_classes,
            num_edge_types=num_edge_types,
            mode=mode,
        )
    elif model_type == 'graph':
        return GraphLevelMultiEdgeGCN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            num_classes=num_classes,
            num_edge_types=num_edge_types,
            mode=mode,
            pooling=pooling,
        )
    elif model_type == 'hierarchical':
        return HierarchicalMultiEdgeGCN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            num_classes=num_classes,
            num_edge_types=num_edge_types,
            mode=mode,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
