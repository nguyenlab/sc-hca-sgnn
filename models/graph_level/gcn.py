import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_add_pool


class GCNEncoder(nn.Module):
    
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
        
        # GCN layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        # Layer normalization for stability
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.norms.append(nn.LayerNorm(hidden_dim))
    
    def forward(self, x, edge_index):
        # Project input to hidden dimension
        x = self.input_linear(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Apply GCN layers with residual connections
        for conv, norm in zip(self.convs, self.norms):
            x_res = x
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x + x_res  # Residual connection
        
        return x


class NodeLevelGCN(nn.Module):
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.3,
        num_classes: int = 2,
    ):
        super().__init__()
        
        self.encoder = GCNEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )
    
    def forward(self, x, edge_index, batch=None):
        node_emb = self.encoder(x, edge_index)
        logits = self.classifier(node_emb)
        return logits
    
    def predict(self, x, edge_index, batch=None):
        """Get predictions and probabilities."""
        logits = self.forward(x, edge_index, batch)
        probs = F.softmax(logits, dim=-1)
        preds = logits.argmax(dim=-1)
        return preds, probs


class GraphLevelGCN(nn.Module):
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.3,
        num_classes: int = 8,
        pooling: str = 'mean',
    ):
        super().__init__()
        
        self.encoder = GCNEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        
        # Pooling function
        self.pooling = pooling
        if pooling == 'mean':
            self.pool = global_mean_pool
        elif pooling == 'add':
            self.pool = global_add_pool
        else:
            self.pool = global_mean_pool
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
    
    def forward(self, x, edge_index, batch):
        # Get node embeddings
        node_emb = self.encoder(x, edge_index)
        
        # Pool to graph-level representation
        graph_emb = self.pool(node_emb, batch)
        
        # Classify
        logits = self.classifier(graph_emb)
        return logits
    
    def predict(self, x, edge_index, batch):
        """Get predictions and probabilities."""
        logits = self.forward(x, edge_index, batch)
        probs = F.softmax(logits, dim=-1)
        preds = logits.argmax(dim=-1)
        return preds, probs


class CombinedGCN(nn.Module):
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.3,
        num_node_classes: int = 2,
        num_graph_classes: int = 8,
        pooling: str = 'mean',
    ):
        super().__init__()
        
        # Shared encoder
        self.encoder = GCNEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        
        # Pooling
        self.pool = global_mean_pool if pooling == 'mean' else global_add_pool
        
        # Node classifier head
        self.node_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_node_classes),
        )
        
        # Graph classifier head
        self.graph_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_graph_classes),
        )
    
    def forward(self, x, edge_index, batch):
        # Shared encoding
        node_emb = self.encoder(x, edge_index)
        
        # Node-level classification
        node_logits = self.node_classifier(node_emb)
        
        # Graph-level classification
        graph_emb = self.pool(node_emb, batch)
        graph_logits = self.graph_classifier(graph_emb)
        
        return node_logits, graph_logits
