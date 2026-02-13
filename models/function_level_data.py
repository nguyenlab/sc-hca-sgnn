import json
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import scatter

from models.data import (
    NUM_NODE_TYPES,
    NUM_EDGE_TYPES,
    VULN_TYPES,
    VULN_TO_IDX,
    get_node_features_from_indices,
)


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class FunctionLevelStats:
    """Statistics about a function-level dataset."""
    num_contracts: int
    num_functions: int
    num_vulnerable_functions: int
    num_clean_functions: int
    num_nodes: int
    num_edges: int
    num_intra_edges: int
    num_inter_edges: int
    avg_functions_per_contract: float
    avg_nodes_per_function: float
    vuln_distribution: Dict[str, int]


# ============================================================================
# Feature Extraction
# ============================================================================

def get_function_edge_index(
    edges: List[Tuple[int, int, int]],
    bidirectional: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    
    if not edges:
        return torch.zeros((2, 0), dtype=torch.long), torch.zeros((0,), dtype=torch.long)
    
    edge_list = []
    edge_types = []
    
    for src, dst, etype in edges:
        edge_list.append([src, dst])
        edge_types.append(etype)
        
        if bidirectional:
            edge_list.append([dst, src])
            edge_types.append(etype)
    
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_types, dtype=torch.long)
    
    return edge_index, edge_attr


# ============================================================================
# PyG Data Conversion
# ============================================================================

class FunctionLevelData(Data):
    
    def __inc__(self, key, value, *args, **kwargs):
        """Handle batching of multiple graphs."""
        if key == 'sub_edge_index':
            return self.num_nodes
        if key == 'inter_edge_index':
            return self.num_nodes
        if key == 'subg':
            # Don't increment subg - it's a node attribute, not an edge index
            return 0
        return super().__inc__(key, value, *args, **kwargs)
    
    def __cat_dim__(self, key, value, *args, **kwargs):
        if key in ['y_func', 'function_batch']:
            return 0  # Concatenate along first dimension
        return super().__cat_dim__(key, value, *args, **kwargs)


def json_to_function_level_data(
    data: Dict[str, Any],
    bidirectional: bool = True,
) -> FunctionLevelData:
    
    # Node features
    node_type_idx = data['node_type_idx']
    start_bytes = data.get('node_start_bytes')
    end_bytes = data.get('node_end_bytes')
    
    x = get_node_features_from_indices(node_type_idx, start_bytes, end_bytes)
    
    # Full edge index (all edges)
    all_edges = [tuple(e) for e in data['edges']]
    edge_index, edge_attr = get_function_edge_index(all_edges, bidirectional)
    
    # Intra-function edges (within same function)
    intra_edges = [tuple(e) for e in data['intra_function_edges']]
    sub_edge_index, sub_edge_attr = get_function_edge_index(intra_edges, bidirectional)
    
    # Inter-function edges (crossing function boundaries)
    inter_edges = [tuple(e) for e in data['inter_function_edges']]
    inter_edge_index, inter_edge_attr = get_function_edge_index(inter_edges, bidirectional)
    
    # Node-to-function assignment
    subg = torch.tensor(data['node_function_id'], dtype=torch.long)
    
    # Per-function labels
    y_func = torch.tensor(data['function_labels'], dtype=torch.long)
    
    # Graph-level vulnerability label (1 if any function is vulnerable)
    vuln_type = data.get('vulnerability_type', 'clean')
    graph_y = torch.tensor([1 if vuln_type != 'clean' else 0], dtype=torch.long)
    
    # Node-level labels
    node_is_vuln = data.get('node_is_vulnerable', [False] * len(node_type_idx))
    y = torch.tensor(node_is_vuln, dtype=torch.long)
    
    # Create FunctionLevelData
    pyg_data = FunctionLevelData(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
        graph_y=graph_y,
        num_nodes=len(node_type_idx),
    )
    
    # Add function-level attributes
    pyg_data.subg = subg
    pyg_data.sub_edge_index = sub_edge_index
    pyg_data.sub_edge_attr = sub_edge_attr
    pyg_data.inter_edge_index = inter_edge_index
    pyg_data.inter_edge_attr = inter_edge_attr
    pyg_data.y_func = y_func
    pyg_data.num_functions = len(data['functions'])
    
    # Metadata
    pyg_data.vuln_type = vuln_type
    pyg_data.contract_path = data.get('contract_path', '')
    pyg_data.function_names = [f['name'] for f in data['functions']]
    
    return pyg_data


# ============================================================================
# Dataset Class
# ============================================================================

class FunctionLevelDataset(Dataset):
    
    def __init__(
        self,
        root: str,
        graph_dir: str = 'function_graphs',
        transform=None,
        pre_transform=None,
        vuln_filter: Optional[List[str]] = None,
        bidirectional: bool = True,
    ):
        self.graph_dir = graph_dir
        self.vuln_filter = vuln_filter
        self.bidirectional = bidirectional
        self._files: List[Path] = []
        
        super().__init__(root, transform, pre_transform)
        self._discover_files()
    
    def _discover_files(self):
        """Find all function-level graph JSON files."""
        graph_path = Path(self.root) / self.graph_dir
        
        if not graph_path.exists():
            raise FileNotFoundError(f"Function graph directory not found: {graph_path}")
        
        self._files = sorted(graph_path.glob('*.json'))
        
        # Filter by vulnerability type if specified
        if self.vuln_filter:
            filtered = []
            for f in self._files:
                with open(f, 'r') as fp:
                    data = json.load(fp)
                if data.get('vulnerability_type') in self.vuln_filter:
                    filtered.append(f)
            self._files = filtered
    
    @property
    def raw_file_names(self):
        return []
    
    @property
    def processed_file_names(self):
        return []
    
    def download(self):
        pass
    
    def process(self):
        pass
    
    def len(self) -> int:
        return len(self._files)
    
    def get(self, idx: int) -> FunctionLevelData:
        with open(self._files[idx], 'r') as f:
            data = json.load(f)
        return json_to_function_level_data(data, self.bidirectional)
    
    def get_stats(self) -> FunctionLevelStats:
        vuln_dist = {v: 0 for v in VULN_TYPES}
        total_functions = 0
        total_vuln_functions = 0
        total_clean_functions = 0
        total_nodes = 0
        total_edges = 0
        total_intra_edges = 0
        total_inter_edges = 0
        
        for f in self._files:
            with open(f, 'r') as fp:
                data = json.load(fp)
            
            vuln_type = data.get('vulnerability_type', 'clean')
            vuln_dist[vuln_type] = vuln_dist.get(vuln_type, 0) + 1
            
            num_funcs = len(data.get('functions', []))
            num_vuln_funcs = sum(data.get('function_labels', []))
            
            total_functions += num_funcs
            total_vuln_functions += num_vuln_funcs
            total_clean_functions += num_funcs - num_vuln_funcs
            total_nodes += data.get('num_nodes', 0)
            total_edges += len(data.get('edges', []))
            total_intra_edges += len(data.get('intra_function_edges', []))
            total_inter_edges += len(data.get('inter_function_edges', []))
        
        num_contracts = len(self._files)
        
        return FunctionLevelStats(
            num_contracts=num_contracts,
            num_functions=total_functions,
            num_vulnerable_functions=total_vuln_functions,
            num_clean_functions=total_clean_functions,
            num_nodes=total_nodes,
            num_edges=total_edges,
            num_intra_edges=total_intra_edges,
            num_inter_edges=total_inter_edges,
            avg_functions_per_contract=total_functions / num_contracts if num_contracts > 0 else 0,
            avg_nodes_per_function=total_nodes / total_functions if total_functions > 0 else 0,
            vuln_distribution=vuln_dist,
        )


# ============================================================================
# Data Loaders
# ============================================================================

def create_function_level_loaders(
    dataset: FunctionLevelDataset,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    batch_size: int = 32,
    shuffle: bool = True,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    
    torch.manual_seed(seed)
    
    num_graphs = len(dataset)
    indices = torch.randperm(num_graphs).tolist()
    
    train_end = int(train_ratio * num_graphs)
    val_end = int((train_ratio + val_ratio) * num_graphs)
    
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]
    
    train_dataset = [dataset[i] for i in train_indices]
    val_dataset = [dataset[i] for i in val_indices]
    test_dataset = [dataset[i] for i in test_indices]
    
    # Create generator for deterministic shuffling
    g = torch.Generator()
    g.manual_seed(seed)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, generator=g)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader


# ============================================================================
# Utility Functions
# ============================================================================

def pool_by_function(
    x: torch.Tensor,
    subg: torch.Tensor,
    num_functions: int,
    method: str = 'mean',
) -> torch.Tensor:
    
    # Only pool nodes that belong to a function (subg > 0)
    # Shift subg by -1 so function indices are 0-based
    func_mask = subg > 0
    func_nodes = x[func_mask]
    func_assignment = subg[func_mask] - 1  # 0-indexed
    
    if func_nodes.size(0) == 0:
        return torch.zeros(num_functions, x.size(1), device=x.device)
    
    if method == 'mean':
        return scatter(func_nodes, func_assignment, dim=0, dim_size=num_functions, reduce='mean')
    elif method == 'max':
        return scatter(func_nodes, func_assignment, dim=0, dim_size=num_functions, reduce='max')
    elif method == 'sum':
        return scatter(func_nodes, func_assignment, dim=0, dim_size=num_functions, reduce='sum')
    else:
        raise ValueError(f"Unknown pooling method: {method}")


def get_function_batch(
    subg_batch: torch.Tensor,
    batch: torch.Tensor,
    num_functions_per_graph: List[int],
) -> torch.Tensor:
    
    function_batch = []
    for graph_idx, num_funcs in enumerate(num_functions_per_graph):
        function_batch.extend([graph_idx] * num_funcs)
    return torch.tensor(function_batch, dtype=torch.long)


def create_function_level_loaders_with_irl_test(
    synthetic_dataset: FunctionLevelDataset,
    realworld_dataset: FunctionLevelDataset,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    batch_size: int = 32,
    shuffle: bool = True,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    
    torch.manual_seed(seed)
    
    # Split synthetic data into train/val/test
    num_synthetic = len(synthetic_dataset)
    indices = torch.randperm(num_synthetic).tolist()
    
    train_end = int(train_ratio * num_synthetic)
    val_end = int((train_ratio + val_ratio) * num_synthetic)
    
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    syn_test_indices = indices[val_end:]
    
    train_dataset = [synthetic_dataset[i] for i in train_indices]
    val_dataset = [synthetic_dataset[i] for i in val_indices]
    syn_test_dataset = [synthetic_dataset[i] for i in syn_test_indices]
    
    # Use all realworld data for IRL test
    irl_test_dataset = [realworld_dataset[i] for i in range(len(realworld_dataset))]
    
    # Create generator for deterministic shuffling
    g = torch.Generator()
    g.manual_seed(seed)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, generator=g)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    syn_test_loader = DataLoader(syn_test_dataset, batch_size=batch_size, shuffle=False)
    irl_test_loader = DataLoader(irl_test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, syn_test_loader, irl_test_loader
