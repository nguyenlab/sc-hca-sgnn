import json
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader


# ============================================================================
# Constants
# ============================================================================

# Vulnerability types
VULN_TYPES = [
    'clean',
    'overflow', 
    'underflow',
    'reentrancy',
    'timestamp',
    'tx_origin', 
    'unchecked_send',
    'unhandled_exception',
    'cross-function',  # Synthetic data
    # Real-world SmartBugs vulnerability types
    'access_control',
    'bad_randomness',
    'dos',
    'front_running',
    'other',
    'short_addresses',
]

VULN_TO_IDX = {v: i for i, v in enumerate(VULN_TYPES)}
IDX_TO_VULN = {i: v for i, v in enumerate(VULN_TYPES)}
NUM_VULN_CLASSES = len(VULN_TYPES)

# AST Node types (Solidity)
# Comprehensive list covering both modern and legacy AST formats
NODE_TYPES = [
    # Special
    'Unknown',
    
    # Top-level
    'SourceUnit',
    'PragmaDirective', 
    'ImportDirective',
    
    # Contract-level definitions
    'ContractDefinition',
    'InterfaceDefinition',
    'LibraryDefinition',
    'InheritanceSpecifier',
    'UsingForDirective',
    'StructDefinition',
    'EnumDefinition',
    'EnumValue',
    
    # State and function declarations
    'VariableDeclaration',
    'FunctionDefinition',
    'ModifierDefinition',
    'ModifierInvocation',
    'EventDefinition',
    'ErrorDefinition',
    'ParameterList',
    'OverrideSpecifier',
    
    # Statements
    'Block',
    'PlaceholderStatement',
    'IfStatement',
    'WhileStatement',
    'ForStatement',
    'DoWhileStatement',
    'Continue',
    'Break',
    'Return',
    'Throw',
    'EmitStatement',
    'RevertStatement',
    'TryStatement',
    'TryCatchClause',
    'VariableDeclarationStatement',
    'ExpressionStatement',
    'UncheckedBlock',
    
    # Expressions
    'Assignment',
    'TupleExpression',
    'UnaryOperation',
    'BinaryOperation',
    'FunctionCall',
    'FunctionCallOptions',
    'NewExpression',
    'MemberAccess',
    'IndexAccess',
    'IndexRangeAccess',
    'Identifier',
    'ElementaryTypeNameExpression',
    'Literal',
    'Conditional',
    
    # Type names
    'ElementaryTypeName',
    'UserDefinedTypeName',
    'FunctionTypeName',
    'Mapping',
    'ArrayTypeName',
    
    # Assembly
    'InlineAssembly',
    'YulBlock',
    'YulVariableDeclaration',
    'YulAssignment',
    'YulFunctionCall',
    'YulIdentifier',
    'YulLiteral',
    'YulExpressionStatement',
    'YulIf',
    'YulSwitch',
    'YulCase',
    'YulForLoop',
    'YulFunctionDefinition',
    'YulTypedName',
    
    # User-defined types (Solidity 0.8+)
    'UserDefinedValueTypeDefinition',
    
    # Elementary types (from legacy AST format where types are nodes)
    'uint',
    'uint8',
    'uint16',
    'uint32',
    'uint64',
    'uint128',
    'uint256',
    'int',
    'int8',
    'int16',
    'int32',
    'int64',
    'int128',
    'int256',
    'bool',
    'address',
    'bytes',
    'bytes1',
    'bytes2',
    'bytes4',
    'bytes8',
    'bytes16',
    'bytes32',
    'string',
    'fixed',
    'ufixed',
]

NODE_TO_IDX = {n: i for i, n in enumerate(NODE_TYPES)}
NUM_NODE_TYPES = len(NODE_TYPES)
IDX_TO_NODE = {i: n for i, n in enumerate(NODE_TYPES)}


def get_node_type_idx(node_type: str) -> int:
    return NODE_TO_IDX.get(node_type, 0)

# Edge types
EDGE_TYPE_AST = 0       # Parent-child (structural)
EDGE_TYPE_REF = 1       # Reference/data flow (Identifier -> Declaration)
EDGE_TYPE_CFG_NEXT = 2  # Control flow: next statement
EDGE_TYPE_CFG_TRUE = 3  # Control flow: true branch
EDGE_TYPE_CFG_FALSE = 4 # Control flow: false branch
EDGE_TYPE_CALL = 5      # Function call (FunctionCall -> FunctionDefinition)
EDGE_TYPE_INHERIT = 6   # Inheritance (Contract -> BaseContract)
EDGE_TYPE_GUARD = 7     # Guard condition (require/assert -> guarded statements)

NUM_EDGE_TYPES = 8  # Total number of edge types

EDGE_TYPE_NAMES = ['ast', 'ref', 'cfg_next', 'cfg_true', 'cfg_false', 'call', 'inherit', 'guard']


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class GraphStats:
    num_graphs: int
    num_nodes: int
    num_edges: int
    num_ast_edges: int
    num_ref_edges: int
    num_vulnerable_nodes: int
    vuln_distribution: Dict[str, int]
    avg_nodes_per_graph: float
    avg_edges_per_graph: float


# ============================================================================
# Feature Extraction
# ============================================================================

def get_node_features(nodes: List[Dict], normalize_pos: bool = True) -> torch.Tensor:
    num_nodes = len(nodes)
    
    # Node type one-hot encoding
    node_type_features = torch.zeros(num_nodes, NUM_NODE_TYPES)
    for i, node in enumerate(nodes):
        # Use pre-computed index if available, otherwise look up by type name
        if 'node_type_idx' in node:
            idx = node['node_type_idx']
        else:
            node_type = node.get('node_type', 'Unknown')
            idx = NODE_TO_IDX.get(node_type, 0)  # Default to Unknown
        node_type_features[i, idx] = 1.0
    
    # Byte position features
    if normalize_pos and nodes:
        max_byte = max(n.get('end_byte', 1) for n in nodes)
        max_byte = max(max_byte, 1)  # Avoid division by zero
    else:
        max_byte = 1
    
    pos_features = torch.zeros(num_nodes, 3)
    for i, node in enumerate(nodes):
        start = node.get('start_byte', 0)
        end = node.get('end_byte', 0)
        pos_features[i, 0] = start / max_byte
        pos_features[i, 1] = end / max_byte
        pos_features[i, 2] = (end - start) / max_byte
    
    # Concatenate features
    features = torch.cat([node_type_features, pos_features], dim=1)
    
    return features


def get_node_features_from_indices(
    node_type_idx: List[int],
    start_bytes: Optional[List[int]] = None,
    end_bytes: Optional[List[int]] = None,
    normalize_pos: bool = True
) -> torch.Tensor:
    
    num_nodes = len(node_type_idx)
    
    # Node type one-hot encoding from indices (efficient)
    node_type_features = torch.zeros(num_nodes, NUM_NODE_TYPES)
    for i, idx in enumerate(node_type_idx):
        if 0 <= idx < NUM_NODE_TYPES:
            node_type_features[i, idx] = 1.0
        else:
            node_type_features[i, 0] = 1.0  # Unknown
    
    # If no position data provided, return just node type features
    if start_bytes is None or end_bytes is None:
        return node_type_features
    
    # Byte position features
    if normalize_pos and end_bytes:
        max_byte = max(max(end_bytes), 1)
    else:
        max_byte = 1
    
    pos_features = torch.zeros(num_nodes, 3)
    for i in range(num_nodes):
        start = start_bytes[i] if i < len(start_bytes) else 0
        end = end_bytes[i] if i < len(end_bytes) else 0
        pos_features[i, 0] = start / max_byte
        pos_features[i, 1] = end / max_byte
        pos_features[i, 2] = (end - start) / max_byte
    
    return torch.cat([node_type_features, pos_features], dim=1)


def get_edge_index_and_attr(
    edges: List, 
    use_edge_attr: bool = True,
    bidirectional: bool = True
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    
    if not edges:
        if use_edge_attr:
            return torch.zeros((2, 0), dtype=torch.long), torch.zeros((0,), dtype=torch.long)
        return torch.zeros((2, 0), dtype=torch.long), None
    
    # Check edge format
    sample_edge = edges[0]
    has_type = len(sample_edge) >= 3
    
    edge_list = []
    edge_types = []
    
    for edge in edges:
        src = edge[0]
        tgt = edge[1]
        etype = edge[2] if has_type else EDGE_TYPE_AST
        
        edge_list.append([src, tgt])
        edge_types.append(etype)
        
        if bidirectional:
            edge_list.append([tgt, src])
            edge_types.append(etype)
    
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    
    if use_edge_attr:
        edge_attr = torch.tensor(edge_types, dtype=torch.long)
        return edge_index, edge_attr
    
    return edge_index, None


def get_edge_index(edges: List[List[int]], bidirectional: bool = True) -> torch.Tensor:
    edge_index, _ = get_edge_index_and_attr(edges, use_edge_attr=False, bidirectional=bidirectional)
    return edge_index


def get_node_labels(nodes: List[Dict]) -> torch.Tensor:
    labels = torch.tensor(
        [1 if n.get('is_vulnerable', False) else 0 for n in nodes],
        dtype=torch.long
    )
    return labels


def get_graph_label(vuln_type: str) -> torch.Tensor:
    # Binary classification: 0 = clean, 1 = any vulnerability
    if vuln_type == 'clean':
        return torch.tensor([0], dtype=torch.long)
    else:
        return torch.tensor([1], dtype=torch.long)


# ============================================================================
# Data Loading
# ============================================================================

def load_graph_json(path: Path) -> Dict[str, Any]:
    with open(path, 'r') as f:
        return json.load(f)


def json_to_pyg(
    data: Dict[str, Any], 
    use_edge_attr: bool = True,
    bidirectional: bool = True
) -> Data:
    edges = data['edges']
    vuln_type = data.get('vulnerability_type', 'clean')
    
    # Handle both old and new formats
    if 'nodes' in data:
        # Old format: full node objects
        nodes = data['nodes']
        x = get_node_features(nodes)
        y = get_node_labels(nodes)
        num_nodes = len(nodes)
    elif 'node_type_idx' in data:
        # New format with pre-computed indices (most efficient)
        node_type_idx = data['node_type_idx']
        node_is_vuln = data.get('node_is_vulnerable', [False] * len(node_type_idx))
        num_nodes = data.get('num_nodes', len(node_type_idx))
        
        # Get position data if available
        start_bytes = data.get('node_start_bytes')
        end_bytes = data.get('node_end_bytes')
        
        # Use pre-computed indices directly with position features
        x = get_node_features_from_indices(node_type_idx, start_bytes, end_bytes)
        y = torch.tensor(node_is_vuln, dtype=torch.long)
    elif 'node_types' in data:
        # New format: separate arrays (with type names)
        node_types = data['node_types']
        node_is_vuln = data.get('node_is_vulnerable', [False] * len(node_types))
        num_nodes = data.get('num_nodes', len(node_types))
        
        # Build node features from type names
        x = get_node_features_from_types(node_types)
        y = torch.tensor(node_is_vuln, dtype=torch.long)
    else:
        raise ValueError("Data must contain either 'nodes', 'node_types', or 'node_type_idx'")
    
    edge_index, edge_attr = get_edge_index_and_attr(edges, use_edge_attr, bidirectional)
    graph_y = get_graph_label(vuln_type)
    
    # Create Data object
    pyg_data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
        graph_y=graph_y,
        num_nodes=num_nodes,
    )
    
    if edge_attr is not None:
        pyg_data.edge_attr = edge_attr
    
    # Metadata
    pyg_data.vuln_type = vuln_type
    pyg_data.contract_path = data.get('contract_path', '')
    pyg_data.num_vulnerable = int(y.sum().item()) if y.dim() > 0 else 0
    
    # Edge counts from new format
    edge_counts = data.get('edge_counts', {})
    pyg_data.num_ast_edges = edge_counts.get('ast', data.get('num_ast_edges', len(edges)))
    pyg_data.num_ref_edges = edge_counts.get('ref', data.get('num_ref_edges', 0))
    
    return pyg_data


def get_node_features_from_types(node_types: List[str]) -> torch.Tensor:
    num_nodes = len(node_types)
    features = torch.zeros(num_nodes, NUM_NODE_TYPES)
    
    for i, ntype in enumerate(node_types):
        type_idx = NODE_TO_IDX.get(ntype, 0)  # Default to 'Unknown'
        features[i, type_idx] = 1.0
    
    return features


# ============================================================================
# Dataset Class
# ============================================================================

class ContractGraphDataset(Dataset):
    
    def __init__(
        self,
        root: str,
        graph_dir: str = None,  # Auto-detect if None
        transform=None,
        pre_transform=None,
        vuln_filter: Optional[List[str]] = None,
        use_edge_attr: bool = True,
        bidirectional: bool = True,
    ):
        self.vuln_filter = vuln_filter
        self.use_edge_attr = use_edge_attr
        self.bidirectional = bidirectional
        self._files: List[Path] = []
        
        # Auto-detect graph directory
        if graph_dir is None:
            root_path = Path(root)
            if (root_path / 'graph_data').exists():
                self.graph_dir = 'graph_data'
            elif (root_path / 'ast_graphs').exists():
                self.graph_dir = 'ast_graphs'
            else:
                raise FileNotFoundError(f"No graph directory found in {root}. Expected 'graph_data' or 'ast_graphs'.")
        else:
            self.graph_dir = graph_dir
        
        super().__init__(root, transform, pre_transform)
        
        # Discover graph files
        self._discover_files()
    
    def _discover_files(self):
        graph_path = Path(self.root) / self.graph_dir
        
        if not graph_path.exists():
            raise FileNotFoundError(f"Graph directory not found: {graph_path}")
        
        self._files = sorted(graph_path.glob('*.json'))
        
        # Filter by vulnerability type if specified
        if self.vuln_filter:
            filtered = []
            for f in self._files:
                data = load_graph_json(f)
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
    
    def get(self, idx: int) -> Data:
        data = load_graph_json(self._files[idx])
        return json_to_pyg(data, self.use_edge_attr, self.bidirectional)
    
    def get_stats(self) -> GraphStats:
        vuln_dist = {v: 0 for v in VULN_TYPES}
        total_nodes = 0
        total_edges = 0
        total_ast_edges = 0
        total_ref_edges = 0
        total_vuln_nodes = 0
        
        for f in self._files:
            data = load_graph_json(f)
            edges = data['edges']
            vuln_type = data.get('vulnerability_type', 'clean')
            
            # Handle both old and new formats
            if 'nodes' in data:
                num_nodes = len(data['nodes'])
                num_vuln = sum(1 for n in data['nodes'] if n.get('is_vulnerable', False))
            elif 'num_nodes' in data:
                num_nodes = data['num_nodes']
                if 'node_is_vulnerable' in data:
                    num_vuln = sum(1 for v in data['node_is_vulnerable'] if v)
                else:
                    num_vuln = len(data.get('vulnerable_node_ids', []))
            else:
                num_nodes = len(data.get('node_types', []))
                num_vuln = len(data.get('vulnerable_node_ids', []))
            
            # Handle edge counts from new format
            edge_counts = data.get('edge_counts', {})
            ast_edges = edge_counts.get('ast', data.get('num_ast_edges', len(edges)))
            ref_edges = edge_counts.get('ref', data.get('num_ref_edges', 0))
            
            vuln_dist[vuln_type] = vuln_dist.get(vuln_type, 0) + 1
            total_nodes += num_nodes
            total_edges += len(edges)
            total_ast_edges += ast_edges
            total_ref_edges += ref_edges
            total_vuln_nodes += num_vuln
        
        num_graphs = len(self._files)
        
        return GraphStats(
            num_graphs=num_graphs,
            num_nodes=total_nodes,
            num_edges=total_edges,
            num_ast_edges=total_ast_edges,
            num_ref_edges=total_ref_edges,
            num_vulnerable_nodes=total_vuln_nodes,
            vuln_distribution=vuln_dist,
            avg_nodes_per_graph=total_nodes / num_graphs if num_graphs > 0 else 0,
            avg_edges_per_graph=total_edges / num_graphs if num_graphs > 0 else 0,
        )


# ============================================================================
# Data Utilities
# ============================================================================

def create_data_loaders(
    dataset: ContractGraphDataset,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    batch_size: int = 32,
    shuffle: bool = True,
    seed: int = 42,
    stratified: bool = True,  # NEW: Enable stratified splitting by default
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    
    torch.manual_seed(seed)
    
    num_graphs = len(dataset)
    
    if stratified:
        # Stratified split: maintain class balance in each split
        # Group indices by class (binary: clean=0, vulnerable=1+)
        clean_indices = []
        vuln_indices = []
        
        for i in range(num_graphs):
            label = dataset.get(i).graph_y.item()
            # Binary conversion: 0 = clean, 1+ = vulnerable
            is_clean = (label == 0)
            if is_clean:
                clean_indices.append(i)
            else:
                vuln_indices.append(i)
        
        # Shuffle each class separately
        import random
        random.seed(seed)
        random.shuffle(clean_indices)
        random.shuffle(vuln_indices)
        
        # Split each class proportionally
        def split_class_indices(indices, train_r, val_r):
            n = len(indices)
            train_end = int(train_r * n)
            val_end = int((train_r + val_r) * n)
            return indices[:train_end], indices[train_end:val_end], indices[val_end:]
        
        clean_train, clean_val, clean_test = split_class_indices(clean_indices, train_ratio, val_ratio)
        vuln_train, vuln_val, vuln_test = split_class_indices(vuln_indices, train_ratio, val_ratio)
        
        # Combine and shuffle
        train_indices = clean_train + vuln_train
        val_indices = clean_val + vuln_val
        test_indices = clean_test + vuln_test
        
        random.shuffle(train_indices)
        random.shuffle(val_indices)
        random.shuffle(test_indices)
    else:
        # Original random split (legacy mode)
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


def get_input_dim() -> int:
    return NUM_NODE_TYPES + 3  # One-hot node type + 3 position features


def get_num_classes() -> int:
    return NUM_VULN_CLASSES


def get_num_edge_types() -> int:
    return NUM_EDGE_TYPES


def create_data_loaders_with_irl_test(
    synthetic_dataset: ContractGraphDataset,
    realworld_dataset: ContractGraphDataset,
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

