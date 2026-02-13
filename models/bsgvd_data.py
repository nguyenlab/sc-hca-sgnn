import json
import torch
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader

from models.data import (
    VULN_TYPES, VULN_TO_IDX, IDX_TO_VULN,
    NODE_TYPES, NODE_TO_IDX, NUM_NODE_TYPES,
    NUM_EDGE_TYPES, EDGE_TYPE_AST,
    get_node_features_from_indices,
)

# Try to import FastText utilities
try:
    from models.graph_level.fasttext_embedding import (
        FastTextEmbedder,
        load_fasttext_model,
        GENSIM_AVAILABLE,
    )
except ImportError:
    GENSIM_AVAILABLE = False
    FastTextEmbedder = None
    load_fasttext_model = None


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class BSGVDDataStats:
    num_graphs: int
    num_nodes: int
    num_edges: int
    feature_dim: int
    num_clean: int
    num_vulnerable: int
    vuln_distribution: Dict[str, int]


# ============================================================================
# BSGVD Dataset
# ============================================================================

class BSGVDDataset(Dataset):
    
    def __init__(
        self,
        graph_dir: Union[str, Path],
        ast_data_dir: Optional[Union[str, Path]] = None,
        fasttext_model_path: Optional[Union[str, Path]] = None,
        fasttext_embedder: Optional['FastTextEmbedder'] = None,
        use_fasttext: bool = True,
        combine_with_type: bool = True,
        cache_features: bool = True,
        transform=None,
    ):
        super().__init__(transform=transform)
        
        self.graph_dir = Path(graph_dir)
        self.ast_data_dir = Path(ast_data_dir) if ast_data_dir else None
        self.use_fasttext = use_fasttext and GENSIM_AVAILABLE
        self.combine_with_type = combine_with_type
        self.cache_features = cache_features
        
        # Initialize FastText embedder
        self.embedder = None
        if self.use_fasttext:
            if fasttext_embedder is not None:
                self.embedder = fasttext_embedder
            elif fasttext_model_path is not None:
                try:
                    model = load_fasttext_model(fasttext_model_path)
                    self.embedder = FastTextEmbedder(
                        model=model,
                        combine_with_type=combine_with_type,
                        num_node_types=NUM_NODE_TYPES,
                    )
                except Exception as e:
                    warnings.warn(f"Failed to load FastText model: {e}. Using one-hot features.")
                    self.use_fasttext = False
        
        if not self.use_fasttext:
            warnings.warn("FastText not available. Using standard one-hot features.")
        
        # Get feature dimension
        if self.embedder is not None:
            self._feature_dim = self.embedder.output_dim
        else:
            self._feature_dim = NUM_NODE_TYPES + 3  # one-hot + position features
        
        # Load graph file list
        self.graph_files = sorted(list(self.graph_dir.glob('*.json')))
        if not self.graph_files:
            raise FileNotFoundError(f"No graph files found in {self.graph_dir}")
        
        # Feature cache
        self._feature_cache: Dict[str, torch.Tensor] = {} if cache_features else None
        
        # Lazy metadata loading - only load when needed (for stats)
        self._metadata: Optional[List[Dict]] = None
    
    def _load_metadata(self):
        if self._metadata is not None:
            return
        
        self._metadata = []
        for graph_file in self.graph_files:
            # Fast partial parse - just read what we need
            try:
                with open(graph_file, 'r') as f:
                    # Read first 2000 chars which should contain metadata
                    content = f.read(2000)
                    # Parse num_nodes
                    import re
                    num_nodes_match = re.search(r'"num_nodes":\s*(\d+)', content)
                    num_nodes = int(num_nodes_match.group(1)) if num_nodes_match else 0
                    # Parse vulnerability type
                    vuln_match = re.search(r'"vulnerability_type":\s*"([^"]+)"', content)
                    vuln_type = vuln_match.group(1) if vuln_match else 'clean'
                    
                self._metadata.append({
                    'filename': graph_file.name,
                    'num_nodes': num_nodes,
                    'vulnerability_type': vuln_type,
                })
            except Exception:
                self._metadata.append({
                    'filename': graph_file.name,
                    'num_nodes': 0,
                    'vulnerability_type': 'clean',
                })
    
    @property
    def feature_dim(self) -> int:
        return self._feature_dim
    
    def len(self) -> int:
        return len(self.graph_files)
    
    def get(self, idx: int) -> Data:
        graph_file = self.graph_files[idx]
        filename = graph_file.name
        
        # Load graph data
        with open(graph_file, 'r') as f:
            graph_data = json.load(f)
        
        # Get node features
        x = self._get_node_features(filename, graph_data)
        
        # Get edges
        edges = graph_data.get('edges', [])
        if edges:
            # Handle edge format: [src, tgt] or [src, tgt, type]
            sample_edge = edges[0]
            has_type = len(sample_edge) >= 3
            
            edge_list = []
            edge_types = []
            for edge in edges:
                edge_list.append([edge[0], edge[1]])
                edge_types.append(edge[2] if has_type else EDGE_TYPE_AST)
            
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_types, dtype=torch.long)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0,), dtype=torch.long)
        
        # Get label
        vuln_type = graph_data.get('vulnerability_type', 'clean')
        vuln_idx = VULN_TO_IDX.get(vuln_type, 0)
        binary_label = 0 if vuln_type == 'clean' else 1
        
        # Create PyG Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            graph_y=torch.tensor([vuln_idx], dtype=torch.long),
            binary_y=torch.tensor([binary_label], dtype=torch.long),
            num_nodes=x.size(0),
        )
        
        return data
    
    def _get_node_features(self, filename: str, graph_data: Dict) -> torch.Tensor:
        # Check cache
        if self._feature_cache is not None and filename in self._feature_cache:
            return self._feature_cache[filename]
        
        num_nodes = graph_data.get('num_nodes', 0)
        
        # Try FastText with AST data
        if self.use_fasttext and self.embedder is not None and self.ast_data_dir is not None:
            ast_file = self.ast_data_dir / filename
            if ast_file.exists():
                try:
                    with open(ast_file, 'r') as f:
                        ast_data = json.load(f)
                    features = self.embedder.embed_graph(ast_data)
                    
                    if self._feature_cache is not None:
                        self._feature_cache[filename] = features
                    return features
                except Exception as e:
                    warnings.warn(f"FastText embedding failed for {filename}: {e}")
        
        # Fallback: one-hot node types from graph_data
        if 'node_type_idx' in graph_data:
            # Format with node type indices (integers)
            node_type_idx = graph_data['node_type_idx']
            start_bytes = graph_data.get('node_start_bytes')
            end_bytes = graph_data.get('node_end_bytes')
            features = get_node_features_from_indices(
                node_type_idx,
                start_bytes=start_bytes,
                end_bytes=end_bytes,
            )
        elif 'nodes' in graph_data and len(graph_data['nodes']) > 0:
            # Old format with node list
            nodes = graph_data['nodes']
            if isinstance(nodes[0], dict):
                node_type_idx = [n.get('node_type_idx', 0) for n in nodes]
            else:
                node_type_idx = [0] * len(nodes)
            features = get_node_features_from_indices(node_type_idx)
        else:
            # Minimal fallback: unknown type for all nodes
            features = torch.zeros((num_nodes, NUM_NODE_TYPES + 3))
            features[:, 0] = 1.0  # Unknown type
        
        # Pad to match expected feature dim if needed
        if features.size(1) < self._feature_dim:
            padding = torch.zeros((features.size(0), self._feature_dim - features.size(1)))
            features = torch.cat([features, padding], dim=1)
        
        if self._feature_cache is not None:
            self._feature_cache[filename] = features
        
        return features
    
    def get_stats(self) -> BSGVDDataStats:
        # Load metadata if not already loaded
        self._load_metadata()
        
        total_nodes = 0
        total_edges = 0
        vuln_dist = {}
        num_clean = 0
        num_vulnerable = 0
        
        for i, meta in enumerate(self._metadata):
            total_nodes += meta['num_nodes']
            vuln_type = meta['vulnerability_type']
            vuln_dist[vuln_type] = vuln_dist.get(vuln_type, 0) + 1
            
            if vuln_type == 'clean':
                num_clean += 1
            else:
                num_vulnerable += 1
        
        # Count edges from first few files
        for graph_file in self.graph_files[:min(100, len(self.graph_files))]:
            with open(graph_file, 'r') as f:
                data = json.load(f)
            total_edges += len(data.get('edges', []))
        
        return BSGVDDataStats(
            num_graphs=len(self),
            num_nodes=total_nodes,
            num_edges=total_edges,
            feature_dim=self._feature_dim,
            num_clean=num_clean,
            num_vulnerable=num_vulnerable,
            vuln_distribution=vuln_dist,
        )


# ============================================================================
# Data Loader Factory Functions
# ============================================================================

def create_bsgvd_loaders(
    split_dir: Union[str, Path] = 'data/synthetic-split',
    ast_data_dir: Optional[Union[str, Path]] = 'data/synthetic/ast_dataset/ast_data',
    fasttext_model_path: Optional[Union[str, Path]] = 'models/fasttext/solidity.model',
    batch_size: int = 32,
    num_workers: int = 0,
    use_fasttext: bool = True,
    combine_with_type: bool = True,
    cache_features: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    split_dir = Path(split_dir)
    
    # Check for required directories
    train_dir = split_dir / 'train' / 'graph_data'
    val_dir = split_dir / 'dev' / 'graph_data'
    test_dir = split_dir / 'test' / 'graph_data'
    
    if not train_dir.exists():
        raise FileNotFoundError(f"Train directory not found: {train_dir}")
    if not val_dir.exists():
        raise FileNotFoundError(f"Val directory not found: {val_dir}")
    if not test_dir.exists():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")
    
    # Load FastText model once and share
    embedder = None
    if use_fasttext and fasttext_model_path:
        fasttext_path = Path(fasttext_model_path)
        if fasttext_path.exists() and GENSIM_AVAILABLE:
            try:
                model = load_fasttext_model(fasttext_path)
                embedder = FastTextEmbedder(
                    model=model,
                    combine_with_type=combine_with_type,
                    num_node_types=NUM_NODE_TYPES,
                )
                print(f"Loaded FastText model from {fasttext_path}")
                print(f"FastText feature dimension: {embedder.output_dim}")
            except Exception as e:
                warnings.warn(f"Failed to load FastText: {e}")
        else:
            if not fasttext_path.exists():
                warnings.warn(f"FastText model not found: {fasttext_path}")
            if not GENSIM_AVAILABLE:
                warnings.warn("gensim not installed. Using one-hot features.")
    
    # Create datasets
    train_dataset = BSGVDDataset(
        graph_dir=train_dir,
        ast_data_dir=ast_data_dir,
        fasttext_embedder=embedder,
        use_fasttext=use_fasttext,
        combine_with_type=combine_with_type,
        cache_features=cache_features,
    )
    
    val_dataset = BSGVDDataset(
        graph_dir=val_dir,
        ast_data_dir=ast_data_dir,
        fasttext_embedder=embedder,
        use_fasttext=use_fasttext,
        combine_with_type=combine_with_type,
        cache_features=cache_features,
    )
    
    test_dataset = BSGVDDataset(
        graph_dir=test_dir,
        ast_data_dir=ast_data_dir,
        fasttext_embedder=embedder,
        use_fasttext=use_fasttext,
        combine_with_type=combine_with_type,
        cache_features=cache_features,
    )
    
    # Print statistics
    train_stats = train_dataset.get_stats()
    print(f"\nBSGVD Dataset Statistics:")
    print(f"  Train: {train_stats.num_graphs} graphs ({train_stats.num_clean} clean, {train_stats.num_vulnerable} vulnerable)")
    print(f"  Val:   {len(val_dataset)} graphs")
    print(f"  Test:  {len(test_dataset)} graphs")
    print(f"  Feature dim: {train_stats.feature_dim}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    
    return train_loader, val_loader, test_loader


def create_bsgvd_loaders_with_irl_test(
    split_dir: Union[str, Path] = 'data/synthetic-split',
    irl_dir: Union[str, Path] = 'data/realworld/ast_dataset',
    ast_data_dir: Optional[Union[str, Path]] = 'data/synthetic/ast_dataset/ast_data',
    irl_ast_data_dir: Optional[Union[str, Path]] = None,
    fasttext_model_path: Optional[Union[str, Path]] = 'models/fasttext/solidity.model',
    batch_size: int = 32,
    num_workers: int = 0,
    use_fasttext: bool = True,
    combine_with_type: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    
    # Get synthetic loaders
    train_loader, val_loader, syn_test_loader = create_bsgvd_loaders(
        split_dir=split_dir,
        ast_data_dir=ast_data_dir,
        fasttext_model_path=fasttext_model_path,
        batch_size=batch_size,
        num_workers=num_workers,
        use_fasttext=use_fasttext,
        combine_with_type=combine_with_type,
    )
    
    # Create IRL test loader
    irl_dir = Path(irl_dir)
    irl_graph_dir = irl_dir / 'graph_data'
    
    if not irl_graph_dir.exists():
        warnings.warn(f"IRL graph directory not found: {irl_graph_dir}")
        return train_loader, val_loader, syn_test_loader, None
    
    # Get IRL AST data directory
    if irl_ast_data_dir is None:
        irl_ast_data_dir = irl_dir / 'ast_data'
        if not irl_ast_data_dir.exists():
            irl_ast_data_dir = None
    
    # Use same embedder
    embedder = None
    if use_fasttext and hasattr(train_loader.dataset, 'embedder'):
        embedder = train_loader.dataset.embedder
    
    irl_dataset = BSGVDDataset(
        graph_dir=irl_graph_dir,
        ast_data_dir=irl_ast_data_dir,
        fasttext_embedder=embedder,
        use_fasttext=use_fasttext,
        combine_with_type=combine_with_type,
        cache_features=True,
    )
    
    print(f"  IRL Test: {len(irl_dataset)} graphs")
    
    irl_test_loader = DataLoader(
        irl_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    
    return train_loader, val_loader, syn_test_loader, irl_test_loader


# ============================================================================
# Utility Functions
# ============================================================================

def get_bsgvd_input_dim(
    fasttext_model_path: Optional[Union[str, Path]] = None,
    use_fasttext: bool = True,
    combine_with_type: bool = True,
) -> int:
    
    if use_fasttext and fasttext_model_path and GENSIM_AVAILABLE:
        try:
            model = load_fasttext_model(fasttext_model_path)
            vector_size = model.wv.vector_size
            if combine_with_type:
                return vector_size + NUM_NODE_TYPES
            return vector_size
        except Exception:
            pass
    
    # Fallback: standard one-hot + position
    return NUM_NODE_TYPES + 3


def compute_bsgvd_class_weights(loader: DataLoader) -> torch.Tensor:
    num_clean = 0
    num_vuln = 0
    
    for batch in loader:
        labels = batch.binary_y.squeeze()
        num_clean += (labels == 0).sum().item()
        num_vuln += (labels == 1).sum().item()
    
    total = num_clean + num_vuln
    if total == 0:
        return torch.ones(2)
    
    # Inverse frequency weighting
    weight_clean = total / (2 * max(num_clean, 1))
    weight_vuln = total / (2 * max(num_vuln, 1))
    
    return torch.tensor([weight_clean, weight_vuln], dtype=torch.float)


# ============================================================================
# CLI for testing
# ============================================================================

if __name__ == '__main__':
    import sys
    import argparse
    
    # Add parent directory to path for imports
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    # Re-import after path fix
    from models.data import (
        VULN_TYPES, VULN_TO_IDX, IDX_TO_VULN,
        NODE_TYPES, NODE_TO_IDX, NUM_NODE_TYPES,
        NUM_EDGE_TYPES, EDGE_TYPE_AST,
        get_node_features_from_indices,
    )
    from models.graph_level.fasttext_embedding import (
        FastTextEmbedder,
        load_fasttext_model,
        GENSIM_AVAILABLE,
    )
    
    parser = argparse.ArgumentParser(description='Test BSGVD data loading')
    parser.add_argument('--split-dir', type=str, default='data/synthetic-split',
                       help='Directory with train/dev/test splits')
    parser.add_argument('--ast-dir', type=str, default='data/synthetic/ast_dataset/ast_data',
                       help='AST data directory')
    parser.add_argument('--fasttext', type=str, default='models/fasttext/solidity.model',
                       help='FastText model path')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--no-fasttext', action='store_true',
                       help='Disable FastText embeddings')
    
    args = parser.parse_args()
    
    print("Testing BSGVD Data Loader")
    print("=" * 60)
    
    # Create loaders
    train_loader, val_loader, test_loader = create_bsgvd_loaders(
        split_dir=args.split_dir,
        ast_data_dir=args.ast_dir,
        fasttext_model_path=args.fasttext if not args.no_fasttext else None,
        batch_size=args.batch_size,
        use_fasttext=not args.no_fasttext,
    )
    
    # Test loading a batch
    print("\nLoading sample batch...")
    batch = next(iter(train_loader))
    print(f"  Batch size: {batch.num_graphs}")
    print(f"  Node features shape: {batch.x.shape}")
    print(f"  Edge index shape: {batch.edge_index.shape}")
    print(f"  Edge attr shape: {batch.edge_attr.shape}")
    print(f"  Labels shape: {batch.binary_y.shape}")
    
    # Compute class weights
    print("\nComputing class weights...")
    weights = compute_bsgvd_class_weights(train_loader)
    print(f"  Class weights: {weights}")
    
    print("\n✓ BSGVD data loader test passed!")
