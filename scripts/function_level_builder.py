import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Set
from collections import defaultdict


# Edge type constants (from dataset_builder.py)
EDGE_TYPE_AST = 0       # Parent-child (structural)
EDGE_TYPE_REF = 1       # Reference/data flow
EDGE_TYPE_CFG_NEXT = 2  # Control flow: next statement
EDGE_TYPE_CFG_TRUE = 3  # Control flow: true branch
EDGE_TYPE_CFG_FALSE = 4 # Control flow: false branch
EDGE_TYPE_CALL = 5      # Function call
EDGE_TYPE_INHERIT = 6   # Inheritance
EDGE_TYPE_GUARD = 7     # Guard condition

NUM_EDGE_TYPES = 8


@dataclass
class FunctionInfo:
    """Information about a function in the contract."""
    node_id: int  # Node ID of FunctionDefinition
    name: str
    start_byte: int
    end_byte: int
    is_vulnerable: bool
    contained_node_ids: List[int] = field(default_factory=list)


@dataclass
class FunctionLevelGraph:
    """Function-level graph representation."""
    contract_path: str
    vulnerability_type: str
    
    # Node data
    num_nodes: int
    node_types: List[str]
    node_type_idx: List[int]
    node_start_bytes: List[int]
    node_end_bytes: List[int]
    node_is_vulnerable: List[bool]
    
    # Function assignment for each node (0 = global/not in function, 1..N = function index)
    node_function_id: List[int]
    
    # Edges
    edges: List[Tuple[int, int, int]]  # (src, dst, edge_type)
    
    # Intra-function edges (edges where both nodes belong to same function)
    intra_function_edges: List[Tuple[int, int, int]]
    
    # Inter-function edges (edges crossing function boundaries)
    inter_function_edges: List[Tuple[int, int, int]]
    
    # Function information
    functions: List[Dict[str, Any]]  # List of function info dicts
    function_labels: List[int]  # 0/1 for each function (is vulnerable)
    
    # Statistics
    edge_counts: Dict[str, int] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'contract_path': self.contract_path,
            'vulnerability_type': self.vulnerability_type,
            'num_nodes': self.num_nodes,
            'node_types': self.node_types,
            'node_type_idx': self.node_type_idx,
            'node_start_bytes': self.node_start_bytes,
            'node_end_bytes': self.node_end_bytes,
            'node_is_vulnerable': self.node_is_vulnerable,
            'node_function_id': self.node_function_id,
            'edges': self.edges,
            'intra_function_edges': self.intra_function_edges,
            'inter_function_edges': self.inter_function_edges,
            'functions': self.functions,
            'function_labels': self.function_labels,
            'num_functions': len(self.functions),
            'edge_counts': self.edge_counts,
        }


def load_ast_data(ast_path: Path) -> Dict[str, Any]:
    """Load AST data from JSON file."""
    with open(ast_path, 'r') as f:
        return json.load(f)


def load_graph_data(graph_path: Path) -> Dict[str, Any]:
    """Load graph data from JSON file."""
    with open(graph_path, 'r') as f:
        return json.load(f)


def find_function_definitions(nodes: List[Dict]) -> List[FunctionInfo]:
    """
    Find all FunctionDefinition nodes and their byte ranges.
    
    Args:
        nodes: List of node dictionaries from AST data
        
    Returns:
        List of FunctionInfo objects
    """
    functions = []
    
    for node in nodes:
        node_type = node.get('node_type', '')
        if node_type == 'FunctionDefinition':
            func_info = FunctionInfo(
                node_id=node['node_id'],
                name=node.get('attributes', {}).get('name', f'function_{node["node_id"]}'),
                start_byte=node['start_byte'],
                end_byte=node['end_byte'],
                is_vulnerable=node.get('is_vulnerable', False),
            )
            functions.append(func_info)
    
    return functions


def assign_nodes_to_functions(
    nodes: List[Dict],
    functions: List[FunctionInfo],
) -> Tuple[List[int], List[FunctionInfo]]:
    """
    Assign each node to a function based on byte range containment.
    
    A node belongs to a function if its byte range is completely contained
    within the function's byte range.
    
    Args:
        nodes: List of node dictionaries
        functions: List of FunctionInfo objects
        
    Returns:
        Tuple of (node_function_ids, updated_functions)
        - node_function_ids: List where index i is the function ID for node i
          (0 = global/not in function, 1..N = function index + 1)
    """
    num_nodes = len(nodes)
    node_function_id = [0] * num_nodes  # 0 means global/not in any function
    
    # Sort functions by start_byte to handle nested functions properly
    # (inner functions should take precedence)
    sorted_functions = sorted(functions, key=lambda f: (-f.start_byte, f.end_byte))
    
    for node in nodes:
        node_id = node['node_id']
        node_start = node.get('start_byte', 0)
        node_end = node.get('end_byte', 0)
        
        # Find the innermost function containing this node
        for func_idx, func in enumerate(sorted_functions):
            if func.start_byte <= node_start and node_end <= func.end_byte:
                # This node is contained in this function
                # Map to original function index (not sorted index)
                original_idx = functions.index(func)
                node_function_id[node_id] = original_idx + 1  # 1-indexed
                func.contained_node_ids.append(node_id)
                
                # Also check if this node is vulnerable
                if node.get('is_vulnerable', False):
                    func.is_vulnerable = True
                break
    
    return node_function_id, functions


def classify_edges(
    edges: List[Tuple[int, int, int]],
    node_function_id: List[int],
) -> Tuple[List[Tuple[int, int, int]], List[Tuple[int, int, int]]]:
    """
    Classify edges into intra-function and inter-function edges.
    
    Args:
        edges: List of (src, dst, edge_type) tuples
        node_function_id: Function assignment for each node
        
    Returns:
        Tuple of (intra_function_edges, inter_function_edges)
    """
    intra_edges = []
    inter_edges = []
    
    for src, dst, etype in edges:
        src_func = node_function_id[src] if src < len(node_function_id) else 0
        dst_func = node_function_id[dst] if dst < len(node_function_id) else 0
        
        if src_func == dst_func and src_func > 0:
            # Both nodes in same function (and not global)
            intra_edges.append((src, dst, etype))
        else:
            # Cross-function or involving global nodes
            inter_edges.append((src, dst, etype))
    
    return intra_edges, inter_edges


def process_contract(
    ast_path: Path,
    graph_path: Path,
) -> Optional[FunctionLevelGraph]:
    """
    Process a single contract into function-level graph format.
    
    Args:
        ast_path: Path to AST JSON file
        graph_path: Path to graph JSON file
        
    Returns:
        FunctionLevelGraph if successful, None otherwise
    """
    try:
        ast_data = load_ast_data(ast_path)
        graph_data = load_graph_data(graph_path)
        
        nodes = ast_data['nodes']
        
        # Find all FunctionDefinition nodes
        functions = find_function_definitions(nodes)
        
        if not functions:
            # No functions in this contract, skip
            return None
        
        # Assign nodes to functions
        node_function_id, functions = assign_nodes_to_functions(nodes, functions)
        
        # Get edges from graph data
        edges = [tuple(e) for e in graph_data['edges']]
        
        # Classify edges
        intra_edges, inter_edges = classify_edges(edges, node_function_id)
        
        # Create function labels (1 if any node in function is vulnerable)
        function_labels = [1 if f.is_vulnerable else 0 for f in functions]
        
        # Create function info dicts
        functions_list = [
            {
                'function_id': i + 1,  # 1-indexed
                'node_id': f.node_id,
                'name': f.name,
                'start_byte': f.start_byte,
                'end_byte': f.end_byte,
                'is_vulnerable': f.is_vulnerable,
                'num_nodes': len(f.contained_node_ids),
            }
            for i, f in enumerate(functions)
        ]
        
        # Edge counts
        edge_counts = graph_data.get('edge_counts', {})
        edge_counts['intra_function'] = len(intra_edges)
        edge_counts['inter_function'] = len(inter_edges)
        
        # Create FunctionLevelGraph
        func_graph = FunctionLevelGraph(
            contract_path=graph_data.get('contract_path', str(graph_path)),
            vulnerability_type=graph_data.get('vulnerability_type', 'unknown'),
            num_nodes=len(nodes),
            node_types=graph_data.get('node_types', [n.get('node_type', 'Unknown') for n in nodes]),
            node_type_idx=graph_data.get('node_type_idx', [n.get('node_type_idx', 0) for n in nodes]),
            node_start_bytes=graph_data.get('node_start_bytes', [n.get('start_byte', 0) for n in nodes]),
            node_end_bytes=graph_data.get('node_end_bytes', [n.get('end_byte', 0) for n in nodes]),
            node_is_vulnerable=graph_data.get('node_is_vulnerable', [n.get('is_vulnerable', False) for n in nodes]),
            node_function_id=node_function_id,
            edges=edges,
            intra_function_edges=intra_edges,
            inter_function_edges=inter_edges,
            functions=functions_list,
            function_labels=function_labels,
            edge_counts=edge_counts,
        )
        
        return func_graph
        
    except Exception as e:
        print(f"Error processing {ast_path}: {e}")
        return None


class FunctionLevelDatasetBuilder:
    """
    Builds function-level dataset from existing AST dataset.
    """
    
    def __init__(self, input_dir: Path, output_dir: Path):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        
        self.ast_dir = self.input_dir / 'ast_data'
        self.graph_dir = self.input_dir / 'graph_data'
        
        self.stats = {
            'total_contracts': 0,
            'successful': 0,
            'failed': 0,
            'skipped_no_functions': 0,
            'total_functions': 0,
            'vulnerable_functions': 0,
            'clean_functions': 0,
            'by_vulnerability_type': defaultdict(int),
            'errors': [],
        }
    
    def discover_contracts(self) -> List[Tuple[Path, Path]]:
        """
        Discover all AST/graph file pairs.
        
        Returns:
            List of (ast_path, graph_path) tuples
        """
        if not self.ast_dir.exists() or not self.graph_dir.exists():
            raise FileNotFoundError(f"AST or graph directory not found in {self.input_dir}")
        
        pairs = []
        for ast_file in sorted(self.ast_dir.glob('*.json')):
            graph_file = self.graph_dir / ast_file.name
            if graph_file.exists():
                pairs.append((ast_file, graph_file))
        
        return pairs
    
    def build_dataset(self, verbose: bool = True) -> Dict[str, Any]:
        """
        Build the function-level dataset.
        
        Args:
            verbose: Print progress information
            
        Returns:
            Dataset statistics
        """
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        func_graph_dir = self.output_dir / 'function_graphs'
        func_graph_dir.mkdir(exist_ok=True)
        
        # Discover contracts
        file_pairs = self.discover_contracts()
        self.stats['total_contracts'] = len(file_pairs)
        
        if verbose:
            print(f"Found {len(file_pairs)} contracts to process")
            print("=" * 60)
        
        # Process each contract
        processed_graphs = []
        
        for i, (ast_path, graph_path) in enumerate(file_pairs):
            if verbose and (i + 1) % 100 == 0:
                print(f"[{i+1}/{len(file_pairs)}] Processing...")
            
            func_graph = process_contract(ast_path, graph_path)
            
            if func_graph is None:
                # Check if it's an error or just no functions
                try:
                    ast_data = load_ast_data(ast_path)
                    functions = find_function_definitions(ast_data['nodes'])
                    if not functions:
                        self.stats['skipped_no_functions'] += 1
                    else:
                        self.stats['failed'] += 1
                except:
                    self.stats['failed'] += 1
                continue
            
            self.stats['successful'] += 1
            self.stats['total_functions'] += len(func_graph.functions)
            self.stats['vulnerable_functions'] += sum(func_graph.function_labels)
            self.stats['clean_functions'] += len(func_graph.function_labels) - sum(func_graph.function_labels)
            self.stats['by_vulnerability_type'][func_graph.vulnerability_type] += 1
            
            # Save function-level graph
            output_file = func_graph_dir / ast_path.name
            with open(output_file, 'w') as f:
                json.dump(func_graph.to_dict(), f, indent=2)
            
            processed_graphs.append(func_graph)
        
        # Convert defaultdicts
        self.stats['by_vulnerability_type'] = dict(self.stats['by_vulnerability_type'])
        
        # Save statistics
        stats_path = self.output_dir / 'dataset_stats.json'
        with open(stats_path, 'w') as f:
            json.dump(self.stats, f, indent=2)
        
        # Save index
        self._save_index(processed_graphs)
        
        if verbose:
            print("=" * 60)
            print(f"Function-level dataset built!")
            print(f"  Total contracts: {self.stats['total_contracts']}")
            print(f"  Successful: {self.stats['successful']}")
            print(f"  Skipped (no functions): {self.stats['skipped_no_functions']}")
            print(f"  Failed: {self.stats['failed']}")
            print(f"\nFunction Statistics:")
            print(f"  Total functions: {self.stats['total_functions']}")
            print(f"  Vulnerable: {self.stats['vulnerable_functions']}")
            print(f"  Clean: {self.stats['clean_functions']}")
            print(f"\nOutput saved to: {self.output_dir}")
        
        return self.stats
    
    def _save_index(self, graphs: List[FunctionLevelGraph]) -> None:
        """Save dataset index."""
        index = {
            'total_contracts': len(graphs),
            'total_functions': self.stats['total_functions'],
            'contracts': [
                {
                    'path': g.contract_path,
                    'vulnerability_type': g.vulnerability_type,
                    'num_nodes': g.num_nodes,
                    'num_functions': len(g.functions),
                    'num_vulnerable_functions': sum(g.function_labels),
                    'num_edges': len(g.edges),
                    'num_intra_edges': len(g.intra_function_edges),
                    'num_inter_edges': len(g.inter_function_edges),
                }
                for g in graphs
            ]
        }
        
        index_path = self.output_dir / 'dataset_index.json'
        with open(index_path, 'w') as f:
            json.dump(index, f, indent=2)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Build function-level dataset for vulnerability detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python function_level_builder.py --input data/ast_dataset --output data/function_level_dataset
"""
    )
    
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input AST dataset directory (with ast_data/ and graph_data/)"
    )
    parser.add_argument(
        "--output", "-o", 
        required=True,
        help="Output directory for function-level dataset"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output"
    )
    
    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_arguments()
    
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    
    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        return 1
    
    builder = FunctionLevelDatasetBuilder(input_dir, output_dir)
    stats = builder.build_dataset(verbose=not args.quiet)
    
    if stats['failed'] > 0:
        print(f"\nWarning: {stats['failed']} contracts failed to process.")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
