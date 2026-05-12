import argparse
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List
import shutil


def load_split_metadata(metadata_file: Path) -> Dict[str, List[str]]:
    """Load template assignments from graph-level split."""
    with open(metadata_file) as f:
        metadata = json.load(f)
    
    template_splits = {}
    for split_name, split_info in metadata['splits'].items():
        template_splits[split_name] = set(split_info['templates'])
    
    return template_splits


def extract_template(filename: str) -> str:
    """
    Extract template name from function-level filename.
    Function-level files have the same name as their corresponding graph-level files.
    
    Examples:
        0x0001.json -> 0x0001.json
        AccessControl_0001.json -> AccessControl_0001.json
    """
    # Function-level files use the same naming as graph-level files
    return filename


def load_function_graphs(data_dir: Path) -> List[Dict]:
    """Load all function-level graph JSON files."""
    func_graph_dir = data_dir / "function_graphs"
    if not func_graph_dir.exists():
        raise FileNotFoundError(f"Function graph directory not found: {func_graph_dir}")
    
    graphs = []
    for json_file in sorted(func_graph_dir.glob("*.json")):
        with open(json_file) as f:
            data = json.load(f)
            data['filename'] = json_file.name
            graphs.append(data)
    
    print(f"Loaded {len(graphs)} function-level graphs from {func_graph_dir}")
    return graphs


def split_function_graphs(graphs: List[Dict], template_splits: Dict[str, set]) -> Dict[str, List[Dict]]:
    """Split function graphs according to template assignments."""
    splits = {
        'train': [],
        'dev': [],
        'test': []
    }
    
    unmatched = []
    
    for graph in graphs:
        filename = graph['filename']
        template = extract_template(filename)
        
        # Find which split this template belongs to
        assigned = False
        for split_name, templates in template_splits.items():
            if template in templates:
                splits[split_name].append(graph)
                assigned = True
                break
        
        if not assigned:
            unmatched.append(filename)
    
    if unmatched:
        print(f"\nWarning: {len(unmatched)} function graphs could not be matched to templates:")
        for fname in unmatched[:10]:
            print(f"  - {fname}")
        if len(unmatched) > 10:
            print(f"  ... and {len(unmatched) - 10} more")
    
    return splits


def save_function_split(graphs: List[Dict], output_dir: Path, split_name: str):
    """Save function-level split to disk."""
    split_dir = output_dir / split_name
    func_graph_dir = split_dir / "function_graphs"
    func_graph_dir.mkdir(parents=True, exist_ok=True)
    
    # Save function graph JSON files
    for graph in graphs:
        filename = graph.get('filename', f'func_{len(graphs)}.json')
        output_file = func_graph_dir / filename
        
        # Remove filename field before saving
        graph_copy = {k: v for k, v in graph.items() if k != 'filename'}
        
        with open(output_file, 'w') as f:
            json.dump(graph_copy, f, indent=2)
    
    # Create processed directory marker
    processed_dir = split_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Saved {len(graphs)} function graphs to {split_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Split function-level dataset using template assignments'
    )
    parser.add_argument(
        '--input',
        type=Path,
        default=Path('data/synthetic/function_level_dataset'),
        help='Input function-level dataset directory'
    )
    parser.add_argument(
        '--metadata',
        type=Path,
        default=Path('data/synthetic-split/split_metadata.json'),
        help='Split metadata file from graph-level split'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('data/synthetic-split'),
        help='Output directory for split datasets'
    )
    
    args = parser.parse_args()
    
    print(f"Loading split metadata from {args.metadata}...")
    template_splits = load_split_metadata(args.metadata)
    
    print(f"\nTemplate splits:")
    for split_name, templates in template_splits.items():
        print(f"  {split_name}: {len(templates)} templates")
    
    print(f"\nLoading function-level graphs from {args.input}...")
    graphs = load_function_graphs(args.input)
    
    print("\nSplitting function graphs by template...")
    splits = split_function_graphs(graphs, template_splits)
    
    print(f"\nFunction graph distribution:")
    for split_name, graphs in splits.items():
        print(f"  {split_name}: {len(graphs)} function graphs")
    
    print(f"\nSaving splits to {args.output}...")
    for split_name, graphs in splits.items():
        save_function_split(graphs, args.output, split_name)
    
    print(f"\n{'='*60}")
    print("✓ Function-level dataset split successfully!")
    print(f"{'='*60}")
    print(f"Output directory: {args.output}")
    print(f"  - train: {len(splits['train'])} function graphs")
    print(f"  - dev: {len(splits['dev'])} function graphs")
    print(f"  - test: {len(splits['test'])} function graphs")


if __name__ == '__main__':
    main()
