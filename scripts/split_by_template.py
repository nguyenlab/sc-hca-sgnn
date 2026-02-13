import argparse
import json
import shutil
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set
import random


def load_graph_data(data_dir: Path) -> List[Dict]:
    """Load all graph data JSON files."""
    graph_data_dir = data_dir / "graph_data"
    if not graph_data_dir.exists():
        raise FileNotFoundError(f"Graph data directory not found: {graph_data_dir}")
    
    graphs = []
    for json_file in sorted(graph_data_dir.glob("*.json")):
        with open(json_file) as f:
            data = json.load(f)
            data['filename'] = json_file.name
            graphs.append(data)
    
    print(f"Loaded {len(graphs)} graphs from {graph_data_dir}")
    return graphs


def extract_template(contract_name: str) -> str:
    """
    Extract template name from contract name.
    
    Examples:
        AccessControl_0001.sol -> AccessControl
        Reentrancy_0042.sol -> Reentrancy
        BadRandomness_v2_0010.sol -> BadRandomness_v2
    """
    # Remove .sol extension
    name = contract_name.replace('.sol', '')
    
    # Find the last underscore followed by digits
    parts = name.split('_')
    
    # If last part is numeric, it's the instance number
    if parts[-1].isdigit():
        template = '_'.join(parts[:-1])
    else:
        template = name
    
    return template


def group_by_template(graphs: List[Dict]) -> Dict[str, List[Dict]]:
    """Group contracts by their template."""
    template_groups = defaultdict(list)
    
    for graph in graphs:
        contract_name = graph.get('contract_name', graph['filename'])
        template = extract_template(contract_name)
        template_groups[template].append(graph)
    
    return dict(template_groups)


def split_templates(template_groups: Dict[str, List[Dict]], 
                    train_ratio: float = 0.7,
                    val_ratio: float = 0.15,
                    test_ratio: float = 0.15,
                    seed: int = 42) -> Dict[str, List[Dict]]:
    """
    Split templates into train/val/test sets.
    Each template appears in only ONE split.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    
    random.seed(seed)
    
    # Get all templates and shuffle
    templates = list(template_groups.keys())
    random.shuffle(templates)
    
    # Calculate split points
    n_templates = len(templates)
    n_train = int(n_templates * train_ratio)
    n_val = int(n_templates * val_ratio)
    
    # Split templates
    train_templates = templates[:n_train]
    val_templates = templates[n_train:n_train + n_val]
    test_templates = templates[n_train + n_val:]
    
    # Collect graphs for each split
    splits = {
        'train': [],
        'dev': [],
        'test': []
    }
    
    for template in train_templates:
        splits['train'].extend(template_groups[template])
    
    for template in val_templates:
        splits['dev'].extend(template_groups[template])
    
    for template in test_templates:
        splits['test'].extend(template_groups[template])
    
    # Print statistics
    print(f"\n{'='*60}")
    print("TEMPLATE-BASED SPLIT STATISTICS")
    print(f"{'='*60}")
    print(f"Total templates: {n_templates}")
    print(f"Train templates: {len(train_templates)} ({len(train_templates)/n_templates*100:.1f}%)")
    print(f"Val templates: {len(val_templates)} ({len(val_templates)/n_templates*100:.1f}%)")
    print(f"Test templates: {len(test_templates)} ({len(test_templates)/n_templates*100:.1f}%)")
    print(f"\nTotal contracts: {sum(len(graphs) for graphs in splits.values())}")
    print(f"Train contracts: {len(splits['train'])} ({len(splits['train'])/sum(len(graphs) for graphs in splits.values())*100:.1f}%)")
    print(f"Val contracts: {len(splits['dev'])} ({len(splits['dev'])/sum(len(graphs) for graphs in splits.values())*100:.1f}%)")
    print(f"Test contracts: {len(splits['test'])} ({len(splits['test'])/sum(len(graphs) for graphs in splits.values())*100:.1f}%)")
    
    # Verify no overlap
    train_set = {extract_template(g.get('contract_name', g['filename'])) for g in splits['train']}
    val_set = {extract_template(g.get('contract_name', g['filename'])) for g in splits['dev']}
    test_set = {extract_template(g.get('contract_name', g['filename'])) for g in splits['test']}
    
    overlap_train_val = train_set & val_set
    overlap_train_test = train_set & test_set
    overlap_val_test = val_set & test_set
    
    print(f"\n{'='*60}")
    print("OVERLAP CHECK (should all be 0)")
    print(f"{'='*60}")
    print(f"Train-Val overlap: {len(overlap_train_val)}")
    print(f"Train-Test overlap: {len(overlap_train_test)}")
    print(f"Val-Test overlap: {len(overlap_val_test)}")
    
    if overlap_train_val or overlap_train_test or overlap_val_test:
        raise RuntimeError("Template overlap detected! This should not happen.")
    
    return splits


def save_split(graphs: List[Dict], output_dir: Path, split_name: str):
    """Save a split to disk."""
    split_dir = output_dir / split_name
    graph_data_dir = split_dir / "graph_data"
    graph_data_dir.mkdir(parents=True, exist_ok=True)
    
    # Save graph JSON files
    for i, graph in enumerate(graphs):
        filename = graph.get('filename', f'graph_{i:04d}.json')
        output_file = graph_data_dir / filename
        
        # Remove filename field before saving (it was added for tracking)
        graph_copy = {k: v for k, v in graph.items() if k != 'filename'}
        
        with open(output_file, 'w') as f:
            json.dump(graph_copy, f, indent=2)
    
    # Create processed directory marker
    processed_dir = split_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Saved {len(graphs)} graphs to {split_dir}")


def save_split_metadata(template_groups: Dict[str, List[Dict]], 
                       splits: Dict[str, List[Dict]], 
                       output_dir: Path):
    """Save metadata about the split for reference."""
    metadata = {
        'total_templates': len(template_groups),
        'total_graphs': sum(len(graphs) for graphs in template_groups.values()),
        'splits': {}
    }
    
    for split_name, graphs in splits.items():
        templates = {extract_template(g.get('contract_name', g['filename'])) 
                    for g in graphs}
        metadata['splits'][split_name] = {
            'num_templates': len(templates),
            'num_graphs': len(graphs),
            'templates': sorted(list(templates))
        }
    
    metadata_file = output_dir / 'split_metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nSaved split metadata to {metadata_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Split dataset by template to avoid data leakage'
    )
    parser.add_argument(
        '--input',
        type=Path,
        default=Path('data/synthetic/ast_dataset'),
        help='Input dataset directory'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('data/synthetic-split'),
        help='Output directory for split datasets'
    )
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.7,
        help='Ratio of templates for training (default: 0.7)'
    )
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.15,
        help='Ratio of templates for validation (default: 0.15)'
    )
    parser.add_argument(
        '--test-ratio',
        type=float,
        default=0.15,
        help='Ratio of templates for testing (default: 0.15)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    args = parser.parse_args()
    
    print(f"Loading dataset from {args.input}...")
    graphs = load_graph_data(args.input)
    
    print("\nGrouping by template...")
    template_groups = group_by_template(graphs)
    print(f"Found {len(template_groups)} unique templates")
    
    # Show template statistics
    print(f"\nTemplate distribution:")
    template_sizes = sorted([(t, len(graphs)) for t, graphs in template_groups.items()],
                           key=lambda x: -x[1])
    for template, count in template_sizes[:10]:
        print(f"  {template}: {count} contracts")
    if len(template_sizes) > 10:
        print(f"  ... and {len(template_sizes) - 10} more templates")
    
    print("\nSplitting templates...")
    splits = split_templates(
        template_groups,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )
    
    print(f"\nSaving splits to {args.output}...")
    args.output.mkdir(parents=True, exist_ok=True)
    
    for split_name, graphs in splits.items():
        save_split(graphs, args.output, split_name)
    
    save_split_metadata(template_groups, splits, args.output)
    
    print(f"\n{'='*60}")
    print("✓ Dataset split successfully!")
    print(f"{'='*60}")
    print(f"Output directory: {args.output}")
    print(f"  - train: {len(splits['train'])} contracts")
    print(f"  - dev: {len(splits['dev'])} contracts")
    print(f"  - test: {len(splits['test'])} contracts")
    print("\nNo template overlap between splits - data leakage prevented!")


if __name__ == '__main__':
    main()
