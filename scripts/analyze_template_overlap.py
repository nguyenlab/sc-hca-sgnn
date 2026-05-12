import json
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple
import torch


def load_dataset_index(data_path: Path) -> Dict:
    """Load dataset index with contract metadata."""
    index_path = data_path / 'dataset_index.json'
    if not index_path.exists():
        raise FileNotFoundError(f"Dataset index not found: {index_path}")
    
    with open(index_path, 'r') as f:
        return json.load(f)


def analyze_template_distribution(dataset_index: Dict) -> Dict[str, List[str]]:
    """Group contracts by template name."""
    template_to_contracts = defaultdict(list)
    
    for contract in dataset_index['contracts']:
        template = contract.get('template_name', 'unknown')
        contract_path = contract['path']
        template_to_contracts[template].append(contract_path)
    
    return dict(template_to_contracts)


def simulate_data_split(
    dataset_index: Dict,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[Set[str], Set[str], Set[str]]:
    """
    Simulate the train/val/test split used by create_data_loaders.
    
    Returns:
        Tuple of (train_templates, val_templates, test_templates)
    """
    torch.manual_seed(seed)
    
    num_graphs = len(dataset_index['contracts'])
    indices = torch.randperm(num_graphs).tolist()
    
    train_end = int(train_ratio * num_graphs)
    val_end = int((train_ratio + val_ratio) * num_graphs)
    
    train_indices = set(indices[:train_end])
    val_indices = set(indices[train_end:val_end])
    test_indices = set(indices[val_end:])
    
    # Get templates for each split
    train_templates = set()
    val_templates = set()
    test_templates = set()
    
    for i, contract in enumerate(dataset_index['contracts']):
        template = contract.get('template_name', 'unknown')
        if i in train_indices:
            train_templates.add(template)
        elif i in val_indices:
            val_templates.add(template)
        elif i in test_indices:
            test_templates.add(template)
    
    return train_templates, val_templates, test_templates


def analyze_overlap(
    train_templates: Set[str],
    val_templates: Set[str],
    test_templates: Set[str],
    template_to_contracts: Dict[str, List[str]],
) -> Dict[str, any]:
    """Analyze template overlap between splits."""
    
    # Find overlaps
    train_val_overlap = train_templates & val_templates
    train_test_overlap = train_templates & test_templates
    val_test_overlap = val_templates & test_templates
    all_overlap = train_templates & val_templates & test_templates
    
    # Count contracts affected
    def count_contracts(templates: Set[str]) -> int:
        return sum(len(template_to_contracts[t]) for t in templates)
    
    return {
        'summary': {
            'total_templates': len(template_to_contracts),
            'train_templates': len(train_templates),
            'val_templates': len(val_templates),
            'test_templates': len(test_templates),
        },
        'overlaps': {
            'train_val': {
                'num_templates': len(train_val_overlap),
                'num_contracts': count_contracts(train_val_overlap),
                'templates': sorted(list(train_val_overlap)),
            },
            'train_test': {
                'num_templates': len(train_test_overlap),
                'num_contracts': count_contracts(train_test_overlap),
                'templates': sorted(list(train_test_overlap)),
            },
            'val_test': {
                'num_templates': len(val_test_overlap),
                'num_contracts': count_contracts(val_test_overlap),
                'templates': sorted(list(val_test_overlap)),
            },
            'all_three': {
                'num_templates': len(all_overlap),
                'num_contracts': count_contracts(all_overlap),
                'templates': sorted(list(all_overlap)),
            },
        },
        'template_distribution': {
            template: {
                'count': len(contracts),
                'in_train': template in train_templates,
                'in_val': template in val_templates,
                'in_test': template in test_templates,
            }
            for template, contracts in template_to_contracts.items()
        }
    }


def print_analysis_report(analysis: Dict):
    """Print a formatted analysis report."""
    print("\n" + "="*80)
    print("TEMPLATE OVERLAP ANALYSIS")
    print("="*80)
    
    summary = analysis['summary']
    print(f"\nDataset Summary:")
    print(f"  Total unique templates: {summary['total_templates']}")
    print(f"  Templates in train: {summary['train_templates']}")
    print(f"  Templates in val:   {summary['val_templates']}")
    print(f"  Templates in test:  {summary['test_templates']}")
    
    overlaps = analysis['overlaps']
    
    print(f"\n{'='*80}")
    print("OVERLAP ANALYSIS")
    print("="*80)
    
    def print_overlap(name: str, data: Dict):
        print(f"\n{name}:")
        print(f"  Templates: {data['num_templates']}")
        print(f"  Contracts: {data['num_contracts']}")
        if data['templates']:
            print(f"  Templates list: {', '.join(data['templates'][:10])}")
            if len(data['templates']) > 10:
                print(f"  ... and {len(data['templates']) - 10} more")
    
    print_overlap("Train-Val Overlap (⚠️ MINOR ISSUE)", overlaps['train_val'])
    print_overlap("Train-Test Overlap (❌ DATA LEAKAGE!)", overlaps['train_test'])
    print_overlap("Val-Test Overlap (⚠️ MINOR ISSUE)", overlaps['val_test'])
    print_overlap("All Three Splits (❌ CRITICAL!)", overlaps['all_three'])
    
    # Print verdict
    print(f"\n{'='*80}")
    print("VERDICT")
    print("="*80)
    
    if overlaps['train_test']['num_templates'] > 0:
        print("\n❌ DATA LEAKAGE DETECTED!")
        print(f"   {overlaps['train_test']['num_templates']} templates appear in BOTH training and test sets")
        print(f"   This affects {overlaps['train_test']['num_contracts']} contracts")
        print("\n   RECOMMENDATION: Use stratified splitting by template to prevent leakage")
        print("   or ensure each template's contracts all go to one split.")
    else:
        print("\n✅ NO TRAIN-TEST LEAKAGE")
        print("   Templates are properly separated between training and testing")
    
    if overlaps['train_val']['num_templates'] > 0 or overlaps['val_test']['num_templates'] > 0:
        print("\n⚠️  Train-Val or Val-Test overlap exists (less critical but not ideal)")
    
    # Show template distribution details
    print(f"\n{'='*80}")
    print("TEMPLATE DISTRIBUTION DETAILS")
    print("="*80)
    
    dist = analysis['template_distribution']
    
    # Templates in all three splits
    all_three = [t for t, d in dist.items() if d['in_train'] and d['in_val'] and d['in_test']]
    if all_three:
        print(f"\n❌ Templates in ALL splits ({len(all_three)}):")
        for template in sorted(all_three)[:20]:
            count = dist[template]['count']
            print(f"  - {template}: {count} contracts")
        if len(all_three) > 20:
            print(f"  ... and {len(all_three) - 20} more")
    
    # Templates in train and test (but not val)
    train_test_only = [t for t, d in dist.items() if d['in_train'] and d['in_test'] and not d['in_val']]
    if train_test_only:
        print(f"\n❌ Templates in TRAIN and TEST only ({len(train_test_only)}):")
        for template in sorted(train_test_only)[:20]:
            count = dist[template]['count']
            print(f"  - {template}: {count} contracts")
        if len(train_test_only) > 20:
            print(f"  ... and {len(train_test_only) - 20} more")
    
    # Templates unique to each split
    train_only = [t for t, d in dist.items() if d['in_train'] and not d['in_val'] and not d['in_test']]
    val_only = [t for t, d in dist.items() if not d['in_train'] and d['in_val'] and not d['in_test']]
    test_only = [t for t, d in dist.items() if not d['in_train'] and not d['in_val'] and d['in_test']]
    
    print(f"\n✅ Templates unique to TRAIN: {len(train_only)}")
    print(f"✅ Templates unique to VAL:   {len(val_only)}")
    print(f"✅ Templates unique to TEST:  {len(test_only)}")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze template overlap in train/val/test splits',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument('--data', '-d', default='data/synthetic/ast_dataset',
                        help='Dataset directory')
    parser.add_argument('--train-ratio', type=float, default=0.7,
                        help='Train ratio (default: 0.7)')
    parser.add_argument('--val-ratio', type=float, default=0.15,
                        help='Validation ratio (default: 0.15)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output JSON file for detailed results')
    
    args = parser.parse_args()
    
    data_path = Path(args.data)
    
    # Load dataset
    print(f"Loading dataset from {data_path}...")
    dataset_index = load_dataset_index(data_path)
    
    print(f"Total contracts: {len(dataset_index['contracts'])}")
    
    # Analyze template distribution
    print("\nAnalyzing template distribution...")
    template_to_contracts = analyze_template_distribution(dataset_index)
    
    print(f"Total unique templates: {len(template_to_contracts)}")
    
    # Simulate data split
    print("\nSimulating train/val/test split...")
    train_templates, val_templates, test_templates = simulate_data_split(
        dataset_index,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    
    # Analyze overlap
    print("\nAnalyzing template overlap...")
    analysis = analyze_overlap(
        train_templates,
        val_templates,
        test_templates,
        template_to_contracts,
    )
    
    # Print report
    print_analysis_report(analysis)
    
    # Save detailed results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        print(f"\n\nDetailed results saved to: {output_path}")


if __name__ == '__main__':
    main()
