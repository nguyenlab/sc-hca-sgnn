import json
from pathlib import Path
from collections import defaultdict


def get_templates_from_split(split_dir: Path) -> set:
    """Get all templates in a split."""
    graph_dir = split_dir / "graph_data"
    templates = set()
    
    for json_file in graph_dir.glob("*.json"):
        templates.add(json_file.name)
    
    return templates


def main():
    split_dir = Path("data/synthetic-split")
    
    train_templates = get_templates_from_split(split_dir / "train")
    dev_templates = get_templates_from_split(split_dir / "dev")
    test_templates = get_templates_from_split(split_dir / "test")
    
    print("="*60)
    print("SPLIT VERIFICATION")
    print("="*60)
    print(f"Train templates: {len(train_templates)}")
    print(f"Dev templates: {len(dev_templates)}")
    print(f"Test templates: {len(test_templates)}")
    print(f"Total: {len(train_templates) + len(dev_templates) + len(test_templates)}")
    
    # Check overlaps
    train_dev = train_templates & dev_templates
    train_test = train_templates & test_templates
    dev_test = dev_templates & test_templates
    
    print("\n" + "="*60)
    print("OVERLAP CHECK")
    print("="*60)
    print(f"Train-Dev overlap: {len(train_dev)}")
    if train_dev:
        print(f"  Examples: {list(train_dev)[:5]}")
    
    print(f"Train-Test overlap: {len(train_test)}")
    if train_test:
        print(f"  Examples: {list(train_test)[:5]}")
    
    print(f"Dev-Test overlap: {len(dev_test)}")
    if dev_test:
        print(f"  Examples: {list(dev_test)[:5]}")
    
    if not (train_dev or train_test or dev_test):
        print("\n✓ NO OVERLAP DETECTED - Split is clean!")
    else:
        print("\n✗ OVERLAP DETECTED - Data leakage risk!")
        return 1
    
    # Verify function-level matches graph-level
    print("\n" + "="*60)
    print("FUNCTION-LEVEL VERIFICATION")
    print("="*60)
    
    for split_name in ['train', 'dev', 'test']:
        graph_templates = get_templates_from_split(split_dir / split_name)
        func_dir = split_dir / split_name / "function_graphs"
        func_templates = {f.name for f in func_dir.glob("*.json")}
        
        missing_in_func = graph_templates - func_templates
        extra_in_func = func_templates - graph_templates
        
        print(f"{split_name}:")
        print(f"  Graph-level: {len(graph_templates)} files")
        print(f"  Function-level: {len(func_templates)} files")
        print(f"  Missing in function-level: {len(missing_in_func)}")
        print(f"  Extra in function-level: {len(extra_in_func)}")
        
        if missing_in_func:
            print(f"    Examples: {list(missing_in_func)[:3]}")
        if extra_in_func:
            print(f"    Examples: {list(extra_in_func)[:3]}")
    
    print("\n✓ Verification complete!")
    return 0


if __name__ == '__main__':
    exit(main())
