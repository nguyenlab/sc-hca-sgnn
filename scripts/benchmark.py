import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import warnings

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

from models import (
    create_model,
    create_dr_gcn_model,
    create_tmp_model,
    create_gat_model,
    create_transformer_gnn_model,
    create_bugsweeper_model,
    create_bugsweeper_light_model,
    create_scvhunter_model,
    create_mlagnn_model,
    create_bsgvd_model,
    ContractGraphDataset,
    create_data_loaders,
    create_data_loaders_with_irl_test,
    get_input_dim,
    get_num_classes,
    get_num_edge_types,
    IDX_TO_VULN,
)
from models.bsgvd_data import (
    create_bsgvd_loaders,
    create_bsgvd_loaders_with_irl_test,
    compute_bsgvd_class_weights,
)
from training.utils import (
    setup_seed,
    setup_device,
    train_with_early_stopping,
    evaluate_binary,
    evaluate_multiclass,
    compute_binary_weights,
)


def load_custom_configs(config_files: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Load custom per-model hyperparameters from grid search result files.
    
    Args:
        config_files: List of JSON file paths (supports glob patterns)
        
    Returns:
        Dictionary mapping model_key -> {hidden_dim, num_layers, heads, dropout, lr, ...}
    """
    import glob
    
    custom_configs = {}
    
    # Expand glob patterns
    all_files = []
    for pattern in config_files:
        expanded = glob.glob(pattern)
        if expanded:
            all_files.extend(expanded)
        elif Path(pattern).exists():
            all_files.append(pattern)
    
    for filepath in all_files:
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Handle grid search result format
            if 'results' in data and isinstance(data['results'], dict):
                # Format: {"results": {"model_name": {"best_config": {...}}}}
                for model_key, model_data in data['results'].items():
                    if model_data and 'best_config' in model_data and model_data['best_config']:
                        custom_configs[model_key] = model_data['best_config']['params']
            elif 'best_config' in data and data['best_config']:
                # Single model grid search format
                model_key = data.get('model', Path(filepath).stem.replace('grid_search_', ''))
                custom_configs[model_key] = data['best_config']['params']
            
        except Exception as e:
            print(f"Warning: Failed to load config from {filepath}: {e}")
    
    return custom_configs


# Model configurations
MODEL_CONFIGS = {
    'hierarchical-rgcn': {
        'name': 'Hierarchical RGCN',
        'paper': 'Baseline',
        'factory': lambda input_dim, hidden_dim, num_layers, heads, num_classes, dropout, num_edge_types: 
            create_model(
                model_type='hierarchical',
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
                num_classes=num_classes,
                num_edge_types=num_edge_types,
                mode='rgcn',
            ),
    },
    'dr-gcn': {
        'name': 'DR-GCN',
        'paper': 'IJCAI 2020',
        'factory': lambda input_dim, hidden_dim, num_layers, heads, num_classes, dropout, num_edge_types:
            create_dr_gcn_model(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                num_classes=num_classes,
                dropout=dropout,
            ),
    },
    'tmp': {
        'name': 'TMP',
        'paper': 'IJCAI 2020',
        'factory': lambda input_dim, hidden_dim, num_layers, heads, num_classes, dropout, num_edge_types:
            create_tmp_model(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                heads=heads,
                num_classes=num_classes,
                dropout=dropout,
            ),
    },
    'gat': {
        'name': 'GAT',
        'paper': 'ICLR 2018',
        'factory': lambda input_dim, hidden_dim, num_layers, heads, num_classes, dropout, num_edge_types:
            create_gat_model(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                heads=heads,
                num_classes=num_classes,
                dropout=dropout,
            ),
    },
    'transformer': {
        'name': 'TransformerGNN',
        'paper': 'Graph Transformer',
        'factory': lambda input_dim, hidden_dim, num_layers, heads, num_classes, dropout, num_edge_types:
            create_transformer_gnn_model(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                heads=heads,
                num_classes=num_classes,
                dropout=dropout,
                use_positional_encoding=True,
            ),
    },
    'bugsweeper': {
        'name': 'BugSweeper',
        'paper': 'AAAI 2026',
        'factory': lambda input_dim, hidden_dim, num_layers, heads, num_classes, dropout, num_edge_types:
            create_bugsweeper_model(
                input_dim=input_dim,
                num_classes=num_classes,
                dropout=dropout,
            ),
    },
    'bugsweeper-light': {
        'name': 'BugSweeperLight',
        'paper': 'AAAI 2026',
        'factory': lambda input_dim, hidden_dim, num_layers, heads, num_classes, dropout, num_edge_types:
            create_bugsweeper_light_model(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_classes=num_classes,
                dropout=dropout,
            ),
    },
    'scvhunter': {
        'name': 'SCVHUNTER',
        'paper': 'ICSE 2024',
        'factory': lambda input_dim, hidden_dim, num_layers, heads, num_classes, dropout, num_edge_types:
            create_scvhunter_model(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                num_edge_types=num_edge_types,
                heads=heads,
                num_classes=num_classes,
                dropout=dropout,
            ),
    },
    'mlagnn': {
        'name': 'ML-AGNN',
        'paper': 'MSN 2024',
        'factory': lambda input_dim, hidden_dim, num_layers, heads, num_classes, dropout, num_edge_types:
            create_mlagnn_model(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                num_edge_types=num_edge_types,
                heads=heads,
                num_classes=num_classes,
                dropout=dropout,
            ),
    },
    'bsgvd': {
        'name': 'BSGVD',
        'paper': 'IEEE 2024',
        'uses_fasttext': True,  # Flag indicating this model needs custom data loader
        'factory': lambda input_dim, hidden_dim, num_layers, heads, num_classes, dropout, num_edge_types:
            create_bsgvd_model(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                heads=heads,
                num_classes=num_classes,
                dropout=dropout,
                pooling='mean',
            ),
    },
}


def create_model_instance(
    model_key: str,
    input_dim: int,
    hidden_dim: int,
    num_layers: int,
    heads: int,
    num_classes: int,
    dropout: float,
    num_edge_types: int,
    device: str,
) -> nn.Module:
    """Create a model instance from configuration."""
    config = MODEL_CONFIGS[model_key]
    model = config['factory'](
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        heads=heads,
        num_classes=num_classes,
        dropout=dropout,
        num_edge_types=num_edge_types,
    )
    return model.to(device)


def benchmark_model(
    model_key: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    syn_test_loader: DataLoader,
    irl_test_loader: Optional[DataLoader],
    device: str,
    input_dim: int,
    hidden_dim: int,
    num_layers: int,
    heads: int,
    num_classes: int,
    dropout: float,
    num_edge_types: int,
    epochs: int,
    lr: float,
    patience: int,
    class_weights: Optional[torch.Tensor] = None,
    binary: bool = True,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Benchmark a single model on both synthetic and realworld test sets.
    
    Returns:
        Dictionary with metrics and training info
    """
    config = MODEL_CONFIGS[model_key]
    print(f"\n{'='*70}")
    print(f"Benchmarking: {config['name']} ({config['paper']})")
    print(f"{'='*70}")
    
    # Set seed for reproducibility
    setup_seed(seed)
    
    # Create model
    try:
        model = create_model_instance(
            model_key=model_key,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            heads=heads,
            num_classes=num_classes,
            dropout=dropout,
            num_edge_types=num_edge_types,
            device=device,
        )
    except Exception as e:
        print(f"  Error creating model: {e}")
        return {'model': config['name'], 'error': str(e)}
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {num_params:,}")
    
    # Create output directory for checkpoint
    output_dir = Path('outputs/benchmark/contract_level') / model_key
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Train
    start_time = time.time()
    try:
        _, history = train_with_early_stopping(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=epochs,
            lr=lr,
            patience=patience,
            output_dir=output_dir,
            class_weights=class_weights,
            binary=binary,
            verbose=False,
        )
    except Exception as e:
        print(f"  Error during training: {e}")
        return {'model': config['name'], 'error': str(e)}
    
    train_time = time.time() - start_time
    
    # Evaluate on synthetic test
    criterion = nn.CrossEntropyLoss()
    if binary:
        syn_metrics = evaluate_binary(model, syn_test_loader, criterion, device)
    else:
        syn_metrics = evaluate_multiclass(model, syn_test_loader, criterion, device)
    
    # Evaluate on realworld test if available
    irl_metrics = None
    if irl_test_loader is not None:
        if binary:
            irl_metrics = evaluate_binary(model, irl_test_loader, criterion, device)
        else:
            irl_metrics = evaluate_multiclass(model, irl_test_loader, criterion, device)
    
    # Compile results
    checkpoint_path = str(output_dir / 'best_model.pt')
    result = {
        'model': config['name'],
        'paper': config['paper'],
        'parameters': num_params,
        'train_time_sec': round(train_time, 2),
        'best_epoch': history.get('best_epoch', epochs),
        'checkpoint': checkpoint_path,
        'synthetic_test': {
            'accuracy': round(syn_metrics['accuracy'], 4),
            'precision': round(syn_metrics['precision'], 4),
            'recall': round(syn_metrics['recall'], 4),
            'f1': round(syn_metrics['f1'], 4),
        },
    }
    
    if binary:
        result['synthetic_test']['auc'] = round(syn_metrics['auc'], 4)
    
    # Print synthetic results
    print(f"  [Synthetic Test]")
    print(f"    Accuracy:  {result['synthetic_test']['accuracy']:.4f}")
    print(f"    Precision: {result['synthetic_test']['precision']:.4f}")
    print(f"    Recall:    {result['synthetic_test']['recall']:.4f}")
    print(f"    F1 Score:  {result['synthetic_test']['f1']:.4f}")
    if binary:
        print(f"    AUC:       {result['synthetic_test']['auc']:.4f}")
    
    # Add realworld results if available
    if irl_metrics is not None:
        result['realworld_test'] = {
            'accuracy': round(irl_metrics['accuracy'], 4),
            'precision': round(irl_metrics['precision'], 4),
            'recall': round(irl_metrics['recall'], 4),
            'f1': round(irl_metrics['f1'], 4),
        }
        if binary:
            result['realworld_test']['auc'] = round(irl_metrics['auc'], 4)
        
        print(f"  [Real-World Test]")
        print(f"    Accuracy:  {result['realworld_test']['accuracy']:.4f}")
        print(f"    Precision: {result['realworld_test']['precision']:.4f}")
        print(f"    Recall:    {result['realworld_test']['recall']:.4f}")
        print(f"    F1 Score:  {result['realworld_test']['f1']:.4f}")
        if binary:
            print(f"    AUC:       {result['realworld_test']['auc']:.4f}")
    
    print(f"  Train time: {train_time:.1f}s")
    
    return result


def run_benchmark(
    synthetic_data_path: str,
    realworld_data_path: Optional[str],
    models: List[str],
    task: str = 'binary',
    hidden_dim: int = 128,
    num_layers: int = 4,
    heads: int = 4,
    dropout: float = 0.3,
    epochs: int = 50,
    lr: float = 0.001,
    patience: int = 20,
    batch_size: int = 32,
    seed: int = 42,
    synthetic_only: bool = False,
    custom_configs: Optional[Dict[str, Dict[str, Any]]] = None,
    use_presplit: bool = True,
    fasttext_model_path: Optional[str] = 'models/fasttext/solidity_test.model',
    use_fasttext: bool = True,
    ast_data_dir: Optional[str] = 'data/synthetic/ast_dataset/ast_data',
) -> Dict[str, Any]:
    """
    Run benchmark on all specified models.
    
    Supports two modes:
    1. Full mode (default): Train on synthetic, test on both synthetic and realworld
    2. Synthetic-only mode: Train and test only on synthetic data (legacy mode)
    
    Returns:
        Dictionary with all results and configuration
    """
    setup_seed(seed)
    device = setup_device()
    
    print("="*70)
    print("SMART CONTRACT VULNERABILITY DETECTION BENCHMARK")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Task: {task}")
    print(f"  Device: {device}")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Num layers: {num_layers}")
    print(f"  Heads: {heads}")
    print(f"  Dropout: {dropout}")
    print(f"  Epochs: {epochs}")
    print(f"  Learning rate: {lr}")
    print(f"  Patience: {patience}")
    print(f"  Batch size: {batch_size}")
    print(f"  Seed: {seed}")
    print(f"  Mode: {'Synthetic-only' if synthetic_only else 'Synthetic + Real-world'}")
    print(f"  Data split: {'Pre-split (template-based)' if use_presplit else 'Random split (legacy)'}")
    if custom_configs:
        print(f"  Custom configs: {len(custom_configs)} models")
    
    # Check if using pre-split data
    base_path = Path(synthetic_data_path)
    has_presplit = (base_path / 'train').exists() and (base_path / 'dev').exists() and (base_path / 'test').exists()
    
    if use_presplit and has_presplit:
        print(f"\n[Synthetic Data] Loading pre-split data from {synthetic_data_path}...")
        train_dataset = ContractGraphDataset(root=str(base_path / 'train'), use_edge_attr=True, bidirectional=True)
        val_dataset = ContractGraphDataset(root=str(base_path / 'dev'), use_edge_attr=True, bidirectional=True)
        syn_test_dataset = ContractGraphDataset(root=str(base_path / 'test'), use_edge_attr=True, bidirectional=True)
        
        # Get combined stats
        train_stats = train_dataset.get_stats()
        val_stats = val_dataset.get_stats()
        test_stats = syn_test_dataset.get_stats()
        
        print(f"  Train: {train_stats.num_graphs} graphs")
        print(f"  Dev: {val_stats.num_graphs} graphs")
        print(f"  Test: {test_stats.num_graphs} graphs")
        print(f"  Total: {train_stats.num_graphs + val_stats.num_graphs + test_stats.num_graphs} graphs")
        
        # Use train stats for class weights
        syn_stats = train_stats
    else:
        if use_presplit and not has_presplit:
            print(f"\nWarning: Pre-split directories not found in {synthetic_data_path}")
            print("Falling back to random split (legacy mode)")
        
        print(f"\n[Synthetic Data] Loading from {synthetic_data_path}...")
        synthetic_dataset = ContractGraphDataset(root=synthetic_data_path, use_edge_attr=True, bidirectional=True)
        syn_stats = synthetic_dataset.get_stats()
        
        print(f"  Total graphs: {syn_stats.num_graphs}")
        print(f"  Vulnerability distribution:")
        for vtype, count in syn_stats.vuln_distribution.items():
            if count > 0:
                print(f"    {vtype}: {count}")
    
    # Load realworld dataset if provided and not in synthetic-only mode
    realworld_dataset = None
    irl_stats = None
    if realworld_data_path and not synthetic_only:
        print(f"\n[Real-World Data] Loading from {realworld_data_path}...")
        realworld_dataset = ContractGraphDataset(root=realworld_data_path, use_edge_attr=True, bidirectional=True)
        irl_stats = realworld_dataset.get_stats()
        
        print(f"  Total graphs: {irl_stats.num_graphs}")
        print(f"  Vulnerability distribution:")
        for vtype, count in irl_stats.vuln_distribution.items():
            if count > 0:
                print(f"    {vtype}: {count}")
    
    # Create data loaders
    if use_presplit and has_presplit:
        # Pre-split mode: use template-based splits
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        syn_test_loader = DataLoader(syn_test_dataset, batch_size=batch_size, shuffle=False)
        
        # Load realworld data if available
        irl_test_loader = None
        if realworld_data_path and not synthetic_only:
            realworld_dataset = ContractGraphDataset(root=realworld_data_path, use_edge_attr=True, bidirectional=True)
            irl_test_loader = DataLoader(realworld_dataset, batch_size=batch_size, shuffle=False)
        
        print(f"\n[Data Loaders] Pre-split (template-based):")
        print(f"  Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(syn_test_loader.dataset)}")
        if irl_test_loader:
            print(f"  Real-world IRL Test: {len(irl_test_loader.dataset)}")
    elif synthetic_only or realworld_dataset is None:
        # Legacy mode: split synthetic data into train/val/test
        train_loader, val_loader, syn_test_loader = create_data_loaders(
            synthetic_dataset, batch_size=batch_size, seed=seed
        )
        irl_test_loader = None
        print(f"\n[Data Split] Synthetic only (random split):")
        print(f"  Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(syn_test_loader.dataset)}")
    else:
        # Full mode: use synthetic for train/val/syn_test, realworld for irl_test
        train_loader, val_loader, syn_test_loader, irl_test_loader = create_data_loaders_with_irl_test(
            synthetic_dataset, realworld_dataset, batch_size=batch_size, seed=seed
        )
        print(f"\n[Data Split] Random split (legacy):")
        print(f"  Synthetic - Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(syn_test_loader.dataset)}")
        print(f"  Real-world - IRL Test: {len(irl_test_loader.dataset)}")
    
    # Setup task-specific params
    binary = (task == 'binary')
    if binary:
        num_classes = 2
        clean_count = syn_stats.vuln_distribution.get('clean', 0)
        vuln_count = sum(syn_stats.vuln_distribution.values()) - clean_count
        class_weights = compute_binary_weights(clean_count, vuln_count)
        print(f"\n  Class weights: clean={class_weights[0]:.2f}, vuln={class_weights[1]:.2f}")
    else:
        num_classes = get_num_classes()
        class_weights = None
    
    input_dim = get_input_dim()
    num_edge_types = get_num_edge_types()
    
    # Prepare BSGVD-specific data loaders if needed
    bsgvd_loaders = None
    bsgvd_input_dim = None
    bsgvd_class_weights = None
    if 'bsgvd' in models and use_presplit:
        print(f"\n[BSGVD] Creating FastText-based data loaders...")
        try:
            if realworld_data_path and not synthetic_only:
                bsgvd_train, bsgvd_val, bsgvd_syn_test, bsgvd_irl_test = create_bsgvd_loaders_with_irl_test(
                    split_dir=synthetic_data_path,
                    irl_dir=realworld_data_path,
                    ast_data_dir=ast_data_dir,
                    fasttext_model_path=fasttext_model_path if use_fasttext else None,
                    batch_size=batch_size,
                    use_fasttext=use_fasttext,
                )
                bsgvd_loaders = (bsgvd_train, bsgvd_val, bsgvd_syn_test, bsgvd_irl_test)
            else:
                bsgvd_train, bsgvd_val, bsgvd_syn_test = create_bsgvd_loaders(
                    split_dir=synthetic_data_path,
                    ast_data_dir=ast_data_dir,
                    fasttext_model_path=fasttext_model_path if use_fasttext else None,
                    batch_size=batch_size,
                    use_fasttext=use_fasttext,
                )
                bsgvd_loaders = (bsgvd_train, bsgvd_val, bsgvd_syn_test, None)
            
            bsgvd_input_dim = bsgvd_loaders[0].dataset.feature_dim
            bsgvd_class_weights = compute_bsgvd_class_weights(bsgvd_loaders[0].dataset)
            print(f"  BSGVD feature dim: {bsgvd_input_dim}")
            print(f"  BSGVD class weights: {bsgvd_class_weights.tolist()}")
        except Exception as e:
            print(f"  Warning: Failed to create BSGVD loaders: {e}")
            print(f"  BSGVD will use standard data loaders (no FastText)")
            bsgvd_loaders = None
    
    # Run benchmarks
    results = []
    total_start = time.time()
    
    for model_key in models:
        if model_key not in MODEL_CONFIGS:
            print(f"\nWarning: Unknown model '{model_key}', skipping...")
            continue
        
        # Use custom config if available, otherwise use defaults
        if custom_configs and model_key in custom_configs:
            cfg = custom_configs[model_key]
            model_hidden_dim = cfg.get('hidden_dim', hidden_dim)
            model_num_layers = cfg.get('num_layers', num_layers)
            model_heads = cfg.get('heads', heads)
            model_dropout = cfg.get('dropout', dropout)
            model_lr = cfg.get('lr', lr)
            print(f"\n  Using custom config: hidden_dim={model_hidden_dim}, num_layers={model_num_layers}, "
                  f"heads={model_heads}, dropout={model_dropout}, lr={model_lr}")
        else:
            model_hidden_dim = hidden_dim
            model_num_layers = num_layers
            model_heads = heads
            model_dropout = dropout
            model_lr = lr
        
        # Use BSGVD-specific loaders and input_dim if available
        if model_key == 'bsgvd' and bsgvd_loaders is not None:
            model_train_loader, model_val_loader, model_syn_test_loader, model_irl_test_loader = bsgvd_loaders
            model_input_dim = bsgvd_input_dim
            model_class_weights = bsgvd_class_weights
        else:
            model_train_loader = train_loader
            model_val_loader = val_loader
            model_syn_test_loader = syn_test_loader
            model_irl_test_loader = irl_test_loader
            model_input_dim = input_dim
            model_class_weights = class_weights
        
        result = benchmark_model(
            model_key=model_key,
            train_loader=model_train_loader,
            val_loader=model_val_loader,
            syn_test_loader=model_syn_test_loader,
            irl_test_loader=model_irl_test_loader,
            device=device,
            input_dim=model_input_dim,
            hidden_dim=model_hidden_dim,
            num_layers=model_num_layers,
            heads=model_heads,
            num_classes=num_classes,
            dropout=model_dropout,
            num_edge_types=num_edge_types,
            epochs=epochs,
            lr=model_lr,
            patience=patience,
            class_weights=model_class_weights,
            binary=binary,
            seed=seed,
        )
        
        # Add config info to result
        if custom_configs and model_key in custom_configs:
            result['config'] = custom_configs[model_key]
        
        results.append(result)
    
    total_time = time.time() - total_start
    
    # Compile final output
    output = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'task': task,
            'synthetic_data_path': synthetic_data_path,
            'realworld_data_path': realworld_data_path,
            'synthetic_only': synthetic_only,
            'hidden_dim': hidden_dim,
            'num_layers': num_layers,
            'heads': heads,
            'dropout': dropout,
            'epochs': epochs,
            'lr': lr,
            'patience': patience,
            'batch_size': batch_size,
            'seed': seed,
        },
        'synthetic_stats': {
            'num_graphs': syn_stats.num_graphs,
            'vuln_distribution': syn_stats.vuln_distribution,
        },
        'results': results,
        'total_time_sec': round(total_time, 2),
    }
    
    if irl_stats:
        output['realworld_stats'] = {
            'num_graphs': irl_stats.num_graphs,
            'vuln_distribution': irl_stats.vuln_distribution,
        }
    
    return output


def print_results_table(results: Dict[str, Any]):
    """Print results as a formatted table with both synthetic and realworld results."""
    print("\n")
    print("="*120)
    print("BENCHMARK RESULTS")
    print("="*120)
    
    task = results['config']['task']
    has_irl = any('realworld_test' in r for r in results['results'] if 'error' not in r)
    
    if has_irl:
        # Two-section table for synthetic and realworld
        print("\n📊 SYNTHETIC TEST RESULTS")
        print("-"*100)
        if task == 'binary':
            header = f"{'Model':<25} {'Paper':<12} {'Params':>10} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'AUC':>7} {'Time':>7}"
        else:
            header = f"{'Model':<25} {'Paper':<12} {'Params':>10} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'Time':>7}"
        print(header)
        print("-"*100)
        
        sorted_results = sorted(
            [r for r in results['results'] if 'error' not in r],
            key=lambda x: x['synthetic_test']['f1'],
            reverse=True
        )
        
        for r in sorted_results:
            m = r['synthetic_test']
            paper = r.get('paper', 'N/A')[:12]
            if task == 'binary':
                row = f"{r['model']:<25} {paper:<12} {r['parameters']:>10,} {m['accuracy']:>7.4f} {m['precision']:>7.4f} {m['recall']:>7.4f} {m['f1']:>7.4f} {m['auc']:>7.4f} {r['train_time_sec']:>6.1f}s"
            else:
                row = f"{r['model']:<25} {paper:<12} {r['parameters']:>10,} {m['accuracy']:>7.4f} {m['precision']:>7.4f} {m['recall']:>7.4f} {m['f1']:>7.4f} {r['train_time_sec']:>6.1f}s"
            print(row)
        
        print("\n🌍 REAL-WORLD (IRL) TEST RESULTS")
        print("-"*100)
        print(header)
        print("-"*100)
        
        sorted_irl = sorted(
            [r for r in results['results'] if 'error' not in r and 'realworld_test' in r],
            key=lambda x: x['realworld_test']['f1'],
            reverse=True
        )
        
        for r in sorted_irl:
            m = r['realworld_test']
            paper = r.get('paper', 'N/A')[:12]
            if task == 'binary':
                row = f"{r['model']:<25} {paper:<12} {r['parameters']:>10,} {m['accuracy']:>7.4f} {m['precision']:>7.4f} {m['recall']:>7.4f} {m['f1']:>7.4f} {m['auc']:>7.4f} {r['train_time_sec']:>6.1f}s"
            else:
                row = f"{r['model']:<25} {paper:<12} {r['parameters']:>10,} {m['accuracy']:>7.4f} {m['precision']:>7.4f} {m['recall']:>7.4f} {m['f1']:>7.4f} {r['train_time_sec']:>6.1f}s"
            print(row)
        
        # Summary comparison
        if task == 'binary':
            print("\n📈 COMPARISON SUMMARY (Sorted by IRL F1)")
            print("-"*85)
            print(f"{'Model':<25} {'Syn F1':>10} {'IRL F1':>10} {'Gap':>10} {'Syn AUC':>10} {'IRL AUC':>10}")
            print("-"*85)
            
            for r in sorted_irl:
                syn_f1 = r['synthetic_test']['f1']
                irl_f1 = r['realworld_test']['f1']
                gap = irl_f1 - syn_f1
                syn_auc = r['synthetic_test']['auc']
                irl_auc = r['realworld_test']['auc']
                gap_str = f"{gap:+.4f}"
                print(f"{r['model']:<25} {syn_f1:>10.4f} {irl_f1:>10.4f} {gap_str:>10} {syn_auc:>10.4f} {irl_auc:>10.4f}")
        
        if sorted_irl:
            best_syn = sorted_results[0]
            best_irl = sorted_irl[0]
            print(f"\n🏆 Best by Synthetic F1: {best_syn['model']} (F1={best_syn['synthetic_test']['f1']:.4f})")
            print(f"🏆 Best by Real-World F1: {best_irl['model']} (F1={best_irl['realworld_test']['f1']:.4f})")
    else:
        # Single table for synthetic-only mode
        if task == 'binary':
            header = f"{'Model':<25} {'Paper':<15} {'Params':>12} {'Acc':>8} {'Prec':>8} {'Rec':>8} {'F1':>8} {'AUC':>8} {'Time':>8}"
        else:
            header = f"{'Model':<25} {'Paper':<15} {'Params':>12} {'Acc':>8} {'Prec':>8} {'Rec':>8} {'F1':>8} {'Time':>8}"
        
        print(header)
        print("-"*100)
        
        sorted_results = sorted(
            [r for r in results['results'] if 'error' not in r],
            key=lambda x: x['synthetic_test']['f1'],
            reverse=True
        )
        
        for r in sorted_results:
            m = r['synthetic_test']
            if task == 'binary':
                row = f"{r['model']:<25} {r['paper']:<15} {r['parameters']:>12,} {m['accuracy']:>8.4f} {m['precision']:>8.4f} {m['recall']:>8.4f} {m['f1']:>8.4f} {m['auc']:>8.4f} {r['train_time_sec']:>7.1f}s"
            else:
                row = f"{r['model']:<25} {r['paper']:<15} {r['parameters']:>12,} {m['accuracy']:>8.4f} {m['precision']:>8.4f} {m['recall']:>8.4f} {m['f1']:>8.4f} {r['train_time_sec']:>7.1f}s"
            print(row)
        
        if sorted_results:
            best = sorted_results[0]
            print(f"\n🏆 Best model by F1: {best['model']} (F1={best['synthetic_test']['f1']:.4f})")
    
    # Print errors if any
    errors = [r for r in results['results'] if 'error' in r]
    if errors:
        print("\n❌ Failed models:")
        for r in errors:
            print(f"  {r.get('model', 'Unknown')}: {r['error']}")
    
    print("-"*120)
    print(f"Total benchmark time: {results['total_time_sec']:.1f}s")


def main():
    parser = argparse.ArgumentParser(
        description='Benchmark GNN models for Smart Contract Vulnerability Detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run benchmark with both synthetic and realworld test (default)
    python benchmark.py --epochs 50 --task binary
    
    # Run benchmark on specific models
    python benchmark.py --models dr-gcn tmp gat scvhunter --epochs 50
    
    # Specify custom data paths
    python benchmark.py --synthetic-data data/synthetic/ast_dataset \\
                        --realworld-data data/realworld/ast_dataset
    
    # Legacy mode: synthetic data only
    python benchmark.py --synthetic-only --synthetic-data data/ast_dataset --epochs 50
    
    # Quick test with fewer epochs
    python benchmark.py --epochs 10 --models dr-gcn gat
    
    # Use best configs from grid search (per-model hyperparameters)
    python benchmark.py --config results/grid_search/contract_level/grid_search_*.json \\
                        --models scvhunter mlagnn dr-gcn gat --epochs 50
        """
    )
    
    # Data paths
    parser.add_argument('--synthetic-data', '-s', default='data/synthetic-split',
                        help='Synthetic (injected) dataset directory')
    parser.add_argument('--realworld-data', '-r', default='data/realworld/ast_dataset',
                        help='Real-world dataset directory')
    parser.add_argument('--synthetic-only', action='store_true',
                        help='Use only synthetic data (legacy mode)')
    
    # Legacy argument for backward compatibility
    parser.add_argument('--data', '-d', default=None,
                        help='[DEPRECATED] Use --synthetic-data instead')
    
    parser.add_argument('--task', '-t', choices=['binary', 'multiclass'], default='binary',
                        help='Task type (binary or multiclass)')
    parser.add_argument('--models', '-m', nargs='+', default=list(MODEL_CONFIGS.keys()),
                        choices=list(MODEL_CONFIGS.keys()),
                        help='Models to benchmark')
    parser.add_argument('--hidden-dim', type=int, default=128,
                        help='Hidden dimension')
    parser.add_argument('--num-layers', type=int, default=4,
                        help='Number of GNN layers')
    parser.add_argument('--heads', type=int, default=4,
                        help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout rate')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Maximum training epochs')
    parser.add_argument('--lr', type=float, default=0.0005,
                        help='Learning rate')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--output', '-o', type=str, default='results/benchmark/contract_level.json',
                        help='Output JSON file for results')
    parser.add_argument('--config', '-c', nargs='+', default=None,
                        help='Custom config JSON files (e.g., grid search results). '
                             'Supports glob patterns like results/grid_search/*.json')
    parser.add_argument('--use-presplit', action='store_true', default=True,
                        help='Use pre-split data if available (default: True)')
    parser.add_argument('--random-split', action='store_true',
                        help='Force random split even if pre-split data exists')
    
    # BSGVD-specific arguments
    parser.add_argument('--fasttext-model', type=str, default='models/fasttext/solidity_test.model',
                        help='Path to FastText model for BSGVD (default: models/fasttext/solidity_test.model)')
    parser.add_argument('--no-fasttext', action='store_true',
                        help='Disable FastText for BSGVD (use one-hot features only)')
    parser.add_argument('--ast-data-dir', type=str, default='data/synthetic/ast_dataset/ast_data',
                        help='Directory with AST JSON files for FastText embeddings')
    
    args = parser.parse_args()
    
    # Handle random-split flag
    if args.random_split:
        args.use_presplit = False
    
    # Handle legacy --data argument
    if args.data is not None:
        print("Warning: --data is deprecated, use --synthetic-data instead")
        synthetic_data_path = args.data
        args.synthetic_only = True  # Assume legacy mode if --data is used
    else:
        synthetic_data_path = args.synthetic_data
    
    # Check if synthetic dataset exists
    syn_path = Path(synthetic_data_path)
    if not syn_path.exists():
        print(f"Error: Synthetic dataset not found at {syn_path}")
        print("Please build the dataset first:")
        print(f"  python main.py build --input data/sc-source/vulnerable --output {synthetic_data_path}")
        return 1
    
    # Check realworld dataset if not in synthetic-only mode
    realworld_data_path = None
    if not args.synthetic_only:
        irl_path = Path(args.realworld_data)
        if irl_path.exists():
            realworld_data_path = args.realworld_data
        else:
            print(f"Warning: Real-world dataset not found at {irl_path}")
            print("Running in synthetic-only mode.")
            print("To build real-world dataset, run:")
            print(f"  python dataset_builder.py --input data/realworld/sc-source --output {args.realworld_data}")
    
    # Load custom configs if provided
    custom_configs = None
    if args.config:
        custom_configs = load_custom_configs(args.config)
        if custom_configs:
            print(f"\nLoaded custom configs for {len(custom_configs)} models:")
            for model_key in custom_configs:
                print(f"  - {model_key}")
        else:
            print("\nWarning: No valid configs found in provided files.")
    
    # Run benchmark
    results = run_benchmark(
        synthetic_data_path=synthetic_data_path,
        realworld_data_path=realworld_data_path,
        models=args.models,
        task=args.task,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        heads=args.heads,
        dropout=args.dropout,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
        batch_size=args.batch_size,
        seed=args.seed,
        synthetic_only=args.synthetic_only,
        custom_configs=custom_configs,
        use_presplit=args.use_presplit,
        fasttext_model_path=args.fasttext_model,
        use_fasttext=not args.no_fasttext,
        ast_data_dir=args.ast_data_dir,
    )
    
    # Print results table
    print_results_table(results)
    
    # Save to file
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")
    
    return 0


if __name__ == '__main__':
    exit(main())
