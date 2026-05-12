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

warnings.filterwarnings('ignore')

from models.data import get_input_dim, get_num_edge_types
from models.function_level_data import (
    FunctionLevelDataset,
    create_function_level_loaders,
    create_function_level_loaders_with_irl_test,
)
# Import base function-level model
from models.subgraph_level.function_level_gnn import create_function_level_model
# Import adapted models from subgraph_level
from models.subgraph_level.dr_gcn import (
    create_function_level_dr_gcn_model,
    create_function_level_tmp_model,
)
from models.subgraph_level.attention_models import (
    create_function_level_gat_model,
    create_function_level_transformer_model,
)
from models.subgraph_level.scvhunter import create_function_level_scvhunter_model
from models.subgraph_level.mlagnn import create_function_level_mlagnn_model
from models.subgraph_level.hierarchical_cross_attention import (
    create_hca_sgnn_model,
    create_function_level_hca_model,
)

from training.function_level_utils import (
    train_function_level_with_early_stopping,
    evaluate_function_level,
    evaluate_contract_level,
    compute_function_weights,
    print_function_level_results,
)
from training.utils import setup_seed, setup_device


# Model configurations for benchmark
# Includes both base models and adapted graph-level architectures
MODEL_CONFIGS = {
    # ========== Base FunctionLevelGNN variants ==========
    'func-transformer-mean': {
        'name': 'FunctionLevel-Transformer-Mean',
        'type': 'base',
        'conv_type': 'transformer',
        'pooling': 'mean',
        'paper': 'Baseline',
    },
    'func-gat-mean': {
        'name': 'FunctionLevel-GAT-Mean',
        'type': 'base',
        'conv_type': 'gat',
        'pooling': 'mean',
        'paper': 'Baseline',
    },
    'func-sage-mean': {
        'name': 'FunctionLevel-SAGE-Mean',
        'type': 'base',
        'conv_type': 'sage',
        'pooling': 'mean',
        'paper': 'Baseline',
    },
    'func-sage-max': {
        'name': 'FunctionLevel-SAGE-Max',
        'type': 'base',
        'conv_type': 'sage',
        'pooling': 'max',
        'paper': 'Baseline',
    },
    
    # ========== Adapted Graph-Level Architectures ==========
    'func-dr-gcn': {
        'name': 'FunctionLevel-DR-GCN',
        'type': 'dr-gcn',
        'pooling': 'mean',
        'paper': 'IJCAI 2020',
    },
    'func-dr-gcn-both': {
        'name': 'FunctionLevel-DR-GCN-Both',
        'type': 'dr-gcn',
        'pooling': 'both',
        'paper': 'IJCAI 2020',
    },
    'func-tmp': {
        'name': 'FunctionLevel-TMP',
        'type': 'tmp',
        'pooling': 'mean',
        'paper': 'IJCAI 2020',
    },
    'func-tmp-both': {
        'name': 'FunctionLevel-TMP-Both',
        'type': 'tmp',
        'pooling': 'both',
        'paper': 'IJCAI 2020',
    },
    'func-gat-adapted': {
        'name': 'FunctionLevel-GAT-Adapted',
        'type': 'gat-adapted',
        'pooling': 'mean',
        'paper': 'ICLR 2018',
    },
    'func-transformer-adapted': {
        'name': 'FunctionLevel-Transformer-Adapted',
        'type': 'transformer-adapted',
        'pooling': 'mean',
        'paper': 'Graph Transformer',
    },
    'func-scvhunter': {
        'name': 'FunctionLevel-SCVHUNTER',
        'type': 'scvhunter',
        'pooling': 'mean',
        'paper': 'ICSE 2024',
    },
    'func-scvhunter-both': {
        'name': 'FunctionLevel-SCVHUNTER-Both',
        'type': 'scvhunter',
        'pooling': 'both',
        'paper': 'ICSE 2024',
    },
    'func-mlagnn': {
        'name': 'FunctionLevel-MLAGNN',
        'type': 'mlagnn',
        'pooling': 'mean',
        'paper': 'MSN 2024',
    },
    'func-mlagnn-both': {
        'name': 'FunctionLevel-MLAGNN-Both',
        'type': 'mlagnn',
        'pooling': 'both',
        'paper': 'MSN 2024',
    },
    
    # ========== Novel HCA-SGNN Architecture ==========
    'func-hca': {
        'name': 'HCA-SGNN',
        'type': 'hca',
        'pooling': 'hierarchical',
        'use_cross_attention': True,
        'paper': 'Ours',
    },
    'func-hca-nocross': {
        'name': 'HCA-SGNN-NoCross',
        'type': 'hca',
        'pooling': 'hierarchical',
        'use_cross_attention': False,
        'paper': 'Ours (ablation)',
    },
    'func-hca-mean': {
        'name': 'HCA-SGNN-Mean',
        'type': 'hca',
        'pooling': 'mean',
        'use_cross_attention': True,
        'paper': 'Ours (ablation)',
    },
}


def create_model_for_benchmark(
    model_key: str,
    config: Dict[str, Any],
    input_dim: int,
    hidden_dim: int,
    num_layers: int,
    heads: int,
    dropout: float,
    num_edge_types: int,
    device: torch.device,
) -> nn.Module:
    """
    Create a function-level model based on configuration.
    """
    model_type = config['type']
    pooling = config.get('pooling', 'mean')
    
    if model_type == 'base':
        # Base FunctionLevelGNN
        model = create_function_level_model(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            heads=heads,
            num_classes=2,
            dropout=dropout,
            conv_type=config['conv_type'],
            pooling=pooling,
        )
    elif model_type == 'dr-gcn':
        model = create_function_level_dr_gcn_model(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=2,
            dropout=dropout,
            pooling=pooling,
        )
    elif model_type == 'tmp':
        model = create_function_level_tmp_model(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            heads=heads,
            num_classes=2,
            dropout=dropout,
            pooling=pooling,
        )
    elif model_type == 'gat-adapted':
        model = create_function_level_gat_model(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            heads=heads,
            num_classes=2,
            dropout=dropout,
            pooling=pooling,
        )
    elif model_type == 'transformer-adapted':
        model = create_function_level_transformer_model(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            heads=heads,
            num_classes=2,
            dropout=dropout,
            pooling=pooling,
        )
    elif model_type == 'scvhunter':
        model = create_function_level_scvhunter_model(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_edge_types=num_edge_types,
            heads=heads,
            num_classes=2,
            dropout=dropout,
            pooling=pooling,
        )
    elif model_type == 'mlagnn':
        model = create_function_level_mlagnn_model(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_edge_types=num_edge_types,
            heads=heads,
            num_classes=2,
            dropout=dropout,
            pooling=pooling,
        )
    elif model_type == 'hca':
        model = create_hca_sgnn_model(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_edge_types=num_edge_types,
            heads=heads,
            num_classes=2,
            dropout=dropout,
            pooling=pooling,
            use_cross_attention=config.get('use_cross_attention', True),
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model.to(device)


def benchmark_model(
    model_key: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    syn_test_loader: DataLoader,
    irl_test_loader: Optional[DataLoader],
    device: torch.device,
    input_dim: int,
    hidden_dim: int,
    num_layers: int,
    heads: int,
    dropout: float,
    epochs: int,
    lr: float,
    patience: int,
    num_edge_types: int,
    class_weights: Optional[torch.Tensor] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Benchmark a single function-level model on both synthetic and realworld test sets.
    """
    config = MODEL_CONFIGS[model_key]
    print(f"\n{'='*70}")
    print(f"Benchmarking: {config['name']} ({config.get('paper', 'N/A')})")
    print(f"{'='*70}")
    
    setup_seed(seed)
    
    # Create model
    try:
        model = create_model_for_benchmark(
            model_key=model_key,
            config=config,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            heads=heads,
            dropout=dropout,
            num_edge_types=num_edge_types,
            device=device,
        )
    except Exception as e:
        print(f"  Error creating model: {e}")
        import traceback
        traceback.print_exc()
        return {'model': config['name'], 'error': str(e)}
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {num_params:,}")
    print(f"  Type: {config['type']}, Pooling: {config.get('pooling', 'mean')}")
    
    # Create output directory for checkpoint
    output_dir = Path('outputs/benchmark/function_level') / model_key
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Train
    start_time = time.time()
    try:
        _, history = train_function_level_with_early_stopping(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=epochs,
            lr=lr,
            patience=patience,
            output_dir=output_dir,
            class_weights=class_weights,
            verbose=False,
        )
    except Exception as e:
        print(f"  Error during training: {e}")
        import traceback
        traceback.print_exc()
        return {'model': config['name'], 'error': str(e)}
    
    train_time = time.time() - start_time
    
    # Evaluate on synthetic test
    criterion = nn.CrossEntropyLoss()
    syn_metrics = evaluate_function_level(model, syn_test_loader, criterion, device)
    
    # Evaluate contract-level performance on synthetic test
    syn_contract_metrics = evaluate_contract_level(model, syn_test_loader, device)
    
    # Evaluate on realworld test if available
    irl_metrics = None
    irl_contract_metrics = None
    if irl_test_loader is not None:
        irl_metrics = evaluate_function_level(model, irl_test_loader, criterion, device)
        irl_contract_metrics = evaluate_contract_level(model, irl_test_loader, device)
    
    checkpoint_path = str(output_dir / 'best_model.pt')
    result = {
        'model': config['name'],
        'type': config['type'],
        'pooling': config.get('pooling', 'mean'),
        'paper': config.get('paper', 'N/A'),
        'parameters': num_params,
        'train_time_sec': round(train_time, 2),
        'checkpoint': checkpoint_path,
        'synthetic_test': {
            'function_level': {
                'accuracy': round(syn_metrics['accuracy'], 4),
                'precision': round(syn_metrics['precision'], 4),
                'recall': round(syn_metrics['recall'], 4),
                'f1': round(syn_metrics['f1'], 4),
                'auc': round(syn_metrics['auc'], 4),
                'num_functions': syn_metrics['num_functions'],
            },
            'contract_level': {
                'accuracy': round(syn_contract_metrics['accuracy'], 4),
                'precision': round(syn_contract_metrics['precision'], 4),
                'recall': round(syn_contract_metrics['recall'], 4),
                'f1': round(syn_contract_metrics['f1'], 4),
                'auc': round(syn_contract_metrics['auc'], 4),
                'num_contracts': syn_contract_metrics['num_contracts'],
            },
        },
    }
    
    print(f"  [Synthetic Test - Function Level]")
    print(f"    Functions: {syn_metrics['num_functions']}")
    print(f"    Accuracy:  {syn_metrics['accuracy']:.4f}")
    print(f"    Precision: {syn_metrics['precision']:.4f}")
    print(f"    Recall:    {syn_metrics['recall']:.4f}")
    print(f"    F1 Score:  {syn_metrics['f1']:.4f}")
    print(f"    AUC:       {syn_metrics['auc']:.4f}")
    
    print(f"  [Synthetic Test - Contract Level]")
    print(f"    Contracts: {syn_contract_metrics['num_contracts']}")
    print(f"    Accuracy:  {syn_contract_metrics['accuracy']:.4f}")
    print(f"    Precision: {syn_contract_metrics['precision']:.4f}")
    print(f"    Recall:    {syn_contract_metrics['recall']:.4f}")
    print(f"    F1 Score:  {syn_contract_metrics['f1']:.4f}")
    print(f"    AUC:       {syn_contract_metrics['auc']:.4f}")
    
    if irl_metrics is not None:
        result['realworld_test'] = {
            'function_level': {
                'accuracy': round(irl_metrics['accuracy'], 4),
                'precision': round(irl_metrics['precision'], 4),
                'recall': round(irl_metrics['recall'], 4),
                'f1': round(irl_metrics['f1'], 4),
                'auc': round(irl_metrics['auc'], 4),
                'num_functions': irl_metrics['num_functions'],
            },
            'contract_level': {
                'accuracy': round(irl_contract_metrics['accuracy'], 4),
                'precision': round(irl_contract_metrics['precision'], 4),
                'recall': round(irl_contract_metrics['recall'], 4),
                'f1': round(irl_contract_metrics['f1'], 4),
                'auc': round(irl_contract_metrics['auc'], 4),
                'num_contracts': irl_contract_metrics['num_contracts'],
            },
        }
        print(f"  [Real-World Test - Function Level]")
        print(f"    Functions: {irl_metrics['num_functions']}")
        print(f"    Accuracy:  {irl_metrics['accuracy']:.4f}")
        print(f"    Precision: {irl_metrics['precision']:.4f}")
        print(f"    Recall:    {irl_metrics['recall']:.4f}")
        print(f"    F1 Score:  {irl_metrics['f1']:.4f}")
        print(f"    AUC:       {irl_metrics['auc']:.4f}")
        
        print(f"  [Real-World Test - Contract Level]")
        print(f"    Contracts: {irl_contract_metrics['num_contracts']}")
        print(f"    Accuracy:  {irl_contract_metrics['accuracy']:.4f}")
        print(f"    Precision: {irl_contract_metrics['precision']:.4f}")
        print(f"    Recall:    {irl_contract_metrics['recall']:.4f}")
        print(f"    F1 Score:  {irl_contract_metrics['f1']:.4f}")
        print(f"    AUC:       {irl_contract_metrics['auc']:.4f}")
    
    print(f"  Train time: {train_time:.1f}s")
    
    return result


def run_benchmark(
    synthetic_data_path: str,
    realworld_data_path: Optional[str],
    models: List[str],
    hidden_dim: int = 128,
    num_layers: int = 3,
    heads: int = 4,
    dropout: float = 0.3,
    epochs: int = 50,
    lr: float = 0.001,
    patience: int = 20,
    batch_size: int = 16,  # Smaller batch for function-level
    seed: int = 42,
    synthetic_only: bool = False,
    use_presplit: bool = True,
) -> Dict[str, Any]:
    """
    Run function-level benchmark on all specified models.
    
    Supports two modes:
    1. Full mode (default): Train on synthetic, test on both synthetic and realworld
    2. Synthetic-only mode: Train and test only on synthetic data (legacy mode)
    """
    setup_seed(seed)
    device = setup_device()
    
    print("="*70)
    print("FUNCTION-LEVEL VULNERABILITY DETECTION BENCHMARK")
    print("="*70)
    print(f"\nConfiguration:")
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
    
    # Check if using pre-split data
    base_path = Path(synthetic_data_path)
    has_presplit = (base_path / 'train').exists() and (base_path / 'dev').exists() and (base_path / 'test').exists()
    
    if use_presplit and has_presplit:
        print(f"\n[Synthetic Data] Loading pre-split data from {synthetic_data_path}...")
        train_dataset = FunctionLevelDataset(root=str(base_path / 'train'), bidirectional=True)
        val_dataset = FunctionLevelDataset(root=str(base_path / 'dev'), bidirectional=True)
        syn_test_dataset = FunctionLevelDataset(root=str(base_path / 'test'), bidirectional=True)
        
        # Get combined stats
        train_stats = train_dataset.get_stats()
        val_stats = val_dataset.get_stats()
        test_stats = syn_test_dataset.get_stats()
        
        print(f"  Train: {train_stats.num_contracts} contracts, {train_stats.num_functions} functions")
        print(f"  Dev: {val_stats.num_contracts} contracts, {val_stats.num_functions} functions")
        print(f"  Test: {test_stats.num_contracts} contracts, {test_stats.num_functions} functions")
        print(f"  Total: {train_stats.num_contracts + val_stats.num_contracts + test_stats.num_contracts} contracts")
        
        # Use train stats for class weights
        syn_stats = train_stats
    else:
        if use_presplit and not has_presplit:
            print(f"\nWarning: Pre-split directories not found in {synthetic_data_path}")
            print("Falling back to random split (legacy mode)")
        
        print(f"\n[Synthetic Data] Loading from {synthetic_data_path}...")
        synthetic_dataset = FunctionLevelDataset(root=synthetic_data_path, bidirectional=True)
        syn_stats = synthetic_dataset.get_stats()
        
        print(f"  Contracts: {syn_stats.num_contracts}")
        print(f"  Functions: {syn_stats.num_functions}")
        print(f"    Vulnerable: {syn_stats.num_vulnerable_functions}")
        print(f"    Clean: {syn_stats.num_clean_functions}")
        print(f"  Avg functions/contract: {syn_stats.avg_functions_per_contract:.2f}")
    
    # Load realworld dataset if provided and not in synthetic-only mode
    realworld_dataset = None
    irl_stats = None
    if realworld_data_path and not synthetic_only:
        print(f"\n[Real-World Data] Loading from {realworld_data_path}...")
        realworld_dataset = FunctionLevelDataset(root=realworld_data_path, bidirectional=True)
        irl_stats = realworld_dataset.get_stats()
        
        print(f"  Contracts: {irl_stats.num_contracts}")
        print(f"  Functions: {irl_stats.num_functions}")
        print(f"    Vulnerable: {irl_stats.num_vulnerable_functions}")
        print(f"    Clean: {irl_stats.num_clean_functions}")
        print(f"  Avg functions/contract: {irl_stats.avg_functions_per_contract:.2f}")
    
    # Create data loaders
    if use_presplit and has_presplit:
        # Pre-split mode: use template-based splits
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        syn_test_loader = DataLoader(syn_test_dataset, batch_size=batch_size, shuffle=False)
        
        # Load realworld data if available
        irl_test_loader = None
        if realworld_data_path and not synthetic_only:
            realworld_dataset = FunctionLevelDataset(root=realworld_data_path, bidirectional=True)
            irl_test_loader = DataLoader(realworld_dataset, batch_size=batch_size, shuffle=False)
        
        print(f"\n[Data Loaders] Pre-split (template-based):")
        print(f"  Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(syn_test_loader.dataset)}")
        if irl_test_loader:
            print(f"  Real-world IRL Test: {len(irl_test_loader.dataset)}")
    elif synthetic_only or realworld_dataset is None:
        # Legacy mode: split synthetic data into train/val/test
        train_loader, val_loader, syn_test_loader = create_function_level_loaders(
            synthetic_dataset, batch_size=batch_size, seed=seed
        )
        irl_test_loader = None
        print(f"\n[Data Split] Synthetic only (random split):")
        print(f"  Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(syn_test_loader.dataset)}")
    else:
        # Full mode: use synthetic for train/val/syn_test, realworld for irl_test
        train_loader, val_loader, syn_test_loader, irl_test_loader = create_function_level_loaders_with_irl_test(
            synthetic_dataset, realworld_dataset, batch_size=batch_size, seed=seed
        )
        print(f"\n[Data Split] Random split (legacy):")
        print(f"  Synthetic - Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(syn_test_loader.dataset)}")
        print(f"  Real-world - IRL Test: {len(irl_test_loader.dataset)}")
    
    # Compute class weights from training data (synthetic)
    class_weights = compute_function_weights(
        syn_stats.num_clean_functions,
        syn_stats.num_vulnerable_functions
    )
    print(f"\n  Class weights: clean={class_weights[0]:.2f}, vuln={class_weights[1]:.2f}")
    
    input_dim = get_input_dim()
    num_edge_types = get_num_edge_types()
    
    # Run benchmarks
    results = []
    total_start = time.time()
    
    for model_key in models:
        if model_key not in MODEL_CONFIGS:
            print(f"\nWarning: Unknown model '{model_key}', skipping...")
            continue
        
        result = benchmark_model(
            model_key=model_key,
            train_loader=train_loader,
            val_loader=val_loader,
            syn_test_loader=syn_test_loader,
            irl_test_loader=irl_test_loader,
            device=device,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            heads=heads,
            dropout=dropout,
            epochs=epochs,
            lr=lr,
            patience=patience,
            num_edge_types=num_edge_types,
            class_weights=class_weights,
            seed=seed,
        )
        results.append(result)
    
    total_time = time.time() - total_start
    
    output = {
        'timestamp': datetime.now().isoformat(),
        'config': {
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
            'num_contracts': syn_stats.num_contracts,
            'num_functions': syn_stats.num_functions,
            'num_vulnerable_functions': syn_stats.num_vulnerable_functions,
            'num_clean_functions': syn_stats.num_clean_functions,
            'num_edges': syn_stats.num_edges,
            'num_intra_edges': syn_stats.num_intra_edges,
            'num_inter_edges': syn_stats.num_inter_edges,
        },
        'results': results,
        'total_time_sec': round(total_time, 2),
    }
    
    if irl_stats:
        output['realworld_stats'] = {
            'num_contracts': irl_stats.num_contracts,
            'num_functions': irl_stats.num_functions,
            'num_vulnerable_functions': irl_stats.num_vulnerable_functions,
            'num_clean_functions': irl_stats.num_clean_functions,
            'num_edges': irl_stats.num_edges,
            'num_intra_edges': irl_stats.num_intra_edges,
            'num_inter_edges': irl_stats.num_inter_edges,
        }
    
    return output


def print_results_table(results: Dict[str, Any]):
    """Print results as a formatted table with both function-level and contract-level results."""
    print("\n")
    print("="*130)
    print("FUNCTION-LEVEL BENCHMARK RESULTS")
    print("="*130)
    
    has_irl = any('realworld_test' in r for r in results['results'] if 'error' not in r)
    
    if has_irl:
        # ========== FUNCTION-LEVEL RESULTS ==========
        print("\n📊 FUNCTION-LEVEL: SYNTHETIC TEST RESULTS")
        print("-"*110)
        header = f"{'Model':<35} {'Paper':<12} {'Pool':<6} {'Params':>10} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'AUC':>7}"
        print(header)
        print("-"*110)
        
        sorted_results = sorted(
            [r for r in results['results'] if 'error' not in r],
            key=lambda x: x['synthetic_test']['function_level']['f1'],
            reverse=True
        )
        
        for r in sorted_results:
            m = r['synthetic_test']['function_level']
            paper = r.get('paper', 'N/A')[:12]
            row = f"{r['model']:<35} {paper:<12} {r['pooling']:<6} {r['parameters']:>10,} {m['accuracy']:>7.4f} {m['precision']:>7.4f} {m['recall']:>7.4f} {m['f1']:>7.4f} {m['auc']:>7.4f}"
            print(row)
        
        print("\n🌍 FUNCTION-LEVEL: REAL-WORLD (IRL) TEST RESULTS")
        print("-"*110)
        print(header)
        print("-"*110)
        
        sorted_irl = sorted(
            [r for r in results['results'] if 'error' not in r and 'realworld_test' in r],
            key=lambda x: x['realworld_test']['function_level']['f1'],
            reverse=True
        )
        
        for r in sorted_irl:
            m = r['realworld_test']['function_level']
            paper = r.get('paper', 'N/A')[:12]
            row = f"{r['model']:<35} {paper:<12} {r['pooling']:<6} {r['parameters']:>10,} {m['accuracy']:>7.4f} {m['precision']:>7.4f} {m['recall']:>7.4f} {m['f1']:>7.4f} {m['auc']:>7.4f}"
            print(row)
        
        # ========== CONTRACT-LEVEL RESULTS ==========
        print("\n\n📊 CONTRACT-LEVEL: SYNTHETIC TEST RESULTS (Aggregated from Functions)")
        print("-"*110)
        header = f"{'Model':<35} {'Paper':<12} {'Pool':<6} {'Contracts':>10} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'AUC':>7}"
        print(header)
        print("-"*110)
        
        sorted_contract = sorted(
            [r for r in results['results'] if 'error' not in r],
            key=lambda x: x['synthetic_test']['contract_level']['f1'],
            reverse=True
        )
        
        for r in sorted_contract:
            m = r['synthetic_test']['contract_level']
            paper = r.get('paper', 'N/A')[:12]
            row = f"{r['model']:<35} {paper:<12} {r['pooling']:<6} {m['num_contracts']:>10} {m['accuracy']:>7.4f} {m['precision']:>7.4f} {m['recall']:>7.4f} {m['f1']:>7.4f} {m['auc']:>7.4f}"
            print(row)
        
        print("\n🌍 CONTRACT-LEVEL: REAL-WORLD (IRL) TEST RESULTS (Aggregated from Functions)")
        print("-"*110)
        print(header)
        print("-"*110)
        
        sorted_irl_contract = sorted(
            [r for r in results['results'] if 'error' not in r and 'realworld_test' in r],
            key=lambda x: x['realworld_test']['contract_level']['f1'],
            reverse=True
        )
        
        for r in sorted_irl_contract:
            m = r['realworld_test']['contract_level']
            paper = r.get('paper', 'N/A')[:12]
            row = f"{r['model']:<35} {paper:<12} {r['pooling']:<6} {m['num_contracts']:>10} {m['accuracy']:>7.4f} {m['precision']:>7.4f} {m['recall']:>7.4f} {m['f1']:>7.4f} {m['auc']:>7.4f}"
            print(row)
        
        # Summary comparison
        print("\n📈 COMPARISON SUMMARY (Sorted by IRL F1)")
        print("-"*120)
        print(f"{'Model':<35} {'Level':<10} {'Syn F1':>10} {'IRL F1':>10} {'Gap':>10} {'Syn AUC':>10} {'IRL AUC':>10}")
        print("-"*120)
        
        for r in sorted_irl:
            # Function-level
            syn_f1 = r['synthetic_test']['function_level']['f1']
            irl_f1 = r['realworld_test']['function_level']['f1']
            gap = irl_f1 - syn_f1
            syn_auc = r['synthetic_test']['function_level']['auc']
            irl_auc = r['realworld_test']['function_level']['auc']
            gap_str = f"{gap:+.4f}"
            print(f"{r['model']:<35} {'Function':<10} {syn_f1:>10.4f} {irl_f1:>10.4f} {gap_str:>10} {syn_auc:>10.4f} {irl_auc:>10.4f}")
        
        for r in sorted_irl_contract:
            # Contract-level
            syn_f1 = r['synthetic_test']['contract_level']['f1']
            irl_f1 = r['realworld_test']['contract_level']['f1']
            gap = irl_f1 - syn_f1
            syn_auc = r['synthetic_test']['contract_level']['auc']
            irl_auc = r['realworld_test']['contract_level']['auc']
            gap_str = f"{gap:+.4f}"
            print(f"{r['model']:<35} {'Contract':<10} {syn_f1:>10.4f} {irl_f1:>10.4f} {gap_str:>10} {syn_auc:>10.4f} {irl_auc:>10.4f}")
        
        if sorted_irl:
            best_syn_func = sorted_results[0]
            best_irl_func = sorted_irl[0]
            best_syn_contract = sorted_contract[0]
            best_irl_contract = sorted_irl_contract[0]
            print(f"\n🏆 Best by Synthetic F1 (Function): {best_syn_func['model']} (F1={best_syn_func['synthetic_test']['function_level']['f1']:.4f})")
            print(f"🏆 Best by Real-World F1 (Function): {best_irl_func['model']} (F1={best_irl_func['realworld_test']['function_level']['f1']:.4f})")
            print(f"🏆 Best by Synthetic F1 (Contract): {best_syn_contract['model']} (F1={best_syn_contract['synthetic_test']['contract_level']['f1']:.4f})")
            print(f"🏆 Best by Real-World F1 (Contract): {best_irl_contract['model']} (F1={best_irl_contract['realworld_test']['contract_level']['f1']:.4f})")
    else:
        # Single table for synthetic-only mode
        header = f"{'Model':<35} {'Paper':<15} {'Pool':<8} {'Params':>10} {'Acc':>8} {'Prec':>8} {'Rec':>8} {'F1':>8} {'AUC':>8}"
        print(header)
        print("-"*110)
        
        sorted_results = sorted(
            [r for r in results['results'] if 'error' not in r],
            key=lambda x: x['synthetic_test']['function_level']['f1'],
            reverse=True
        )
        
        for r in sorted_results:
            m = r['synthetic_test']['function_level']
            paper = r.get('paper', 'N/A')[:15]
            row = f"{r['model']:<35} {paper:<15} {r['pooling']:<8} {r['parameters']:>10,} {m['accuracy']:>8.4f} {m['precision']:>8.4f} {m['recall']:>8.4f} {m['f1']:>8.4f} {m['auc']:>8.4f}"
            print(row)
        
        if sorted_results:
            best = sorted_results[0]
            print(f"\n🏆 Best model by F1: {best['model']} (F1={best['synthetic_test']['function_level']['f1']:.4f})")
    
    # Print errors if any
    errors = [r for r in results['results'] if 'error' in r]
    if errors:
        print("\n❌ Failed models:")
        for r in errors:
            print(f"  {r.get('model', 'Unknown')}: {r['error']}")
    
    print("-"*130)
    print(f"Total benchmark time: {results['total_time_sec']:.1f}s")


def main():
    parser = argparse.ArgumentParser(
        description='Function-Level Vulnerability Detection Benchmark',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run benchmark with both synthetic and realworld test (default)
    python function_level_benchmark.py --epochs 50
    
    # Run specific models
    python function_level_benchmark.py --models func-hca-mean func-mlagnn --epochs 50
    
    # Specify custom data paths
    python function_level_benchmark.py --synthetic-data data/synthetic/function_level_dataset \\
                                       --realworld-data data/realworld/function_level_dataset
    
    # Legacy mode: synthetic data only
    python function_level_benchmark.py --synthetic-only --synthetic-data data/function_level_dataset
    
    # Quick test
    python function_level_benchmark.py --epochs 10 --models func-sage-mean func-gat-mean
        """
    )
    
    # Data paths
    parser.add_argument('--synthetic-data', '-s', default='data/synthetic-split',
                        help='Synthetic (injected) function-level dataset directory')
    parser.add_argument('--realworld-data', '-r', default='data/realworld/function_level_dataset',
                        help='Real-world function-level dataset directory')
    parser.add_argument('--synthetic-only', action='store_true',
                        help='Use only synthetic data (legacy mode)')
    
    # Legacy argument for backward compatibility
    parser.add_argument('--data', '-d', default=None,
                        help='[DEPRECATED] Use --synthetic-data instead')
    
    # Model selection
    parser.add_argument('--models', '-m', nargs='+', default=list(MODEL_CONFIGS.keys()),
                        choices=list(MODEL_CONFIGS.keys()),
                        help='Models to benchmark')
    
    # Hyperparameters
    parser.add_argument('--hidden-dim', type=int, default=128,
                        help='Hidden dimension')
    parser.add_argument('--num-layers', type=int, default=3,
                        help='Number of GNN layers')
    parser.add_argument('--heads', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout rate')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Maximum training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--patience', type=int, default=20,
                        help='Early stopping patience')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--output', '-o', type=str, default='results/benchmark/function_level.json',
                        help='Output JSON file for results')
    parser.add_argument('--use-presplit', action='store_true', default=True,
                        help='Use pre-split data if available (default: True)')
    parser.add_argument('--random-split', action='store_true',
                        help='Force random split even if pre-split data exists')
    
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
        print("Please build the function-level dataset first:")
        print(f"  python function_level_builder.py --input data/synthetic/ast_dataset --output {synthetic_data_path}")
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
            print(f"  python dataset_builder.py --input data/realworld/sc-source --output data/realworld/ast_dataset")
            print(f"  python function_level_builder.py --input data/realworld/ast_dataset --output {args.realworld_data}")
    
    # Run benchmark
    results = run_benchmark(
        synthetic_data_path=synthetic_data_path,
        realworld_data_path=realworld_data_path,
        models=args.models,
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
        use_presplit=args.use_presplit,
    )
    
    # Print results
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
