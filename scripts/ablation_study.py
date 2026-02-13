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
from models.function_level_data import FunctionLevelDataset, create_function_level_loaders
from models.subgraph_level.hierarchical_cross_attention import create_hca_sgnn_model

from training.function_level_utils import (
    train_function_level_with_early_stopping,
    evaluate_function_level,
    evaluate_contract_level,
    compute_function_weights,
)
from training.utils import setup_seed, setup_device


# ============================================================================
# Ablation Configurations
# ============================================================================

ABLATION_CONFIGS = {
    # Full model (default configuration)
    'full': {
        'name': 'HCA-SGNN (Full)',
        'description': 'Complete model with all components enabled',
        'use_rgcn': False,
        'use_dual_channel': True,
        'use_gated_fusion': True,
        'use_node_attention': False,
        'use_cross_attention': True,
        'pooling': 'mean',
    },
    
    # ========== Edge Handling Ablations ==========
    'rgcn': {
        'name': 'HCA-SGNN (RGCN)',
        'description': 'Use RGCNConv instead of EdgeTypeAwareConv',
        'use_rgcn': True,
        'use_dual_channel': True,
        'use_gated_fusion': True,
        'use_node_attention': False,
        'use_cross_attention': True,
        'pooling': 'mean',
    },
    
    # ========== Dual-Channel Ablations ==========
    'no_dual_channel': {
        'name': 'HCA-SGNN (No Dual-Channel)',
        'description': 'Single-channel encoder (only full graph, no subgraph)',
        'use_rgcn': False,
        'use_dual_channel': False,
        'use_gated_fusion': True,  # Ignored when dual_channel=False
        'use_node_attention': False,
        'use_cross_attention': True,
        'pooling': 'mean',
    },
    
    # ========== Attention Ablations ==========
    'no_cross_attention': {
        'name': 'HCA-SGNN (No Cross-Attention)',
        'description': 'Disable inter-function cross-attention',
        'use_rgcn': False,
        'use_dual_channel': True,
        'use_gated_fusion': True,
        'use_node_attention': False,
        'use_cross_attention': False,
        'pooling': 'mean',
    },
    
    # ========== Pooling Ablations ==========
    'hierarchical_pooling': {
        'name': 'HCA-SGNN (Hierarchical Pool)',
        'description': 'Use hierarchical multi-scale pooling',
        'use_rgcn': False,
        'use_dual_channel': True,
        'use_gated_fusion': True,
        'use_node_attention': False,
        'use_cross_attention': True,
        'pooling': 'hierarchical',
    },
    'max_pooling': {
        'name': 'HCA-SGNN (Max Pool)',
        'description': 'Use max pooling instead of mean',
        'use_rgcn': False,
        'use_dual_channel': True,
        'use_gated_fusion': True,
        'use_node_attention': False,
        'use_cross_attention': True,
        'pooling': 'max',
    },
    'attention_pooling': {
        'name': 'HCA-SGNN (Attention Pool)',
        'description': 'Use attention-based pooling',
        'use_rgcn': False,
        'use_dual_channel': True,
        'use_gated_fusion': True,
        'use_node_attention': False,
        'use_cross_attention': True,
        'pooling': 'attention',
    },
    
}

# Quick ablation configs (most important comparisons)
QUICK_CONFIGS = [
    'full',
    'rgcn',
    'no_dual_channel',
    'no_cross_attention',
]


def create_ablation_model(
    config: Dict[str, Any],
    input_dim: int,
    hidden_dim: int,
    num_layers: int,
    num_edge_types: int,
    heads: int,
    dropout: float,
    device: torch.device,
) -> nn.Module:
    """Create an HCA-SGNN model with specific ablation configuration."""
    model = create_hca_sgnn_model(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_edge_types=num_edge_types,
        heads=heads,
        num_classes=2,
        dropout=dropout,
        use_cross_attention=config['use_cross_attention'],
        pooling=config['pooling'],
        use_rgcn=config['use_rgcn'],
        use_dual_channel=config['use_dual_channel'],
        use_gated_fusion=config['use_gated_fusion'],
        use_node_attention=config['use_node_attention'],
    )
    return model.to(device)


def run_single_ablation(
    config_key: str,
    config: Dict[str, Any],
    train_loader: DataLoader,
    val_loader: DataLoader,
    syn_test_loader: DataLoader,
    irl_test_loader: Optional[DataLoader],
    device: torch.device,
    input_dim: int,
    hidden_dim: int,
    num_layers: int,
    num_edge_types: int,
    heads: int,
    dropout: float,
    epochs: int,
    lr: float,
    patience: int,
    class_weights: Optional[torch.Tensor],
    seed: int,
    output_dir: Path,
) -> Dict[str, Any]:
    """Run a single ablation configuration and return results."""
    print(f"\n{'='*70}")
    print(f"Ablation: {config['name']}")
    print(f"Description: {config['description']}")
    print(f"{'='*70}")
    
    setup_seed(seed)
    
    # Create model
    try:
        model = create_ablation_model(
            config=config,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_edge_types=num_edge_types,
            heads=heads,
            dropout=dropout,
            device=device,
        )
    except Exception as e:
        print(f"  Error creating model: {e}")
        import traceback
        traceback.print_exc()
        return {
            'config': config_key,
            'name': config['name'],
            'error': str(e),
        }
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {num_params:,}")
    print(f"  Config: RGCN={config['use_rgcn']}, DualChannel={config['use_dual_channel']}, "
          f"GatedFusion={config['use_gated_fusion']}, NodeAttn={config['use_node_attention']}, "
          f"CrossAttn={config['use_cross_attention']}, Pooling={config['pooling']}")
    
    # Create checkpoint directory
    checkpoint_dir = output_dir / config_key
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
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
            output_dir=checkpoint_dir,
            class_weights=class_weights,
            verbose=False,
        )
    except Exception as e:
        print(f"  Error during training: {e}")
        import traceback
        traceback.print_exc()
        return {
            'config': config_key,
            'name': config['name'],
            'error': str(e),
        }
    
    train_time = time.time() - start_time
    
    # Evaluate
    criterion = nn.CrossEntropyLoss()
    syn_metrics = evaluate_function_level(model, syn_test_loader, criterion, device)
    syn_contract_metrics = evaluate_contract_level(model, syn_test_loader, device)
    
    result = {
        'config': config_key,
        'name': config['name'],
        'description': config['description'],
        'ablation_flags': {
            'use_rgcn': config['use_rgcn'],
            'use_dual_channel': config['use_dual_channel'],
            'use_gated_fusion': config['use_gated_fusion'],
            'use_node_attention': config['use_node_attention'],
            'use_cross_attention': config['use_cross_attention'],
            'pooling': config['pooling'],
        },
        'parameters': num_params,
        'train_time_sec': round(train_time, 2),
        'best_epoch': history.get('best_epoch', epochs),
        'synthetic_test': {
            'function_level': {
                'accuracy': round(syn_metrics['accuracy'], 4),
                'precision': round(syn_metrics['precision'], 4),
                'recall': round(syn_metrics['recall'], 4),
                'f1': round(syn_metrics['f1'], 4),
                'auc': round(syn_metrics['auc'], 4),
            },
            'contract_level': {
                'accuracy': round(syn_contract_metrics['accuracy'], 4),
                'precision': round(syn_contract_metrics['precision'], 4),
                'recall': round(syn_contract_metrics['recall'], 4),
                'f1': round(syn_contract_metrics['f1'], 4),
                'auc': round(syn_contract_metrics['auc'], 4),
            },
        },
    }
    
    # Print synthetic results
    print(f"  [Synthetic Test - Function Level]")
    print(f"    F1: {syn_metrics['f1']:.4f}, AUC: {syn_metrics['auc']:.4f}, "
          f"Prec: {syn_metrics['precision']:.4f}, Rec: {syn_metrics['recall']:.4f}")
    print(f"  [Synthetic Test - Contract Level]")
    print(f"    F1: {syn_contract_metrics['f1']:.4f}, AUC: {syn_contract_metrics['auc']:.4f}")
    
    # Evaluate on real-world if available
    if irl_test_loader is not None:
        irl_metrics = evaluate_function_level(model, irl_test_loader, criterion, device)
        irl_contract_metrics = evaluate_contract_level(model, irl_test_loader, device)
        
        result['realworld_test'] = {
            'function_level': {
                'accuracy': round(irl_metrics['accuracy'], 4),
                'precision': round(irl_metrics['precision'], 4),
                'recall': round(irl_metrics['recall'], 4),
                'f1': round(irl_metrics['f1'], 4),
                'auc': round(irl_metrics['auc'], 4),
            },
            'contract_level': {
                'accuracy': round(irl_contract_metrics['accuracy'], 4),
                'precision': round(irl_contract_metrics['precision'], 4),
                'recall': round(irl_contract_metrics['recall'], 4),
                'f1': round(irl_contract_metrics['f1'], 4),
                'auc': round(irl_contract_metrics['auc'], 4),
            },
        }
        
        print(f"  [Real-World Test - Function Level]")
        print(f"    F1: {irl_metrics['f1']:.4f}, AUC: {irl_metrics['auc']:.4f}, "
              f"Prec: {irl_metrics['precision']:.4f}, Rec: {irl_metrics['recall']:.4f}")
        print(f"  [Real-World Test - Contract Level]")
        print(f"    F1: {irl_contract_metrics['f1']:.4f}, AUC: {irl_contract_metrics['auc']:.4f}")
    
    print(f"  Train time: {train_time:.1f}s, Best epoch: {result['best_epoch']}")
    
    return result


def run_ablation_study(
    synthetic_data_path: str,
    realworld_data_path: Optional[str],
    configs: List[str],
    hidden_dim: int = 256,
    num_layers: int = 4,
    heads: int = 4,
    dropout: float = 0.2,
    epochs: int = 50,
    lr: float = 0.001,
    patience: int = 20,
    batch_size: int = 16,
    seed: int = 42,
    num_runs: int = 1,
    output_path: str = 'results/ablation_study.json',
) -> Dict[str, Any]:
    """
    Run ablation study on HCA-SGNN.
    
    Args:
        synthetic_data_path: Path to synthetic function-level dataset
        realworld_data_path: Path to realworld function-level dataset (optional)
        configs: List of ablation configurations to run
        hidden_dim: Hidden dimension
        num_layers: Number of GNN layers
        heads: Number of attention heads
        dropout: Dropout rate
        epochs: Maximum training epochs
        lr: Learning rate
        patience: Early stopping patience
        batch_size: Batch size
        seed: Random seed
        num_runs: Number of runs for statistical significance
        output_path: Path to save results JSON
    """
    device = setup_device()
    
    print("="*70)
    print("HCA-SGNN ABLATION STUDY")
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
    print(f"  Base seed: {seed}")
    print(f"  Num runs: {num_runs}")
    print(f"  Configs: {configs}")
    
    # Load datasets
    base_path = Path(synthetic_data_path)
    has_presplit = (base_path / 'train').exists() and (base_path / 'dev').exists()
    
    if has_presplit:
        # Pre-split mode: use template-based splits
        print(f"\n[Synthetic Data] Loading pre-split from {synthetic_data_path}...")
        train_dataset = FunctionLevelDataset(root=str(base_path / 'train'), bidirectional=True)
        val_dataset = FunctionLevelDataset(root=str(base_path / 'dev'), bidirectional=True)
        syn_test_dataset = FunctionLevelDataset(root=str(base_path / 'test'), bidirectional=True)
        
        print(f"  Train: {len(train_dataset)} contracts")
        print(f"  Dev: {len(val_dataset)} contracts")
        print(f"  Test: {len(syn_test_dataset)} contracts")
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        syn_test_loader = DataLoader(syn_test_dataset, batch_size=batch_size, shuffle=False)
        
        # Compute class weights from train stats
        train_stats = train_dataset.get_stats()
    else:
        # Random split mode (legacy)
        print(f"\n[Synthetic Data] Loading from {synthetic_data_path} (random split)...")
        synthetic_dataset = FunctionLevelDataset(root=synthetic_data_path, bidirectional=True)
        syn_stats = synthetic_dataset.get_stats()
        
        print(f"  Contracts: {syn_stats.num_contracts}")
        print(f"  Functions: {syn_stats.num_functions}")
        print(f"    Vulnerable: {syn_stats.num_vulnerable_functions}")
        print(f"    Clean: {syn_stats.num_clean_functions}")
        
        # Create data loaders with random split
        train_loader, val_loader, syn_test_loader = create_function_level_loaders(
            synthetic_dataset, batch_size=batch_size, seed=seed
        )
        
        print(f"  Split: Train={len(train_loader.dataset)}, Val={len(val_loader.dataset)}, Test={len(syn_test_loader.dataset)}")
        
        # Use full dataset stats for class weights
        train_stats = syn_stats
    
    # Load realworld dataset
    irl_test_dataset = None
    irl_test_loader = None
    if realworld_data_path:
        realworld_path = Path(realworld_data_path)
        if realworld_path.exists():
            print(f"\n[Real-World Data] Loading from {realworld_data_path}...")
            irl_test_dataset = FunctionLevelDataset(root=realworld_data_path, bidirectional=True)
            print(f"  Contracts: {len(irl_test_dataset)}")
            irl_test_loader = DataLoader(irl_test_dataset, batch_size=batch_size, shuffle=False)
        else:
            print(f"\n[Real-World Data] Path not found: {realworld_data_path}, skipping...")
    
    # Get dimensions
    input_dim = get_input_dim()
    num_edge_types = get_num_edge_types()
    
    # Compute class weights
    class_weights = compute_function_weights(
        train_stats.num_clean_functions,
        train_stats.num_vulnerable_functions,
    ).to(device)
    print(f"\n  Class weights: {class_weights.tolist()}")
    
    # Output directory
    output_dir = Path('outputs/ablation_study')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run ablation study
    all_results = []
    
    for config_key in configs:
        if config_key not in ABLATION_CONFIGS:
            print(f"\nWarning: Unknown config '{config_key}', skipping...")
            continue
        
        config = ABLATION_CONFIGS[config_key]
        run_results = []
        
        for run_idx in range(num_runs):
            run_seed = seed + run_idx
            
            if num_runs > 1:
                print(f"\n--- Run {run_idx + 1}/{num_runs} (seed={run_seed}) ---")
            
            result = run_single_ablation(
                config_key=config_key,
                config=config,
                train_loader=train_loader,
                val_loader=val_loader,
                syn_test_loader=syn_test_loader,
                irl_test_loader=irl_test_loader,
                device=device,
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                num_edge_types=num_edge_types,
                heads=heads,
                dropout=dropout,
                epochs=epochs,
                lr=lr,
                patience=patience,
                class_weights=class_weights,
                seed=run_seed,
                output_dir=output_dir / f"run_{run_idx}",
            )
            result['seed'] = run_seed
            result['run'] = run_idx
            run_results.append(result)
        
        # Aggregate results across runs
        if num_runs > 1 and all('error' not in r for r in run_results):
            aggregated = aggregate_results(run_results)
            aggregated['individual_runs'] = run_results
            all_results.append(aggregated)
        else:
            all_results.extend(run_results)
    
    # Create summary
    summary = create_summary(all_results)
    
    # Final results
    final_results = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'config': {
            'hidden_dim': hidden_dim,
            'num_layers': num_layers,
            'heads': heads,
            'dropout': dropout,
            'epochs': epochs,
            'lr': lr,
            'patience': patience,
            'batch_size': batch_size,
            'num_runs': num_runs,
        },
        'data': {
            'synthetic_path': synthetic_data_path,
            'realworld_path': realworld_data_path,
            'train_size': len(train_loader.dataset),
            'val_size': len(val_loader.dataset),
            'test_size': len(syn_test_loader.dataset),
            'irl_test_size': len(irl_test_loader.dataset) if irl_test_loader else 0,
            'split_mode': 'pre-split' if has_presplit else 'random',
        },
        'summary': summary,
        'results': all_results,
    }
    
    # Save results
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(final_results, f, indent=2)
    print(f"\nResults saved to {output_path}")
    
    # Print summary
    print_summary(summary)
    
    return final_results


def aggregate_results(run_results: List[Dict]) -> Dict:
    """Aggregate results from multiple runs (mean ± std)."""
    import numpy as np
    
    config = run_results[0]
    aggregated = {
        'config': config['config'],
        'name': config['name'],
        'description': config['description'],
        'ablation_flags': config['ablation_flags'],
        'parameters': config['parameters'],
        'num_runs': len(run_results),
    }
    
    # Aggregate metrics
    for test_set in ['synthetic_test', 'realworld_test']:
        if test_set not in run_results[0]:
            continue
        
        aggregated[test_set] = {}
        for level in ['function_level', 'contract_level']:
            metrics = {}
            for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
                values = [r[test_set][level][metric] for r in run_results]
                metrics[metric] = {
                    'mean': round(np.mean(values), 4),
                    'std': round(np.std(values), 4),
                }
            aggregated[test_set][level] = metrics
    
    # Aggregate train time
    train_times = [r['train_time_sec'] for r in run_results]
    aggregated['train_time_sec'] = {
        'mean': round(np.mean(train_times), 2),
        'std': round(np.std(train_times), 2),
    }
    
    return aggregated


def create_summary(results: List[Dict]) -> Dict:
    """Create summary table of ablation results."""
    summary = {
        'by_f1': [],
        'component_analysis': {},
    }
    
    # Sort by F1 score
    valid_results = [r for r in results if 'error' not in r]
    
    for result in valid_results:
        syn_f1 = result['synthetic_test']['function_level']
        if isinstance(syn_f1, dict) and 'f1' in syn_f1:
            # Single run
            f1 = syn_f1['f1']
        elif isinstance(syn_f1, dict) and 'mean' in syn_f1.get('f1', {}):
            # Aggregated runs
            f1 = syn_f1['f1']['mean']
        else:
            continue
        
        entry = {
            'config': result['config'],
            'name': result['name'],
            'f1': f1,
            'parameters': result['parameters'],
        }
        
        if 'realworld_test' in result:
            irl_f1 = result['realworld_test']['function_level']
            if isinstance(irl_f1, dict) and 'f1' in irl_f1:
                entry['irl_f1'] = irl_f1['f1']
            elif isinstance(irl_f1, dict) and 'mean' in irl_f1.get('f1', {}):
                entry['irl_f1'] = irl_f1['f1']['mean']
        
        summary['by_f1'].append(entry)
    
    # Sort by F1 descending
    summary['by_f1'].sort(key=lambda x: x['f1'], reverse=True)
    
    # Component analysis (compare full vs ablated)
    full_result = next((r for r in valid_results if r['config'] == 'full'), None)
    if full_result:
        full_f1 = full_result['synthetic_test']['function_level']
        if isinstance(full_f1, dict) and 'f1' in full_f1:
            full_f1_val = full_f1['f1']
        elif isinstance(full_f1, dict) and 'mean' in full_f1.get('f1', {}):
            full_f1_val = full_f1['f1']['mean']
        else:
            full_f1_val = None
        
        if full_f1_val is not None:
            summary['component_analysis']['baseline'] = full_f1_val
            
            for result in valid_results:
                if result['config'] == 'full':
                    continue
                
                syn_f1 = result['synthetic_test']['function_level']
                if isinstance(syn_f1, dict) and 'f1' in syn_f1:
                    f1 = syn_f1['f1']
                elif isinstance(syn_f1, dict) and 'mean' in syn_f1.get('f1', {}):
                    f1 = syn_f1['f1']['mean']
                else:
                    continue
                
                delta = round(f1 - full_f1_val, 4)
                summary['component_analysis'][result['config']] = {
                    'f1': f1,
                    'delta': delta,
                    'impact': 'positive' if delta > 0 else 'negative' if delta < 0 else 'neutral',
                }
    
    return summary


def print_summary(summary: Dict):
    """Print ablation study summary."""
    print("\n" + "="*70)
    print("ABLATION STUDY SUMMARY")
    print("="*70)
    
    print("\n[Ranking by F1 Score]")
    print(f"{'Rank':<5} {'Config':<25} {'F1':<8} {'IRL F1':<8} {'Params':<12}")
    print("-"*60)
    
    for i, entry in enumerate(summary['by_f1'], 1):
        irl_f1 = entry.get('irl_f1', '-')
        if isinstance(irl_f1, float):
            irl_f1 = f"{irl_f1:.4f}"
        print(f"{i:<5} {entry['config']:<25} {entry['f1']:.4f}   {irl_f1:<8} {entry['parameters']:,}")
    
    if 'component_analysis' in summary and summary['component_analysis']:
        print("\n[Component Analysis (vs Full Model)]")
        baseline = summary['component_analysis'].get('baseline')
        if baseline:
            print(f"Baseline (full): F1 = {baseline:.4f}")
            print()
            
            for config, analysis in summary['component_analysis'].items():
                if config == 'baseline':
                    continue
                
                delta_str = f"+{analysis['delta']:.4f}" if analysis['delta'] > 0 else f"{analysis['delta']:.4f}"
                impact = "↑" if analysis['impact'] == 'positive' else "↓" if analysis['impact'] == 'negative' else "="
                print(f"  {config:<25} F1={analysis['f1']:.4f}  ({delta_str}) {impact}")


def main():
    parser = argparse.ArgumentParser(
        description='HCA-SGNN Ablation Study',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Data paths
    parser.add_argument(
        '--synthetic-data', type=str,
        default='data/synthetic-split',
        help='Path to synthetic function-level dataset',
    )
    parser.add_argument(
        '--realworld-data', type=str,
        default='data/realworld/function_level_dataset',
        help='Path to realworld function-level dataset',
    )
    
    # Ablation configurations
    parser.add_argument(
        '--configs', type=str, nargs='+',
        default=None,
        help='Ablation configurations to run (default: all)',
    )
    parser.add_argument(
        '--quick', action='store_true',
        help='Run quick ablation (subset of configs)',
    )
    
    # Model hyperparameters
    parser.add_argument('--hidden-dim', type=int, default=128, help='Hidden dimension')
    parser.add_argument('--num-layers', type=int, default=4, help='Number of GNN layers')
    parser.add_argument('--heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    
    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=20, help='Maximum training epochs')
    parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--runs', type=int, default=1, help='Number of runs for statistical significance')
    
    # Output
    parser.add_argument(
        '--output', type=str,
        default='results/ablation_study_hca.json',
        help='Output path for results JSON',
    )
    
    args = parser.parse_args()
    
    # Determine configs to run
    if args.configs:
        configs = args.configs
    elif args.quick:
        configs = QUICK_CONFIGS
    else:
        configs = list(ABLATION_CONFIGS.keys())
    
    # Run ablation study
    run_ablation_study(
        synthetic_data_path=args.synthetic_data,
        realworld_data_path=args.realworld_data,
        configs=configs,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        heads=args.heads,
        dropout=args.dropout,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
        batch_size=args.batch_size,
        seed=args.seed,
        num_runs=args.runs,
        output_path=args.output,
    )


if __name__ == '__main__':
    main()
