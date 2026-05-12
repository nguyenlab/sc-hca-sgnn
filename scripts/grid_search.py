import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from itertools import product
import warnings

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

warnings.filterwarnings('ignore')

# Graph-level imports
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
    get_input_dim,
    get_num_edge_types,
)

# Function-level imports
from models.function_level_data import (
    FunctionLevelDataset,
    create_function_level_loaders,
)
from models.subgraph_level.function_level_gnn import create_function_level_model
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
from models.subgraph_level.hierarchical_cross_attention import create_hca_sgnn_model

from training.utils import (
    setup_seed,
    setup_device,
    train_with_early_stopping,
    evaluate_binary,
    compute_binary_weights,
)
from training.function_level_utils import (
    train_function_level_with_early_stopping,
    evaluate_function_level,
    compute_function_weights,
)


# Hyperparameter search spaces for each model (~50 combinations each)
SEARCH_SPACES = {
    'hierarchical-rgcn': {
        'hidden_dim': [64, 128, 256],
        'num_layers': [3, 4, 5],
        'dropout': [0.2, 0.3, 0.4],
        'lr': [0.001, 0.0005],
    },  # 3*3*3*2 = 54 combinations
    'dr-gcn': {
        'hidden_dim': [64, 128, 256],
        'num_layers': [3, 4],
        'dropout': [0.2, 0.3, 0.4],
        'pooling': ['mean', 'both'],
        'lr': [0.001, 0.0005],
    },  # 3*2*3*2*2 = 72 combinations
    'tmp': {
        'hidden_dim': [64, 128, 256],
        'num_layers': [3, 4],
        'heads': [4, 8],
        'dropout': [0.2, 0.3, 0.4],
        'pooling': ['mean', 'both'],
        'lr': [0.001, 0.0005],
    },  # 3*2*2*3*2*2 = 144 combinations
    'gat': {
        'hidden_dim': [64, 128, 256],
        'num_layers': [3, 4],
        'heads': [4, 8],
        'dropout': [0.2, 0.3, 0.4],
        'pooling': ['mean', 'both'],
        'lr': [0.001, 0.0005],
    },  # 3*2*2*3*2*2 = 144 combinations
    'transformer': {
        'hidden_dim': [64, 128, 256],
        'num_layers': [3, 4],
        'heads': [4, 8],
        'dropout': [0.2, 0.3, 0.4],
        'pooling': ['mean', 'both'],
        'lr': [0.001, 0.0005],
    },  # 3*2*2*3*2*2 = 144 combinations
    'bugsweeper': {
        'hidden_dim': [512, 1024],
        'dropout': [0.2, 0.3, 0.4],
        'num_sage_layers': [3],
        'num_gat_layers': [3],
        'lr': [0.001, 0.0005],
    },  # 2*3*1*1*2 = 12 combinations
    'bugsweeper-light': {
        'hidden_dim': [128, 256, 384],
        'dropout': [0.2, 0.3, 0.4],
        'num_sage_layers': [3, 4],
        'num_gat_layers': [3],
        'pooling': ['mean', 'both'],
        'lr': [0.001, 0.0005],
    },  # 3*3*2*1*2*2 = 72 combinations
    'scvhunter': {
        'hidden_dim': [128, 256, 384],
        'num_layers': [3, 4],
        'heads': [4, 8],
        'dropout': [0.2, 0.3, 0.4],
        'lr': [0.001, 0.0005],
    },  # 3*2*2*3*2 = 72 combinations
    'mlagnn': {
        'hidden_dim': [128, 256, 384],
        'num_layers': [3, 4],
        'heads': [4, 8],
        'dropout': [0.2, 0.3, 0.4],
        'lr': [0.001, 0.0005],
    },  # 3*2*2*3*2 = 72 combinations
    'bsgvd': {
        'hidden_dim': [64, 128, 256],
        'num_layers': [2, 3],
        'heads': [2, 4],
        'dropout': [0.2, 0.3, 0.4],
        'pooling': ['mean', 'both'],
        'lr': [0.001, 0.0005],
    },  # 3*2*2*3*2*2 = 144 combinations
}

# ============================================================================
# Function-Level Model Search Spaces
# ============================================================================

FUNC_SEARCH_SPACES = {
    'func-sage-mean': {
        'hidden_dim': [128, 256],
        'num_layers': [3, 4],
        'dropout': [0.2, 0.3, 0.4],
        'lr': [0.001, 0.0005],
    },  # 2*2*3*2 = 24 combinations
    'func-gat-mean': {
        'hidden_dim': [128, 256],
        'num_layers': [3, 4],
        'heads': [4, 8],
        'dropout': [0.2, 0.3, 0.4],
        'lr': [0.001, 0.0005],
    },  # 2*2*2*3*2 = 48 combinations
    'func-transformer-mean': {
        'hidden_dim': [128, 256],
        'num_layers': [3, 4],
        'heads': [4, 8],
        'dropout': [0.2, 0.3, 0.4],
        'lr': [0.001, 0.0005],
    },  # 2*2*2*3*2 = 48 combinations
    'func-dr-gcn': {
        'hidden_dim': [128, 256],
        'num_layers': [3, 4],
        'dropout': [0.2, 0.3, 0.4],
        'pooling': ['mean', 'both'],
        'lr': [0.001, 0.0005],
    },  # 2*2*3*2*2 = 48 combinations
    'func-tmp': {
        'hidden_dim': [128, 256],
        'num_layers': [3, 4],
        'heads': [4, 8],
        'dropout': [0.2, 0.3, 0.4],
        'pooling': ['mean', 'both'],
        'lr': [0.001, 0.0005],
    },  # 2*2*2*3*2*2 = 96 combinations
    'func-scvhunter': {
        'hidden_dim': [128, 256, 384],
        'num_layers': [3, 4],
        'heads': [4, 8],
        'dropout': [0.2, 0.3, 0.4],
        'pooling': ['mean', 'both'],
        'lr': [0.001, 0.0005],
    },  # 3*2*2*3*2*2 = 144 combinations
    'func-mlagnn': {
        'hidden_dim': [128, 256, 384],
        'num_layers': [3, 4],
        'heads': [4, 8],
        'dropout': [0.2, 0.3, 0.4],
        'pooling': ['mean', 'both'],
        'lr': [0.001, 0.0005],
    },  # 3*2*2*3*2*2 = 144 combinations
    'func-hca-mean': {
        'hidden_dim': [128, 256, 384],
        'num_layers': [3, 4, 5],
        'heads': [4, 8],
        'dropout': [0.2, 0.3, 0.4],
        'edge_dropout': [0.0, 0.1],
        'lr': [0.001, 0.0005],
    },  # 3*3*2*3*2*2 = 216 combinations
    'func-hca': {
        'hidden_dim': [128, 256, 384],
        'num_layers': [3, 4, 5],
        'heads': [4, 8],
        'dropout': [0.2, 0.3, 0.4],
        'edge_dropout': [0.0, 0.1],
        'lr': [0.001, 0.0005],
    },  # 3*3*2*3*2*2 = 216 combinations
}

# Quick search spaces (fewer combinations for faster testing)
QUICK_SEARCH_SPACES = {
    'hierarchical-rgcn': {
        'hidden_dim': [128, 256],
        'num_layers': [4],
        'dropout': [0.3],
        'lr': [0.001],
    },
    'dr-gcn': {
        'hidden_dim': [128, 256],
        'num_layers': [3, 4],
        'dropout': [0.3],
        'pooling': ['mean'],
        'lr': [0.001],
    },
    'tmp': {
        'hidden_dim': [128, 256],
        'num_layers': [3],
        'heads': [4],
        'dropout': [0.3],
        'pooling': ['mean'],
        'lr': [0.001],
    },
    'gat': {
        'hidden_dim': [128, 256],
        'num_layers': [3, 4],
        'heads': [4, 8],
        'dropout': [0.3],
        'pooling': ['mean'],
        'lr': [0.001],
    },
    'transformer': {
        'hidden_dim': [128, 256],
        'num_layers': [3],
        'heads': [4],
        'dropout': [0.3],
        'pooling': ['mean'],
        'lr': [0.001],
    },
    'bugsweeper': {
        'hidden_dim': [1024],
        'dropout': [0.3],
        'num_sage_layers': [3],
        'num_gat_layers': [3],
        'lr': [0.001],
    },
    'bugsweeper-light': {
        'hidden_dim': [128, 256],
        'dropout': [0.3],
        'num_sage_layers': [3],
        'num_gat_layers': [3],
        'pooling': ['mean'],
        'lr': [0.001],
    },
    'scvhunter': {
        'hidden_dim': [128, 256],
        'num_layers': [3],
        'heads': [4],
        'dropout': [0.3],
        'lr': [0.001],
    },
    'mlagnn': {
        'hidden_dim': [128, 256],
        'num_layers': [3],
        'heads': [4],
        'dropout': [0.3],
        'lr': [0.001],
    },
    'bsgvd': {
        'hidden_dim': [128, 256],
        'num_layers': [2, 3],
        'heads': [2, 4],
        'dropout': [0.3],
        'pooling': ['mean'],
        'lr': [0.001],
    },
}

# Quick search spaces for function-level models
FUNC_QUICK_SEARCH_SPACES = {
    'func-sage-mean': {
        'hidden_dim': [128, 256],
        'num_layers': [3],
        'dropout': [0.3],
        'lr': [0.001],
    },
    'func-gat-mean': {
        'hidden_dim': [128, 256],
        'num_layers': [3],
        'heads': [4],
        'dropout': [0.3],
        'lr': [0.001],
    },
    'func-transformer-mean': {
        'hidden_dim': [128, 256],
        'num_layers': [3],
        'heads': [4],
        'dropout': [0.3],
        'lr': [0.001],
    },
    'func-dr-gcn': {
        'hidden_dim': [128, 256],
        'num_layers': [3],
        'dropout': [0.3],
        'pooling': ['mean'],
        'lr': [0.001],
    },
    'func-tmp': {
        'hidden_dim': [128, 256],
        'num_layers': [3],
        'heads': [4],
        'dropout': [0.3],
        'pooling': ['mean'],
        'lr': [0.001],
    },
    'func-scvhunter': {
        'hidden_dim': [128, 256],
        'num_layers': [3],
        'heads': [4],
        'dropout': [0.3],
        'pooling': ['mean'],
        'lr': [0.001],
    },
    'func-mlagnn': {
        'hidden_dim': [128, 256],
        'num_layers': [3],
        'heads': [4],
        'dropout': [0.3],
        'pooling': ['mean'],
        'lr': [0.001],
    },
    'func-hca-mean': {
        'hidden_dim': [128, 256],
        'num_layers': [3, 4],
        'heads': [4],
        'dropout': [0.3],
        'edge_dropout': [0.0, 0.1],
        'lr': [0.001],
    },
    'func-hca': {
        'hidden_dim': [128, 256],
        'num_layers': [3, 4],
        'heads': [4],
        'dropout': [0.3],
        'edge_dropout': [0.0, 0.1],
        'lr': [0.001],
    },
}


def create_model_instance(
    model_key: str,
    params: Dict[str, Any],
    num_classes: int,
    num_edge_types: int,
    device: str,
) -> nn.Module:
    """Create a model instance with given hyperparameters."""
    input_dim = get_input_dim()
    
    if model_key == 'hierarchical-rgcn':
        model = create_model(
            model_type='hierarchical',
            input_dim=input_dim,
            hidden_dim=params['hidden_dim'],
            num_layers=params['num_layers'],
            dropout=params['dropout'],
            num_classes=num_classes,
            num_edge_types=num_edge_types,
            mode='rgcn',
        )
    elif model_key == 'dr-gcn':
        model = create_dr_gcn_model(
            input_dim=input_dim,
            hidden_dim=params['hidden_dim'],
            num_layers=params['num_layers'],
            num_classes=num_classes,
            dropout=params['dropout'],
            pooling=params.get('pooling', 'mean'),
        )
    elif model_key == 'tmp':
        model = create_tmp_model(
            input_dim=input_dim,
            hidden_dim=params['hidden_dim'],
            num_layers=params['num_layers'],
            heads=params.get('heads', 4),
            num_classes=num_classes,
            dropout=params['dropout'],
            pooling=params.get('pooling', 'mean'),
        )
    elif model_key == 'gat':
        model = create_gat_model(
            input_dim=input_dim,
            hidden_dim=params['hidden_dim'],
            num_layers=params['num_layers'],
            heads=params.get('heads', 4),
            num_classes=num_classes,
            dropout=params['dropout'],
            pooling=params.get('pooling', 'mean'),
        )
    elif model_key == 'transformer':
        model = create_transformer_gnn_model(
            input_dim=input_dim,
            hidden_dim=params['hidden_dim'],
            num_layers=params['num_layers'],
            heads=params.get('heads', 4),
            num_classes=num_classes,
            dropout=params['dropout'],
            pooling=params.get('pooling', 'mean'),
            use_positional_encoding=False,
        )
    elif model_key == 'bugsweeper':
        model = create_bugsweeper_model(
            input_dim=input_dim,
            hidden_dim=params['hidden_dim'],
            num_classes=num_classes,
            num_sage_layers=params.get('num_sage_layers', 3),
            num_gat_layers=params.get('num_gat_layers', 3),
            dropout=params['dropout'],
        )
    elif model_key == 'bugsweeper-light':
        model = create_bugsweeper_light_model(
            input_dim=input_dim,
            hidden_dim=params['hidden_dim'],
            num_classes=num_classes,
            num_sage_layers=params.get('num_sage_layers', 3),
            num_gat_layers=params.get('num_gat_layers', 3),
            dropout=params['dropout'],
            pooling=params.get('pooling', 'mean'),
        )
    elif model_key == 'scvhunter':
        model = create_scvhunter_model(
            input_dim=input_dim,
            hidden_dim=params['hidden_dim'],
            num_layers=params['num_layers'],
            num_edge_types=num_edge_types,
            heads=params.get('heads', 4),
            num_classes=num_classes,
            dropout=params['dropout'],
        )
    elif model_key == 'mlagnn':
        model = create_mlagnn_model(
            input_dim=input_dim,
            hidden_dim=params['hidden_dim'],
            num_layers=params['num_layers'],
            num_edge_types=num_edge_types,
            heads=params.get('heads', 4),
            num_classes=num_classes,
            dropout=params['dropout'],
        )
    elif model_key == 'bsgvd':
        model = create_bsgvd_model(
            input_dim=input_dim,
            hidden_dim=params['hidden_dim'],
            num_layers=params['num_layers'],
            heads=params.get('heads', 2),
            num_classes=num_classes,
            dropout=params['dropout'],
            pooling=params.get('pooling', 'mean'),
        )
    else:
        raise ValueError(f"Unknown model: {model_key}")
    
    return model.to(device)


def create_function_level_model_instance(
    model_key: str,
    params: Dict[str, Any],
    num_classes: int,
    num_edge_types: int,
    device: str,
) -> nn.Module:
    """Create a function-level model instance with given hyperparameters."""
    input_dim = get_input_dim()
    
    if model_key == 'func-sage-mean':
        model = create_function_level_model(
            input_dim=input_dim,
            hidden_dim=params['hidden_dim'],
            num_layers=params['num_layers'],
            num_classes=num_classes,
            dropout=params['dropout'],
            conv_type='sage',
            pooling='mean',
        )
    elif model_key == 'func-gat-mean':
        model = create_function_level_model(
            input_dim=input_dim,
            hidden_dim=params['hidden_dim'],
            num_layers=params['num_layers'],
            heads=params.get('heads', 4),
            num_classes=num_classes,
            dropout=params['dropout'],
            conv_type='gat',
            pooling='mean',
        )
    elif model_key == 'func-transformer-mean':
        model = create_function_level_model(
            input_dim=input_dim,
            hidden_dim=params['hidden_dim'],
            num_layers=params['num_layers'],
            heads=params.get('heads', 4),
            num_classes=num_classes,
            dropout=params['dropout'],
            conv_type='transformer',
            pooling='mean',
        )
    elif model_key == 'func-dr-gcn':
        model = create_function_level_dr_gcn_model(
            input_dim=input_dim,
            hidden_dim=params['hidden_dim'],
            num_layers=params['num_layers'],
            num_classes=num_classes,
            dropout=params['dropout'],
            pooling=params.get('pooling', 'mean'),
        )
    elif model_key == 'func-tmp':
        model = create_function_level_tmp_model(
            input_dim=input_dim,
            hidden_dim=params['hidden_dim'],
            num_layers=params['num_layers'],
            heads=params.get('heads', 4),
            num_classes=num_classes,
            dropout=params['dropout'],
            pooling=params.get('pooling', 'mean'),
        )
    elif model_key == 'func-scvhunter':
        model = create_function_level_scvhunter_model(
            input_dim=input_dim,
            hidden_dim=params['hidden_dim'],
            num_layers=params['num_layers'],
            num_edge_types=num_edge_types,
            heads=params.get('heads', 4),
            num_classes=num_classes,
            dropout=params['dropout'],
            pooling=params.get('pooling', 'mean'),
        )
    elif model_key == 'func-mlagnn':
        model = create_function_level_mlagnn_model(
            input_dim=input_dim,
            hidden_dim=params['hidden_dim'],
            num_layers=params['num_layers'],
            num_edge_types=num_edge_types,
            heads=params.get('heads', 4),
            num_classes=num_classes,
            dropout=params['dropout'],
            pooling=params.get('pooling', 'mean'),
        )
    elif model_key in ('func-hca', 'func-hca-mean'):
        pooling = 'mean' if model_key == 'func-hca-mean' else 'hierarchical'
        model = create_hca_sgnn_model(
            input_dim=input_dim,
            hidden_dim=params['hidden_dim'],
            num_layers=params['num_layers'],
            num_edge_types=num_edge_types,
            heads=params.get('heads', 4),
            num_classes=num_classes,
            dropout=params['dropout'],
            pooling=pooling,
            use_cross_attention=True,
            edge_dropout=params.get('edge_dropout', 0.0),
        )
    else:
        raise ValueError(f"Unknown function-level model: {model_key}")
    
    return model.to(device)


def evaluate_config(
    model_key: str,
    params: Dict[str, Any],
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
    epochs: int,
    patience: int,
    class_weights: Optional[torch.Tensor],
    seed: int,
) -> Dict[str, Any]:
    """Evaluate a single hyperparameter configuration."""
    setup_seed(seed)
    
    # Create model
    model = create_model_instance(
        model_key=model_key,
        params=params,
        num_classes=2,
        num_edge_types=get_num_edge_types(),
        device=device,
    )
    
    num_params = sum(p.numel() for p in model.parameters())
    
    # Train
    start_time = time.time()
    best_metrics, history = train_with_early_stopping(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=epochs,
        lr=params.get('lr', 0.001),
        patience=patience,
        output_dir=None,
        class_weights=class_weights,
        binary=True,
        verbose=False,  # Suppress individual training logs
    )
    train_time = time.time() - start_time
    
    return {
        'params': params,
        'num_params': num_params,
        'val_f1': best_metrics['best_f1'],
        'best_epoch': best_metrics['best_epoch'],
        'train_time': round(train_time, 2),
    }


def evaluate_func_config(
    model_key: str,
    params: Dict[str, Any],
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
    epochs: int,
    patience: int,
    class_weights: Optional[torch.Tensor],
    seed: int,
) -> Dict[str, Any]:
    """Evaluate a single hyperparameter configuration for function-level model."""
    setup_seed(seed)
    
    # Create model
    model = create_function_level_model_instance(
        model_key=model_key,
        params=params,
        num_classes=2,
        num_edge_types=get_num_edge_types(),
        device=device,
    )
    
    num_params = sum(p.numel() for p in model.parameters())
    
    # Train
    start_time = time.time()
    best_metrics, history = train_function_level_with_early_stopping(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=epochs,
        lr=params.get('lr', 0.001),
        patience=patience,
        output_dir=None,
        class_weights=class_weights,
        verbose=False,  # Suppress individual training logs
    )
    train_time = time.time() - start_time
    
    return {
        'params': params,
        'num_params': num_params,
        'val_f1': best_metrics['best_f1'],
        'best_epoch': best_metrics['best_epoch'],
        'train_time': round(train_time, 2),
    }


def func_grid_search(
    model_key: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
    epochs: int,
    patience: int,
    class_weights: Optional[torch.Tensor],
    seed: int,
    quick: bool = False,
) -> Dict[str, Any]:
    """Perform grid search for a function-level model."""
    search_space = FUNC_QUICK_SEARCH_SPACES[model_key] if quick else FUNC_SEARCH_SPACES[model_key]
    
    # Generate all parameter combinations
    param_names = list(search_space.keys())
    param_values = [search_space[name] for name in param_names]
    combinations = list(product(*param_values))
    
    print(f"\n{'='*70}")
    print(f"Grid Search (Function-Level): {model_key.upper()}")
    print(f"{'='*70}")
    print(f"Search space: {len(combinations)} combinations")
    for name, values in search_space.items():
        print(f"  {name}: {values}")
    
    results = []
    best_result = None
    best_f1 = 0
    
    for i, combo in enumerate(combinations, 1):
        params = dict(zip(param_names, combo))
        
        print(f"\n[{i}/{len(combinations)}] Testing: {params}")
        
        try:
            result = evaluate_func_config(
                model_key=model_key,
                params=params,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                epochs=epochs,
                patience=patience,
                class_weights=class_weights,
                seed=seed,
            )
            
            results.append(result)
            
            print(f"  Val F1: {result['val_f1']:.4f} | Epoch: {result['best_epoch']} | Time: {result['train_time']:.1f}s")
            
            if result['val_f1'] > best_f1:
                best_f1 = result['val_f1']
                best_result = result
                print(f"  ⭐ New best!")
        
        except Exception as e:
            print(f"  ❌ Failed: {e}")
            results.append({
                'params': params,
                'error': str(e),
            })
    
    return {
        'model': model_key,
        'search_space': search_space,
        'num_combinations': len(combinations),
        'results': results,
        'best_config': best_result,
    }


def grid_search(
    model_key: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
    epochs: int,
    patience: int,
    class_weights: Optional[torch.Tensor],
    seed: int,
    quick: bool = False,
) -> Dict[str, Any]:
    """Perform grid search for a model."""
    search_space = QUICK_SEARCH_SPACES[model_key] if quick else SEARCH_SPACES[model_key]
    
    # Generate all parameter combinations
    param_names = list(search_space.keys())
    param_values = [search_space[name] for name in param_names]
    combinations = list(product(*param_values))
    
    print(f"\n{'='*70}")
    print(f"Grid Search: {model_key.upper()}")
    print(f"{'='*70}")
    print(f"Search space: {len(combinations)} combinations")
    for name, values in search_space.items():
        print(f"  {name}: {values}")
    
    results = []
    best_result = None
    best_f1 = 0
    
    for i, combo in enumerate(combinations, 1):
        params = dict(zip(param_names, combo))
        
        print(f"\n[{i}/{len(combinations)}] Testing: {params}")
        
        try:
            result = evaluate_config(
                model_key=model_key,
                params=params,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                epochs=epochs,
                patience=patience,
                class_weights=class_weights,
                seed=seed,
            )
            
            results.append(result)
            
            print(f"  Val F1: {result['val_f1']:.4f} | Epoch: {result['best_epoch']} | Time: {result['train_time']:.1f}s")
            
            if result['val_f1'] > best_f1:
                best_f1 = result['val_f1']
                best_result = result
                print(f"  ⭐ New best!")
        
        except Exception as e:
            print(f"  ❌ Failed: {e}")
            results.append({
                'params': params,
                'error': str(e),
            })
    
    return {
        'model': model_key,
        'search_space': search_space,
        'num_combinations': len(combinations),
        'results': results,
        'best_config': best_result,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Grid Search for Smart Contract Vulnerability Detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Graph-level models
    python grid_search.py --model dr-gcn --epochs 50
    python grid_search.py --model scvhunter --epochs 50 --quick
    python grid_search.py --all --epochs 30
    
    # Function-level models
    python grid_search.py --model func-hca-mean --level function --epochs 50
    python grid_search.py --all --level function --epochs 30 --quick
        """
    )
    
    # Combine all model choices
    all_graph_models = list(SEARCH_SPACES.keys())
    all_func_models = list(FUNC_SEARCH_SPACES.keys())
    all_models = all_graph_models + all_func_models
    
    parser.add_argument('--model', '-m', 
                        choices=all_models,
                        help='Model to search')
    parser.add_argument('--all', action='store_true',
                        help='Search all models')
    parser.add_argument('--level', '-l', choices=['graph', 'function'], default='graph',
                        help='Model level (graph or function)')
    parser.add_argument('--data', '-d', default=None,
                        help='Dataset directory (auto-detected based on level if not specified)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Max training epochs')
    parser.add_argument('--patience', type=int, default=20,
                        help='Early stopping patience')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--quick', action='store_true',
                        help='Use quick search space (fewer combinations)')
    parser.add_argument('--output', '-o', default=None,
                        help='Output JSON file (auto-generated if not specified)')
    
    args = parser.parse_args()
    
    if not args.model and not args.all:
        parser.error("Must specify either --model or --all")
    
    # Auto-detect level from model name if model is specified
    if args.model:
        if args.model.startswith('func-'):
            args.level = 'function'
        elif args.model in SEARCH_SPACES:
            args.level = 'graph'
    
    # Set default data path based on level
    if args.data is None:
        if args.level == 'function':
            args.data = 'data/synthetic/function_level_dataset'
        else:
            args.data = 'data/synthetic/ast_dataset'
    
    # Set default output path
    if args.output is None:
        if args.level == 'function':
            args.output = 'results/grid_search/function_level/grid_search.json'
        else:
            args.output = 'results/grid_search/contract_level/grid_search.json'
    
    # Setup
    setup_seed(args.seed)
    device = setup_device()
    
    print("="*70)
    print("GRID SEARCH FOR SMART CONTRACT VULNERABILITY DETECTION")
    print("="*70)
    print(f"Level: {args.level}")
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}, Patience: {args.patience}")
    print(f"Quick mode: {args.quick}")
    
    # Load dataset based on level
    if args.level == 'function':
        print(f"\nLoading function-level dataset from {args.data}...")
        dataset = FunctionLevelDataset(root=args.data, bidirectional=True)
        stats = dataset.get_stats()
        
        clean_count = stats.num_clean_functions
        vuln_count = stats.num_vulnerable_functions
        class_weights = compute_function_weights(clean_count, vuln_count)
        
        train_loader, val_loader, test_loader = create_function_level_loaders(
            dataset, batch_size=args.batch_size, seed=args.seed
        )
        print(f"Contracts: {stats.num_contracts}, Functions: {stats.num_functions}")
        print(f"  Vulnerable: {vuln_count}, Clean: {clean_count}")
    else:
        print(f"\nLoading graph-level dataset from {args.data}...")
        dataset = ContractGraphDataset(root=args.data, use_edge_attr=True, bidirectional=True)
        stats = dataset.get_stats()
        
        clean_count = stats.vuln_distribution.get('clean', 0)
        vuln_count = sum(stats.vuln_distribution.values()) - clean_count
        class_weights = compute_binary_weights(clean_count, vuln_count)
        
        train_loader, val_loader, test_loader = create_data_loaders(
            dataset, batch_size=args.batch_size, seed=args.seed
        )
        print(f"Graphs: {stats.num_graphs}")
    
    print(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")
    
    # Determine which models to search
    if args.all:
        if args.level == 'function':
            models_to_search = list(FUNC_SEARCH_SPACES.keys())
        else:
            models_to_search = list(SEARCH_SPACES.keys())
    else:
        models_to_search = [args.model]
    
    # Run grid search
    all_results = {}
    total_start = time.time()
    
    for model_key in models_to_search:
        if args.level == 'function':
            result = func_grid_search(
                model_key=model_key,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                epochs=args.epochs,
                patience=args.patience,
                class_weights=class_weights,
                seed=args.seed,
                quick=args.quick,
            )
        else:
            result = grid_search(
                model_key=model_key,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                epochs=args.epochs,
                patience=args.patience,
                class_weights=class_weights,
                seed=args.seed,
                quick=args.quick,
            )
        all_results[model_key] = result
    
    total_time = time.time() - total_start
    
    # Print summary
    print("\n" + "="*70)
    print(f"GRID SEARCH SUMMARY ({args.level.upper()}-LEVEL)")
    print("="*70)
    print(f"\n{'Model':<25} {'Best Val F1':>12} {'Best Config'}")
    print("-"*70)
    
    for model_key, result in all_results.items():
        if result['best_config']:
            best = result['best_config']
            config_str = ', '.join(f"{k}={v}" for k, v in best['params'].items())
            print(f"{model_key:<25} {best['val_f1']:>12.4f} {config_str}")
        else:
            print(f"{model_key:<25} {'FAILED':>12}")
    
    print(f"\nTotal time: {total_time:.1f}s")
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    output_data = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'level': args.level,
            'epochs': args.epochs,
            'patience': args.patience,
            'batch_size': args.batch_size,
            'seed': args.seed,
            'quick': args.quick,
        },
        'results': all_results,
        'total_time_sec': round(total_time, 2),
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
