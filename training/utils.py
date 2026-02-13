"""
Shared utilities for training smart contract vulnerability detection models.
"""
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support, 
    confusion_matrix, 
    roc_auc_score,
)


def setup_seed(seed: int) -> None:
    """
    Set random seed for reproducibility across all libraries.
    
    This ensures deterministic behavior by:
    - Seeding Python's random module
    - Seeding NumPy
    - Seeding PyTorch (CPU and CUDA)
    - Disabling CuDNN benchmarking
    - Enabling CuDNN deterministic mode
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU
        
        # Disable CuDNN benchmarking for deterministic behavior
        torch.backends.cudnn.benchmark = False
        
        # Enable CuDNN deterministic mode
        torch.backends.cudnn.deterministic = True


def setup_device() -> torch.device:
    """Get the best available device."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def convert_to_binary(labels: torch.Tensor) -> torch.Tensor:
    """Convert multiclass labels to binary (0=clean, 1=vulnerable)."""
    return (labels > 0).long()


def train_epoch(
    model: nn.Module,
    loader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    binary: bool = False,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        edge_type = batch.edge_attr if hasattr(batch, 'edge_attr') else None
        logits = model(batch.x, batch.edge_index, edge_type, batch.batch)
        
        labels = batch.graph_y.squeeze(-1) if batch.graph_y.dim() > 1 else batch.graph_y
        if binary:
            labels = convert_to_binary(labels)
        
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item() * batch.num_graphs
    
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate_binary(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, Any]:
    """Evaluate model for binary classification."""
    model.eval()
    total_loss = 0
    all_preds, all_labels, all_probs = [], [], []
    
    for batch in loader:
        batch = batch.to(device)
        
        edge_type = batch.edge_attr if hasattr(batch, 'edge_attr') else None
        logits = model(batch.x, batch.edge_index, edge_type, batch.batch)
        
        graph_y = batch.graph_y.squeeze(-1) if batch.graph_y.dim() > 1 else batch.graph_y
        labels = convert_to_binary(graph_y)
        loss = criterion(logits, labels)
        total_loss += loss.item() * batch.num_graphs
        
        probs = torch.softmax(logits, dim=-1)[:, 1]
        preds = logits.argmax(dim=-1)
        
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        all_probs.extend(probs.cpu().tolist())
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary', zero_division=0
    )
    
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.5
    
    return {
        'loss': total_loss / len(loader.dataset),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'predictions': all_preds,
        'labels': all_labels,
    }


@torch.no_grad()
def evaluate_multiclass(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, Any]:
    """Evaluate model for multiclass classification."""
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    
    for batch in loader:
        batch = batch.to(device)
        
        edge_type = batch.edge_attr if hasattr(batch, 'edge_attr') else None
        logits = model(batch.x, batch.edge_index, edge_type, batch.batch)
        
        labels = batch.graph_y.squeeze(-1) if batch.graph_y.dim() > 1 else batch.graph_y
        loss = criterion(logits, labels)
        total_loss += loss.item() * batch.num_graphs
        
        preds = logits.argmax(dim=-1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )
    
    return {
        'loss': total_loss / len(loader.dataset),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'predictions': all_preds,
        'labels': all_labels,
    }


def train_with_early_stopping(
    model: nn.Module,
    train_loader,
    val_loader,
    device: torch.device,
    epochs: int = 100,
    lr: float = 0.001,
    weight_decay: float = 1e-4,
    patience: int = 15,
    output_dir: Optional[Path] = None,
    class_weights: Optional[torch.Tensor] = None,
    binary: bool = False,
    verbose: bool = True,
) -> Tuple[Dict[str, Any], Dict[str, List]]:
    """
    Train model with early stopping.
    
    Returns:
        Tuple of (best_metrics, history)
    """
    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    else:
        criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    evaluate_fn = evaluate_binary if binary else evaluate_multiclass
    
    best_val_f1 = 0
    best_epoch = 0
    patience_counter = 0
    history = {'train': [], 'val': []}
    
    # Create progress bar - always show unless explicitly disabled
    pbar = tqdm(range(1, epochs + 1), desc="Training", disable=False, leave=True)
    
    for epoch in pbar:
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, binary)
        train_metrics = evaluate_fn(model, train_loader, criterion, device)
        val_metrics = evaluate_fn(model, val_loader, criterion, device)
        
        history['train'].append(train_metrics)
        history['val'].append(val_metrics)
        scheduler.step()
        
        # Update progress bar with metrics (always show progress)
        auc_str = f" AUC={val_metrics.get('auc', 0):.4f}" if binary else ""
        pbar.set_postfix_str(
            f"Loss={train_loss:.4f} | Val F1={val_metrics['f1']:.4f}{auc_str} | Best={best_val_f1:.4f}"
        )
        
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            best_epoch = epoch
            patience_counter = 0
            if output_dir:
                torch.save(model.state_dict(), output_dir / 'best_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                pbar.set_description(f"Early stop @ {epoch}")
                break
    
    pbar.close()
    if verbose:
        print(f"Best validation F1: {best_val_f1:.4f} at epoch {best_epoch}")
    
    # Load best model
    if output_dir and (output_dir / 'best_model.pt').exists():
        model.load_state_dict(torch.load(output_dir / 'best_model.pt'))
    
    return {'best_f1': best_val_f1, 'best_epoch': best_epoch}, history


def print_binary_results(metrics: Dict[str, Any]) -> None:
    """Print binary classification results."""
    print(f"\nTest Results (Binary Classification):")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    print(f"  AUC:       {metrics.get('auc', 0):.4f}")
    
    cm = confusion_matrix(metrics['labels'], metrics['predictions'])
    print(f"\nConfusion Matrix:")
    print(f"                 Predicted")
    print(f"             Clean  Vulnerable")
    print(f"  Actual Clean    {cm[0,0]:4d}      {cm[0,1]:4d}")
    print(f"  Actual Vuln     {cm[1,0]:4d}      {cm[1,1]:4d}")


def print_multiclass_results(metrics: Dict[str, Any], idx_to_label: Dict[int, str]) -> None:
    """Print multiclass classification results."""
    print(f"\nTest Results (Multiclass Classification):")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    
    print("\nPer-class Results:")
    precision, recall, f1, support = precision_recall_fscore_support(
        metrics['labels'], metrics['predictions'], average=None, zero_division=0
    )
    
    for i, (p, r, f, s) in enumerate(zip(precision, recall, f1, support)):
        if s > 0:
            label = idx_to_label.get(i, f'class_{i}')
            print(f"  {label:20s}: P={p:.3f} R={r:.3f} F1={f:.3f} (n={s})")


def save_results(
    output_dir: Path,
    config: Dict[str, Any],
    metrics: Dict[str, Any],
    num_params: int,
) -> None:
    """Save training results to JSON."""
    results = {
        'config': config,
        'test_metrics': {k: v for k, v in metrics.items() if k not in ['predictions', 'labels']},
        'num_params': num_params,
        'timestamp': datetime.now().isoformat(),
    }
    
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_dir}")


def compute_binary_weights(clean_count: int, vuln_count: int) -> torch.Tensor:
    """Compute class weights for binary classification."""
    total = clean_count + vuln_count
    return torch.tensor([
        total / (2 * clean_count),
        total / (2 * vuln_count),
    ])
