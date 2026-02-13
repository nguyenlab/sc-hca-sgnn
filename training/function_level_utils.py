"""
Training utilities for function-level vulnerability detection.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
)


# ============================================================================
# Advanced Loss Functions
# ============================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Focuses training on hard examples by down-weighting easy ones.
    """
    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = 'mean',
    ):
        super().__init__()
        self.alpha = alpha  # Class weights
        self.gamma = gamma  # Focusing parameter
        self.reduction = reduction
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)  # p_t
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing cross-entropy loss.
    
    Reduces overconfidence by softening hard labels.
    """
    def __init__(
        self,
        num_classes: int = 2,
        smoothing: float = 0.1,
        weight: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.weight = weight
        self.confidence = 1.0 - smoothing
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Create smoothed labels
        with torch.no_grad():
            smooth_labels = torch.zeros_like(log_probs)
            smooth_labels.fill_(self.smoothing / (self.num_classes - 1))
            smooth_labels.scatter_(1, targets.unsqueeze(1), self.confidence)
        
        # Weighted loss
        if self.weight is not None:
            weight = self.weight[targets]
            loss = -(smooth_labels * log_probs).sum(dim=-1) * weight
        else:
            loss = -(smooth_labels * log_probs).sum(dim=-1)
        
        return loss.mean()


# ============================================================================
# Threshold Optimization
# ============================================================================

def find_optimal_threshold(
    probs: List[float],
    labels: List[int],
    metric: str = 'f1',
) -> Tuple[float, float]:
    """
    Find optimal classification threshold that maximizes a metric.
    
    Args:
        probs: Predicted probabilities for positive class
        labels: True labels
        metric: Metric to optimize ('f1', 'precision', 'recall', 'balanced')
        
    Returns:
        Tuple of (optimal_threshold, best_metric_value)
    """
    probs = np.array(probs)
    labels = np.array(labels)
    
    precision, recall, thresholds = precision_recall_curve(labels, probs)
    
    # Compute F1 for each threshold
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    
    if metric == 'f1':
        best_idx = np.argmax(f1_scores)
        best_value = f1_scores[best_idx]
    elif metric == 'balanced':
        # Balance precision and recall
        balanced = np.sqrt(precision * recall)
        best_idx = np.argmax(balanced)
        best_value = balanced[best_idx]
    elif metric == 'precision':
        # Find threshold with highest precision at recall >= 0.8
        valid = recall >= 0.8
        if valid.any():
            best_idx = np.where(valid)[0][np.argmax(precision[valid])]
        else:
            best_idx = np.argmax(precision)
        best_value = precision[best_idx]
    else:  # recall
        # Find threshold with highest recall at precision >= 0.5
        valid = precision >= 0.5
        if valid.any():
            best_idx = np.where(valid)[0][np.argmax(recall[valid])]
        else:
            best_idx = np.argmax(recall)
        best_value = recall[best_idx]
    
    # Handle edge case
    if best_idx >= len(thresholds):
        best_idx = len(thresholds) - 1
    
    return thresholds[best_idx], best_value


def apply_threshold(probs: List[float], threshold: float) -> List[int]:
    """Apply classification threshold to probabilities."""
    return [1 if p >= threshold else 0 for p in probs]


def train_function_level_epoch(
    model: nn.Module,
    loader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """
    Train for one epoch on function-level data.
    
    Args:
        model: Function-level GNN model
        loader: DataLoader with FunctionLevelData
        optimizer: Optimizer
        criterion: Loss function
        device: Device to use
        
    Returns:
        Average loss
    """
    model.train()
    total_loss = 0
    total_functions = 0
    
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        # Get function-level predictions
        # Handle both batched and single graph cases
        if hasattr(batch, 'ptr') and batch.ptr is not None and len(batch.ptr) > 2:
            # Batched case - process each graph
            all_logits = []
            all_labels = []
            
            for i in range(len(batch.ptr) - 1):
                start, end = batch.ptr[i].item(), batch.ptr[i + 1].item()
                
                # Get subgraph info for this graph
                node_x = batch.x[start:end]
                node_subg = batch.subg[start:end]
                
                # Get edges for this graph
                edge_mask = (batch.edge_index[0] >= start) & (batch.edge_index[0] < end)
                graph_edge_index = batch.edge_index[:, edge_mask] - start
                
                sub_edge_mask = (batch.sub_edge_index[0] >= start) & (batch.sub_edge_index[0] < end)
                graph_sub_edge_index = batch.sub_edge_index[:, sub_edge_mask] - start
                
                # Get number of functions
                # y_func contains labels for all functions in batch, need to slice
                func_start = sum(batch.num_functions[:i].tolist()) if i > 0 else 0
                func_end = func_start + batch.num_functions[i].item()
                
                num_funcs = batch.num_functions[i].item()
                
                # Forward pass for this graph
                logits = model(
                    node_x, graph_edge_index, graph_sub_edge_index,
                    node_subg, num_funcs
                )
                
                all_logits.append(logits)
                all_labels.append(batch.y_func[func_start:func_end])
            
            logits = torch.cat(all_logits, dim=0)
            labels = torch.cat(all_labels, dim=0)
        else:
            # Single graph or simple batch
            num_funcs = batch.num_functions
            if isinstance(num_funcs, torch.Tensor):
                if num_funcs.dim() > 0:
                    num_funcs = num_funcs.sum().item()
                else:
                    num_funcs = num_funcs.item()
            
            logits = model(
                batch.x, batch.edge_index, batch.sub_edge_index,
                batch.subg, num_funcs
            )
            labels = batch.y_func
        
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item() * labels.size(0)
        total_functions += labels.size(0)
    
    return total_loss / total_functions if total_functions > 0 else 0


@torch.no_grad()
def evaluate_function_level(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Evaluate function-level model.
    
    Args:
        model: Function-level GNN model
        loader: DataLoader with FunctionLevelData
        criterion: Loss function
        device: Device to use
        
    Returns:
        Dictionary with metrics
    """
    model.eval()
    total_loss = 0
    total_functions = 0
    all_preds, all_labels, all_probs = [], [], []
    
    for batch in loader:
        batch = batch.to(device)
        
        # Same logic as training for handling batches
        if hasattr(batch, 'ptr') and batch.ptr is not None and len(batch.ptr) > 2:
            all_logits = []
            batch_labels = []
            
            for i in range(len(batch.ptr) - 1):
                start, end = batch.ptr[i].item(), batch.ptr[i + 1].item()
                
                node_x = batch.x[start:end]
                node_subg = batch.subg[start:end]
                
                edge_mask = (batch.edge_index[0] >= start) & (batch.edge_index[0] < end)
                graph_edge_index = batch.edge_index[:, edge_mask] - start
                
                sub_edge_mask = (batch.sub_edge_index[0] >= start) & (batch.sub_edge_index[0] < end)
                graph_sub_edge_index = batch.sub_edge_index[:, sub_edge_mask] - start
                
                func_start = sum(batch.num_functions[:i].tolist()) if i > 0 else 0
                func_end = func_start + batch.num_functions[i].item()
                
                num_funcs = batch.num_functions[i].item()
                
                logits = model(
                    node_x, graph_edge_index, graph_sub_edge_index,
                    node_subg, num_funcs
                )
                
                all_logits.append(logits)
                batch_labels.append(batch.y_func[func_start:func_end])
            
            logits = torch.cat(all_logits, dim=0)
            labels = torch.cat(batch_labels, dim=0)
        else:
            num_funcs = batch.num_functions
            if isinstance(num_funcs, torch.Tensor):
                if num_funcs.dim() > 0:
                    num_funcs = num_funcs.sum().item()
                else:
                    num_funcs = num_funcs.item()
            
            logits = model(
                batch.x, batch.edge_index, batch.sub_edge_index,
                batch.subg, num_funcs
            )
            labels = batch.y_func
        
        loss = criterion(logits, labels)
        total_loss += loss.item() * labels.size(0)
        total_functions += labels.size(0)
        
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
        'loss': total_loss / total_functions if total_functions > 0 else 0,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'predictions': all_preds,
        'labels': all_labels,
        'probs': all_probs,  # Added for threshold optimization
        'num_functions': total_functions,
    }


def train_function_level_with_early_stopping(
    model: nn.Module,
    train_loader,
    val_loader,
    device: torch.device,
    epochs: int = 100,
    lr: float = 0.001,
    weight_decay: float = 1e-4,
    patience: int = 15,
    output_dir=None,
    class_weights: Optional[torch.Tensor] = None,
    verbose: bool = True,
    use_focal_loss: bool = False,
    focal_gamma: float = 2.0,
    label_smoothing: float = 0.0,
    optimize_threshold: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, List]]:
    """
    Train function-level model with early stopping and advanced features.
    
    Args:
        model: Function-level GNN model
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device to use
        epochs: Maximum epochs
        lr: Learning rate
        weight_decay: Weight decay
        patience: Early stopping patience
        output_dir: Directory to save best model
        class_weights: Optional class weights for imbalanced data
        verbose: Whether to print progress
        use_focal_loss: Whether to use focal loss
        focal_gamma: Gamma parameter for focal loss
        label_smoothing: Label smoothing factor (0 = no smoothing)
        optimize_threshold: Whether to optimize classification threshold
        
    Returns:
        Tuple of (best_metrics, history)
    """
    # Select loss function
    if use_focal_loss:
        criterion = FocalLoss(
            alpha=class_weights.to(device) if class_weights is not None else None,
            gamma=focal_gamma,
        )
    elif label_smoothing > 0:
        criterion = LabelSmoothingLoss(
            num_classes=2,
            smoothing=label_smoothing,
            weight=class_weights.to(device) if class_weights is not None else None,
        )
    elif class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    else:
        criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_f1 = 0
    best_epoch = 0
    best_threshold = 0.5
    patience_counter = 0
    history = {'train': [], 'val': []}
    
    pbar = tqdm(range(1, epochs + 1), desc="Training", disable=False, leave=True)
    
    for epoch in pbar:
        train_loss = train_function_level_epoch(model, train_loader, optimizer, criterion, device)
        train_metrics = evaluate_function_level(model, train_loader, criterion, device)
        val_metrics = evaluate_function_level(model, val_loader, criterion, device)
        
        # Optionally find optimal threshold
        if optimize_threshold:
            opt_threshold, opt_f1 = find_optimal_threshold(
                val_metrics['probs'], val_metrics['labels'], metric='f1'
            )
            val_metrics['optimal_threshold'] = opt_threshold
            val_metrics['optimal_f1'] = opt_f1
            display_f1 = opt_f1
        else:
            display_f1 = val_metrics['f1']
        
        history['train'].append(train_metrics)
        history['val'].append(val_metrics)
        scheduler.step()
        
        pbar.set_postfix_str(
            f"Loss={train_loss:.4f} | Val F1={display_f1:.4f} AUC={val_metrics['auc']:.4f} | Best={best_val_f1:.4f}"
        )
        
        if display_f1 > best_val_f1:
            best_val_f1 = display_f1
            best_epoch = epoch
            if optimize_threshold:
                best_threshold = opt_threshold
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
        if optimize_threshold:
            print(f"Optimal threshold: {best_threshold:.4f}")
    
    # Load best model
    if output_dir and (output_dir / 'best_model.pt').exists():
        model.load_state_dict(torch.load(output_dir / 'best_model.pt'))
    
    return {
        'best_f1': best_val_f1,
        'best_epoch': best_epoch,
        'best_threshold': best_threshold,
    }, history


def compute_function_weights(clean_count: int, vuln_count: int) -> torch.Tensor:
    """Compute class weights for function-level classification."""
    total = clean_count + vuln_count
    if clean_count == 0 or vuln_count == 0:
        return torch.tensor([1.0, 1.0])
    return torch.tensor([
        total / (2 * clean_count),
        total / (2 * vuln_count),
    ])


def print_function_level_results(metrics: Dict[str, Any], threshold: float = 0.5) -> None:
    """Print function-level classification results."""
    print(f"\nTest Results (Function-Level Classification):")
    print(f"  Total functions evaluated: {metrics['num_functions']}")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    print(f"  AUC:       {metrics['auc']:.4f}")
    
    cm = confusion_matrix(metrics['labels'], metrics['predictions'])
    print(f"\nConfusion Matrix:")
    print(f"                 Predicted")
    print(f"             Clean  Vulnerable")
    print(f"  Actual Clean    {cm[0,0]:4d}      {cm[0,1]:4d}")
    print(f"  Actual Vuln     {cm[1,0]:4d}      {cm[1,1]:4d}")
    
    # Show optimal threshold results if probs available
    if 'probs' in metrics and threshold != 0.5:
        opt_preds = apply_threshold(metrics['probs'], threshold)
        opt_precision, opt_recall, opt_f1, _ = precision_recall_fscore_support(
            metrics['labels'], opt_preds, average='binary', zero_division=0
        )
        print(f"\n  With optimal threshold ({threshold:.3f}):")
        print(f"    Precision: {opt_precision:.4f}")
        print(f"    Recall:    {opt_recall:.4f}")
        print(f"    F1 Score:  {opt_f1:.4f}")


@torch.no_grad()
def evaluate_with_threshold(
    model: nn.Module,
    loader,
    device: torch.device,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Evaluate function-level model with custom threshold.
    
    Args:
        model: Function-level GNN model
        loader: DataLoader with FunctionLevelData
        device: Device to use
        threshold: Classification threshold
        
    Returns:
        Dictionary with metrics
    """
    criterion = nn.CrossEntropyLoss()
    metrics = evaluate_function_level(model, loader, criterion, device)
    
    # Apply custom threshold
    preds_with_threshold = apply_threshold(metrics['probs'], threshold)
    
    accuracy = accuracy_score(metrics['labels'], preds_with_threshold)
    precision, recall, f1, _ = precision_recall_fscore_support(
        metrics['labels'], preds_with_threshold, average='binary', zero_division=0
    )
    
    return {
        'loss': metrics['loss'],
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': metrics['auc'],
        'predictions': preds_with_threshold,
        'labels': metrics['labels'],
        'probs': metrics['probs'],
        'num_functions': metrics['num_functions'],
        'threshold': threshold,
    }


@torch.no_grad()
def evaluate_contract_level(
    model: nn.Module,
    loader,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Evaluate contract-level performance from function-level predictions.
    
    Aggregation rule: If ANY function is predicted vulnerable, contract is vulnerable.
    
    Args:
        model: Function-level GNN model
        loader: DataLoader with FunctionLevelData
        device: Device to use
        
    Returns:
        Dictionary with contract-level metrics
    """
    model.eval()
    
    all_contract_preds = []
    all_contract_labels = []
    all_contract_probs = []
    
    for batch in loader:
        batch = batch.to(device)
        
        # Handle batched data
        if hasattr(batch, 'ptr') and batch.ptr is not None and len(batch.ptr) > 2:
            # Process each contract in the batch
            for i in range(len(batch.ptr) - 1):
                start, end = batch.ptr[i].item(), batch.ptr[i + 1].item()
                
                # Get subgraph info for this contract
                node_x = batch.x[start:end]
                node_subg = batch.subg[start:end]
                
                # Get edges for this contract
                edge_mask = (batch.edge_index[0] >= start) & (batch.edge_index[0] < end)
                graph_edge_index = batch.edge_index[:, edge_mask] - start
                
                sub_edge_mask = (batch.sub_edge_index[0] >= start) & (batch.sub_edge_index[0] < end)
                graph_sub_edge_index = batch.sub_edge_index[:, sub_edge_mask] - start
                
                num_funcs = batch.num_functions[i].item()
                
                # Forward pass for this contract
                func_logits = model(
                    node_x, graph_edge_index, graph_sub_edge_index,
                    node_subg, num_funcs
                )
                
                # Aggregate function predictions to contract level
                # If ANY function is predicted vulnerable (class 1), contract is vulnerable
                func_probs = torch.softmax(func_logits, dim=-1)[:, 1]  # Prob of vulnerable
                func_preds = func_logits.argmax(dim=-1)
                
                # Contract is vulnerable if any function is vulnerable
                contract_pred = 1 if func_preds.max().item() > 0 else 0
                contract_prob = func_probs.max().item()  # Max probability among all functions
                
                # Get contract label
                contract_label = batch.graph_y[i].item()
                
                all_contract_preds.append(contract_pred)
                all_contract_labels.append(contract_label)
                all_contract_probs.append(contract_prob)
        else:
            # Single contract
            num_funcs = batch.num_functions
            if isinstance(num_funcs, torch.Tensor):
                if num_funcs.dim() > 0:
                    num_funcs = num_funcs.sum().item()
                else:
                    num_funcs = num_funcs.item()
            
            func_logits = model(
                batch.x, batch.edge_index, batch.sub_edge_index,
                batch.subg, num_funcs
            )
            
            func_probs = torch.softmax(func_logits, dim=-1)[:, 1]
            func_preds = func_logits.argmax(dim=-1)
            
            contract_pred = 1 if func_preds.max().item() > 0 else 0
            contract_prob = func_probs.max().item()
            contract_label = batch.graph_y.item() if batch.graph_y.dim() == 0 else batch.graph_y[0].item()
            
            all_contract_preds.append(contract_pred)
            all_contract_labels.append(contract_label)
            all_contract_probs.append(contract_prob)
    
    # Compute metrics
    accuracy = accuracy_score(all_contract_labels, all_contract_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_contract_labels, all_contract_preds, average='binary', zero_division=0
    )
    
    try:
        auc = roc_auc_score(all_contract_labels, all_contract_probs)
    except:
        auc = 0.5
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'predictions': all_contract_preds,
        'labels': all_contract_labels,
        'probs': all_contract_probs,
        'num_contracts': len(all_contract_labels),
    }
