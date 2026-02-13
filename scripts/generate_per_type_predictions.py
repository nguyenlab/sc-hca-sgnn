import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from models.data import get_input_dim, get_num_edge_types
from models.function_level_data import FunctionLevelDataset
from models.subgraph_level.hierarchical_cross_attention import create_hca_sgnn_model
from training.utils import setup_device

# Same hyperparameters as ablation_study.py
HIDDEN_DIM = 128
NUM_LAYERS = 4
HEADS = 8
DROPOUT = 0.2
BATCH_SIZE = 32


def load_model(checkpoint_path: str, device: torch.device) -> nn.Module:
    """Load a trained HCA-SGNN model from checkpoint."""
    input_dim = get_input_dim()
    num_edge_types = get_num_edge_types()

    model = create_hca_sgnn_model(
        input_dim=input_dim,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        num_edge_types=num_edge_types,
        heads=HEADS,
        num_classes=2,
        dropout=DROPOUT,
        use_cross_attention=True,
        pooling='mean',
        use_rgcn=False,
        use_dual_channel=True,
        use_gated_fusion=True,
        use_node_attention=False,
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model


@torch.no_grad()
def predict_per_contract(model: nn.Module, loader: DataLoader, device: torch.device):
    """
    Run model on each contract and return per-contract predictions
    with vulnerability type metadata.

    Returns a list of dicts:
    [
        {
            'vuln_type': str,
            'contract_label': int (0 or 1),
            'contract_pred': int (0 or 1),
            'contract_prob': float,
            'functions': [
                {
                    'name': str,
                    'label': int,
                    'pred': int,
                    'prob': float
                }, ...
            ]
        }, ...
    ]
    """
    results = []

    for batch in loader:
        batch = batch.to(device)

        if hasattr(batch, 'ptr') and batch.ptr is not None and len(batch.ptr) > 2:
            for i in range(len(batch.ptr) - 1):
                start, end = batch.ptr[i].item(), batch.ptr[i + 1].item()

                node_x = batch.x[start:end]
                node_subg = batch.subg[start:end]

                edge_mask = (batch.edge_index[0] >= start) & (batch.edge_index[0] < end)
                graph_edge_index = batch.edge_index[:, edge_mask] - start

                sub_edge_mask = (batch.sub_edge_index[0] >= start) & (batch.sub_edge_index[0] < end)
                graph_sub_edge_index = batch.sub_edge_index[:, sub_edge_mask] - start

                num_funcs = batch.num_functions[i].item()

                func_logits = model(
                    node_x, graph_edge_index, graph_sub_edge_index,
                    node_subg, num_funcs
                )

                func_probs = torch.softmax(func_logits, dim=-1)[:, 1]
                func_preds = func_logits.argmax(dim=-1)

                contract_pred = 1 if func_preds.max().item() > 0 else 0
                contract_prob = func_probs.max().item()
                contract_label = batch.graph_y[i].item()

                # Get function labels
                func_start = sum(batch.num_functions[:i].tolist()) if i > 0 else 0
                func_end = func_start + num_funcs
                func_labels = batch.y_func[func_start:func_end]

                # Get vuln_type
                vuln_type = batch.vuln_type[i] if hasattr(batch, 'vuln_type') and isinstance(batch.vuln_type, list) else 'unknown'

                # Get function names
                func_names = batch.function_names[i] if hasattr(batch, 'function_names') and isinstance(batch.function_names, list) else [f'func_{j}' for j in range(num_funcs)]

                functions = []
                for j in range(num_funcs):
                    functions.append({
                        'name': func_names[j] if j < len(func_names) else f'func_{j}',
                        'label': func_labels[j].item(),
                        'pred': func_preds[j].item(),
                        'prob': round(func_probs[j].item(), 4),
                    })

                results.append({
                    'vuln_type': vuln_type,
                    'contract_label': contract_label,
                    'contract_pred': contract_pred,
                    'contract_prob': round(contract_prob, 4),
                    'functions': functions,
                })
        else:
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

            vuln_type = batch.vuln_type if hasattr(batch, 'vuln_type') and isinstance(batch.vuln_type, str) else 'unknown'

            func_names_data = batch.function_names if hasattr(batch, 'function_names') else [f'func_{j}' for j in range(num_funcs)]
            # handle list-of-lists for single contract
            if isinstance(func_names_data, list) and len(func_names_data) > 0 and isinstance(func_names_data[0], list):
                func_names_data = func_names_data[0]

            func_labels = batch.y_func

            functions = []
            for j in range(num_funcs):
                functions.append({
                    'name': func_names_data[j] if j < len(func_names_data) else f'func_{j}',
                    'label': func_labels[j].item(),
                    'pred': func_preds[j].item(),
                    'prob': round(func_probs[j].item(), 4),
                })

            results.append({
                'vuln_type': vuln_type,
                'contract_label': contract_label,
                'contract_pred': contract_pred,
                'contract_prob': round(contract_prob, 4),
                'functions': functions,
            })

    return results


def compute_per_type_metrics(predictions):
    """Compute per-vulnerability-type detection metrics."""
    # Group by vulnerability type
    by_type = defaultdict(list)
    for p in predictions:
        by_type[p['vuln_type']].append(p)

    per_type = {}
    for vtype, contracts in sorted(by_type.items()):
        n = len(contracts)
        # Contract-level
        contract_tp = sum(1 for c in contracts if c['contract_label'] == 1 and c['contract_pred'] == 1)
        contract_fp = sum(1 for c in contracts if c['contract_label'] == 0 and c['contract_pred'] == 1)
        contract_fn = sum(1 for c in contracts if c['contract_label'] == 1 and c['contract_pred'] == 0)
        contract_tn = sum(1 for c in contracts if c['contract_label'] == 0 and c['contract_pred'] == 0)

        # Function-level
        func_tp = func_fp = func_fn = func_tn = 0
        for c in contracts:
            for f in c['functions']:
                if f['label'] == 1 and f['pred'] == 1:
                    func_tp += 1
                elif f['label'] == 0 and f['pred'] == 1:
                    func_fp += 1
                elif f['label'] == 1 and f['pred'] == 0:
                    func_fn += 1
                else:
                    func_tn += 1

        # Compute rates
        def safe_div(a, b):
            return round(a / b, 4) if b > 0 else 0.0

        contract_precision = safe_div(contract_tp, contract_tp + contract_fp)
        contract_recall = safe_div(contract_tp, contract_tp + contract_fn)
        contract_f1 = safe_div(2 * contract_precision * contract_recall, contract_precision + contract_recall)

        func_precision = safe_div(func_tp, func_tp + func_fp)
        func_recall = safe_div(func_tp, func_tp + func_fn)
        func_f1 = safe_div(2 * func_precision * func_recall, func_precision + func_recall)

        per_type[vtype] = {
            'num_contracts': n,
            'contract_level': {
                'TP': contract_tp, 'FP': contract_fp,
                'FN': contract_fn, 'TN': contract_tn,
                'precision': contract_precision,
                'recall': contract_recall,
                'f1': contract_f1,
            },
            'function_level': {
                'TP': func_tp, 'FP': func_fp,
                'FN': func_fn, 'TN': func_tn,
                'precision': func_precision,
                'recall': func_recall,
                'f1': func_f1,
            },
        }

    return per_type


def print_results(per_type: Dict, dataset_name: str):
    """Pretty-print per-type detection results."""
    print(f"\n{'='*80}")
    print(f"  Per-Vulnerability-Type Detection: {dataset_name}")
    print(f"{'='*80}")

    print(f"\n  {'Type':<25} {'#Contracts':>10} {'C-TP':>6} {'C-FN':>6} {'C-Recall':>9} {'C-F1':>7} | {'F-TP':>6} {'F-FN':>6} {'F-Recall':>9} {'F-F1':>7}")
    print(f"  {'-'*25} {'-'*10} {'-'*6} {'-'*6} {'-'*9} {'-'*7} | {'-'*6} {'-'*6} {'-'*9} {'-'*7}")

    for vtype, m in sorted(per_type.items(), key=lambda x: x[1]['num_contracts'], reverse=True):
        c = m['contract_level']
        f = m['function_level']
        print(f"  {vtype:<25} {m['num_contracts']:>10} {c['TP']:>6} {c['FN']:>6} {c['recall']:>9.4f} {c['f1']:>7.4f} | {f['TP']:>6} {f['FN']:>6} {f['recall']:>9.4f} {f['f1']:>7.4f}")


def main():
    checkpoint_path = 'outputs/ablation_study/run_0/full/best_model.pt'
    synthetic_data = 'data/synthetic-split/test'
    realworld_data = 'data/realworld/function_level_dataset'

    device = setup_device()
    print(f"Device: {device}")

    # Load model
    print(f"Loading model from {checkpoint_path}...")
    model = load_model(checkpoint_path, device)
    print(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}")

    results = {}

    # ---- Synthetic test set ----
    print(f"\nLoading synthetic test set from {synthetic_data}...")
    syn_dataset = FunctionLevelDataset(root=synthetic_data, bidirectional=True)
    syn_loader = DataLoader(syn_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"  Contracts: {len(syn_dataset)}")

    syn_predictions = predict_per_contract(model, syn_loader, device)
    syn_per_type = compute_per_type_metrics(syn_predictions)
    print_results(syn_per_type, "Synthetic Test Set")
    results['synthetic'] = syn_per_type

    # ---- Real-world test set ----
    print(f"\nLoading real-world test set from {realworld_data}...")
    irl_dataset = FunctionLevelDataset(root=realworld_data, bidirectional=True)
    irl_loader = DataLoader(irl_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"  Contracts: {len(irl_dataset)}")

    irl_predictions = predict_per_contract(model, irl_loader, device)
    irl_per_type = compute_per_type_metrics(irl_predictions)
    print_results(irl_per_type, "Real-World Test Set")
    results['realworld'] = irl_per_type

    # ---- Save results ----
    output_path = Path('results/per_type_predictions.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
