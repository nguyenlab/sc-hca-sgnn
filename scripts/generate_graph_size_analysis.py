import json
import sys
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, List, Tuple

import torch
import torch.nn as nn
import numpy as np
from torch_geometric.loader import DataLoader

from models.data import get_input_dim, get_num_edge_types
from models.function_level_data import FunctionLevelDataset
from models.subgraph_level.hierarchical_cross_attention import create_hca_sgnn_model
from training.utils import setup_device

# Same hyperparameters as ablation_study.py / generate_per_type_predictions.py
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
def predict_with_graph_info(model: nn.Module, loader: DataLoader, device: torch.device):
    """
    Run model on each contract and return per-contract predictions
    with graph-size metadata (num_nodes, num_edges, num_functions).

    Returns a list of dicts with prediction info and graph structural metadata.
    """
    results = []

    for batch in loader:
        batch = batch.to(device)

        if hasattr(batch, 'ptr') and batch.ptr is not None and len(batch.ptr) > 2:
            for i in range(len(batch.ptr) - 1):
                start, end = batch.ptr[i].item(), batch.ptr[i + 1].item()

                node_x = batch.x[start:end]
                node_subg = batch.subg[start:end]
                num_nodes = end - start

                edge_mask = (batch.edge_index[0] >= start) & (batch.edge_index[0] < end)
                graph_edge_index = batch.edge_index[:, edge_mask] - start
                num_edges = edge_mask.sum().item()

                sub_edge_mask = (batch.sub_edge_index[0] >= start) & (batch.sub_edge_index[0] < end)
                graph_sub_edge_index = batch.sub_edge_index[:, sub_edge_mask] - start
                num_intra_edges = sub_edge_mask.sum().item()

                inter_edge_mask = (batch.inter_edge_index[0] >= start) & (batch.inter_edge_index[0] < end)
                num_inter_edges = inter_edge_mask.sum().item()

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

                func_start = sum(batch.num_functions[:i].tolist()) if i > 0 else 0
                func_end = func_start + num_funcs
                func_labels = batch.y_func[func_start:func_end]

                vuln_type = batch.vuln_type[i] if hasattr(batch, 'vuln_type') and isinstance(batch.vuln_type, list) else 'unknown'

                functions = []
                for j in range(num_funcs):
                    functions.append({
                        'label': func_labels[j].item(),
                        'pred': func_preds[j].item(),
                        'prob': round(func_probs[j].item(), 4),
                    })

                results.append({
                    'vuln_type': vuln_type,
                    'num_nodes': num_nodes,
                    'num_edges': num_edges // 2,  # bidirectional, so halve
                    'num_intra_edges': num_intra_edges // 2,
                    'num_inter_edges': num_inter_edges // 2,
                    'num_functions': num_funcs,
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

            num_nodes = batch.x.size(0)
            num_edges = batch.edge_index.size(1) // 2
            num_intra_edges = batch.sub_edge_index.size(1) // 2
            num_inter_edges = batch.inter_edge_index.size(1) // 2 if hasattr(batch, 'inter_edge_index') else 0

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

            func_labels = batch.y_func

            functions = []
            for j in range(num_funcs):
                functions.append({
                    'label': func_labels[j].item(),
                    'pred': func_preds[j].item(),
                    'prob': round(func_probs[j].item(), 4),
                })

            results.append({
                'vuln_type': vuln_type,
                'num_nodes': num_nodes,
                'num_edges': num_edges,
                'num_intra_edges': num_intra_edges,
                'num_inter_edges': num_inter_edges,
                'num_functions': num_funcs,
                'contract_label': contract_label,
                'contract_pred': contract_pred,
                'contract_prob': round(contract_prob, 4),
                'functions': functions,
            })

    return results


def compute_quartile_bins(predictions: List[Dict], key: str = 'num_nodes') -> List[Tuple[int, int]]:
    """Compute quartile-based bin boundaries for graph sizes."""
    values = sorted([p[key] for p in predictions])
    n = len(values)
    q25 = values[n // 4]
    q50 = values[n // 2]
    q75 = values[3 * n // 4]
    return [
        (0, q25),
        (q25 + 1, q50),
        (q50 + 1, q75),
        (q75 + 1, max(values)),
    ]


def compute_fixed_bins(predictions: List[Dict], key: str = 'num_nodes') -> List[Tuple[int, int, str]]:
    """Compute fixed, interpretable bins for graph sizes based on node count."""
    values = [p[key] for p in predictions]
    min_v, max_v = min(values), max(values)

    # Determine sensible bins based on the data range
    if max_v <= 500:
        boundaries = [0, 50, 100, 200, 500]
    elif max_v <= 2000:
        boundaries = [0, 100, 300, 700, 2000]
    else:
        boundaries = [0, 100, 300, 700, 1500, max_v + 1]

    bins = []
    for i in range(len(boundaries) - 1):
        lo, hi = boundaries[i], boundaries[i + 1] - 1
        label = f"{lo}--{hi}"
        bins.append((lo, hi, label))
    return bins


def safe_div(a, b):
    return round(a / b, 4) if b > 0 else 0.0


def compute_metrics_for_group(contracts: List[Dict]) -> Dict:
    """Compute detection metrics for a group of contracts."""
    n = len(contracts)
    if n == 0:
        return None

    # Contract-level
    c_tp = sum(1 for c in contracts if c['contract_label'] == 1 and c['contract_pred'] == 1)
    c_fp = sum(1 for c in contracts if c['contract_label'] == 0 and c['contract_pred'] == 1)
    c_fn = sum(1 for c in contracts if c['contract_label'] == 1 and c['contract_pred'] == 0)
    c_tn = sum(1 for c in contracts if c['contract_label'] == 0 and c['contract_pred'] == 0)

    # Function-level
    f_tp = f_fp = f_fn = f_tn = 0
    for c in contracts:
        for f in c['functions']:
            if f['label'] == 1 and f['pred'] == 1:
                f_tp += 1
            elif f['label'] == 0 and f['pred'] == 1:
                f_fp += 1
            elif f['label'] == 1 and f['pred'] == 0:
                f_fn += 1
            else:
                f_tn += 1

    c_prec = safe_div(c_tp, c_tp + c_fp)
    c_rec = safe_div(c_tp, c_tp + c_fn)
    c_f1 = safe_div(2 * c_prec * c_rec, c_prec + c_rec)
    c_acc = safe_div(c_tp + c_tn, n)

    f_prec = safe_div(f_tp, f_tp + f_fp)
    f_rec = safe_div(f_tp, f_tp + f_fn)
    f_f1 = safe_div(2 * f_prec * f_rec, f_prec + f_rec)
    f_acc = safe_div(f_tp + f_tn, f_tp + f_fp + f_fn + f_tn)

    # Structural stats
    nodes = [c['num_nodes'] for c in contracts]
    edges = [c['num_edges'] for c in contracts]
    funcs = [c['num_functions'] for c in contracts]

    return {
        'num_contracts': n,
        'num_vulnerable': sum(1 for c in contracts if c['contract_label'] == 1),
        'num_clean': sum(1 for c in contracts if c['contract_label'] == 0),
        'structural_stats': {
            'nodes_mean': round(statistics.mean(nodes), 1),
            'nodes_median': round(statistics.median(nodes), 1),
            'nodes_min': min(nodes),
            'nodes_max': max(nodes),
            'edges_mean': round(statistics.mean(edges), 1),
            'functions_mean': round(statistics.mean(funcs), 1),
        },
        'contract_level': {
            'TP': c_tp, 'FP': c_fp, 'FN': c_fn, 'TN': c_tn,
            'accuracy': c_acc,
            'precision': c_prec,
            'recall': c_rec,
            'f1': c_f1,
        },
        'function_level': {
            'TP': f_tp, 'FP': f_fp, 'FN': f_fn, 'TN': f_tn,
            'accuracy': f_acc,
            'precision': f_prec,
            'recall': f_rec,
            'f1': f_f1,
        },
    }


def compute_graph_size_analysis(predictions: List[Dict]) -> Dict:
    """
    Stratify predictions by graph size (number of ASG nodes) and compute
    detection metrics for each stratum.
    """
    if not predictions:
        return {}

    # Use quartile-based bins
    values = sorted([p['num_nodes'] for p in predictions])
    n = len(values)
    q25 = values[n // 4]
    q50 = values[n // 2]
    q75 = values[3 * n // 4]

    # Build bins with descriptive labels
    bins = [
        (0, q25, f"Q1 (≤{q25} nodes)"),
        (q25 + 1, q50, f"Q2 ({q25+1}--{q50} nodes)"),
        (q50 + 1, q75, f"Q3 ({q50+1}--{q75} nodes)"),
        (q75 + 1, max(values), f"Q4 ({q75+1}--{max(values)} nodes)"),
    ]

    result = {
        'overall': compute_metrics_for_group(predictions),
        'quartile_summary': {
            'Q1_max': q25, 'Q2_max': q50, 'Q3_max': q75, 'Q4_max': max(values),
        },
        'by_quartile': {},
    }

    for lo, hi, label in bins:
        group = [p for p in predictions if lo <= p['num_nodes'] <= hi]
        if group:
            result['by_quartile'][label] = compute_metrics_for_group(group)

    # Also compute by fixed bins for an alternative view
    fixed_bins = compute_fixed_bins(predictions)
    result['by_fixed_bin'] = {}
    for lo, hi, label in fixed_bins:
        group = [p for p in predictions if lo <= p['num_nodes'] <= hi]
        if group:
            result['by_fixed_bin'][label] = compute_metrics_for_group(group)

    return result


def print_graph_size_analysis(analysis: Dict, dataset_name: str):
    """Pretty-print graph-size stratified analysis."""
    print(f"\n{'='*100}")
    print(f"  Graph-Size Performance Analysis: {dataset_name}")
    print(f"{'='*100}")

    # Overall
    ov = analysis['overall']
    print(f"\n  Overall: {ov['num_contracts']} contracts ({ov['num_vulnerable']} vuln, {ov['num_clean']} clean)")
    print(f"    Contract-level: F1={ov['contract_level']['f1']:.4f}  Prec={ov['contract_level']['precision']:.4f}  Rec={ov['contract_level']['recall']:.4f}")
    print(f"    Function-level: F1={ov['function_level']['f1']:.4f}  Prec={ov['function_level']['precision']:.4f}  Rec={ov['function_level']['recall']:.4f}")

    # By quartile
    qs = analysis['quartile_summary']
    print(f"\n  Quartile boundaries: Q1≤{qs['Q1_max']}, Q2≤{qs['Q2_max']}, Q3≤{qs['Q3_max']}, Q4≤{qs['Q4_max']} nodes")

    header = f"  {'Bin':<30} {'#C':>5} {'#Vuln':>6} {'Nodes(μ)':>10} {'C-F1':>7} {'C-Prec':>8} {'C-Rec':>7} | {'F-F1':>7} {'F-Prec':>8} {'F-Rec':>7}"
    sep = f"  {'-'*30} {'-'*5} {'-'*6} {'-'*10} {'-'*7} {'-'*8} {'-'*7}---{'-'*7} {'-'*8} {'-'*7}"

    print(f"\n  --- By Quartile ---")
    print(header)
    print(sep)
    for label, m in analysis['by_quartile'].items():
        c = m['contract_level']
        f = m['function_level']
        s = m['structural_stats']
        print(f"  {label:<30} {m['num_contracts']:>5} {m['num_vulnerable']:>6} {s['nodes_mean']:>10.1f} {c['f1']:>7.4f} {c['precision']:>8.4f} {c['recall']:>7.4f} | {f['f1']:>7.4f} {f['precision']:>8.4f} {f['recall']:>7.4f}")

    print(f"\n  --- By Fixed Bin ---")
    print(header)
    print(sep)
    for label, m in analysis['by_fixed_bin'].items():
        c = m['contract_level']
        f = m['function_level']
        s = m['structural_stats']
        print(f"  {label:<30} {m['num_contracts']:>5} {m['num_vulnerable']:>6} {s['nodes_mean']:>10.1f} {c['f1']:>7.4f} {c['precision']:>8.4f} {c['recall']:>7.4f} | {f['f1']:>7.4f} {f['precision']:>8.4f} {f['recall']:>7.4f}")


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

    syn_predictions = predict_with_graph_info(model, syn_loader, device)
    syn_analysis = compute_graph_size_analysis(syn_predictions)
    print_graph_size_analysis(syn_analysis, "Synthetic Test Set")
    results['synthetic'] = syn_analysis

    # ---- Real-world test set ----
    print(f"\nLoading real-world test set from {realworld_data}...")
    irl_dataset = FunctionLevelDataset(root=realworld_data, bidirectional=True)
    irl_loader = DataLoader(irl_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"  Contracts: {len(irl_dataset)}")

    irl_predictions = predict_with_graph_info(model, irl_loader, device)
    irl_analysis = compute_graph_size_analysis(irl_predictions)
    print_graph_size_analysis(irl_analysis, "Real-World Test Set")
    results['realworld'] = irl_analysis

    # ---- Save results ----
    output_path = Path('results/graph_size_analysis.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
