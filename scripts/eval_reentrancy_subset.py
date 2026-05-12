#!/usr/bin/env python3
"""
Evaluate all benchmarked contract-level baseline models on the
reentrancy-only subset of both the synthetic and real-world test sets.

Loads each saved checkpoint from outputs/benchmark/contract_level/<model>/best_model.pt
and runs inference on ContractGraphDataset filtered to ['reentrancy', 'clean'].

Output: results/baselines/reentrancy_subset_eval.json
"""
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from torch.utils.data import Subset
from collections import defaultdict

from models.function_level_data import FunctionLevelDataset
from models.subgraph_level.hierarchical_cross_attention import create_hca_sgnn_model

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
    get_input_dim,
    get_num_classes,
    get_num_edge_types,
)
from models.bsgvd_data import BSGVDDataset
from models.data import VULN_TO_IDX, IDX_TO_VULN
from training.utils import setup_device, setup_seed

# Mirror MODEL_CONFIGS from benchmark.py
MODEL_CONFIGS = {
    'hierarchical-rgcn': {
        'name': 'Hierarchical RGCN',
        'factory': lambda input_dim, hidden_dim, num_layers, heads, num_classes, dropout, num_edge_types:
            create_model('hierarchical', input_dim=input_dim, hidden_dim=hidden_dim,
                         num_layers=num_layers, dropout=dropout, num_classes=num_classes,
                         num_edge_types=num_edge_types, mode='rgcn'),
    },
    'dr-gcn': {
        'name': 'DR-GCN',
        'factory': lambda input_dim, hidden_dim, num_layers, heads, num_classes, dropout, num_edge_types:
            create_dr_gcn_model(input_dim=input_dim, hidden_dim=hidden_dim,
                                num_layers=num_layers, num_classes=num_classes, dropout=dropout),
    },
    'tmp': {
        'name': 'TMP',
        'factory': lambda input_dim, hidden_dim, num_layers, heads, num_classes, dropout, num_edge_types:
            create_tmp_model(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers,
                             heads=heads, num_classes=num_classes, dropout=dropout),
    },
    'gat': {
        'name': 'GAT',
        'factory': lambda input_dim, hidden_dim, num_layers, heads, num_classes, dropout, num_edge_types:
            create_gat_model(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers,
                             heads=heads, num_classes=num_classes, dropout=dropout),
    },
    'transformer': {
        'name': 'TransformerGNN',
        'factory': lambda input_dim, hidden_dim, num_layers, heads, num_classes, dropout, num_edge_types:
            create_transformer_gnn_model(input_dim=input_dim, hidden_dim=hidden_dim,
                                          num_layers=num_layers, heads=heads,
                                          num_classes=num_classes, dropout=dropout,
                                          use_positional_encoding=True),
    },
    'bugsweeper': {
        'name': 'BugSweeper',
        'factory': lambda input_dim, hidden_dim, num_layers, heads, num_classes, dropout, num_edge_types:
            create_bugsweeper_model(input_dim=input_dim, num_classes=num_classes, dropout=dropout),
    },
    'bugsweeper-light': {
        'name': 'BugSweeperLight',
        'factory': lambda input_dim, hidden_dim, num_layers, heads, num_classes, dropout, num_edge_types:
            create_bugsweeper_light_model(input_dim=input_dim, hidden_dim=hidden_dim, num_classes=num_classes, dropout=dropout),
    },
    'scvhunter': {
        'name': 'SCVHUNTER',
        'factory': lambda input_dim, hidden_dim, num_layers, heads, num_classes, dropout, num_edge_types:
            create_scvhunter_model(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers,
                                    heads=heads, num_classes=num_classes, dropout=dropout,
                                    num_edge_types=num_edge_types),
    },
    'mlagnn': {
        'name': 'ML-AGNN',
        'factory': lambda input_dim, hidden_dim, num_layers, heads, num_classes, dropout, num_edge_types:
            create_mlagnn_model(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers,
                                 heads=heads, num_classes=num_classes, dropout=dropout),
    },
    'bsgvd': {
        'name': 'BSGVD',
        'factory': lambda input_dim, hidden_dim, num_layers, heads, num_classes, dropout, num_edge_types:
            create_bsgvd_model(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers,
                               heads=heads, num_classes=num_classes, dropout=dropout),
    },
}

# Hyperparameters used during the original benchmark run
HIDDEN_DIM = 128
NUM_LAYERS = 4
HEADS = 4
DROPOUT = 0.2
BATCH_SIZE = 32
SEED = 42


def compute_metrics(tp, fp, fn, tn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy  = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0.0
    return {'precision': precision, 'recall': recall, 'f1': f1, 'accuracy': accuracy,
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn}


@torch.no_grad()
def evaluate_on_loader(model, loader, device):
    model.eval()
    tp = fp = fn = tn = 0
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch.x, batch.edge_index,
                       batch.edge_attr if hasattr(batch, 'edge_attr') else None,
                       batch.batch)
        preds = logits.argmax(dim=1)
        labels = (batch.graph_y.squeeze() > 0).long()
        tp += ((preds == 1) & (labels == 1)).sum().item()
        fp += ((preds == 1) & (labels == 0)).sum().item()
        fn += ((preds == 0) & (labels == 1)).sum().item()
        tn += ((preds == 0) & (labels == 0)).sum().item()
    return compute_metrics(tp, fp, fn, tn)


def load_model(model_key, checkpoint_path, device, input_dim, num_edge_types):
    config = MODEL_CONFIGS[model_key]
    model = config['factory'](
        input_dim=input_dim,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        heads=HEADS,
        num_classes=2,
        dropout=DROPOUT,
        num_edge_types=num_edge_types,
    )
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    return model


def main():
    setup_seed(SEED)
    device = setup_device()
    print(f"Device: {device}")

    input_dim     = get_input_dim()
    num_edge_types = get_num_edge_types()
    ckpt_base = ROOT / 'outputs' / 'benchmark' / 'contract_level'

    # Load reentrancy-only test sets
    print("\nLoading synthetic reentrancy-only test set...")
    syn_ds = ContractGraphDataset(
        root=str(ROOT / 'data' / 'synthetic-split' / 'test'),
        use_edge_attr=True, bidirectional=True,
        vuln_filter=['reentrancy', 'cross-function', 'clean'],
    )
    syn_loader = DataLoader(syn_ds, batch_size=BATCH_SIZE, shuffle=False)
    print(f"  {len(syn_ds)} contracts (reentrancy + cross-function + clean)")

    print("Loading real-world reentrancy-only test set...")
    rw_ds = ContractGraphDataset(
        root=str(ROOT / 'data' / 'realworld' / 'ast_dataset'),
        use_edge_attr=True, bidirectional=True,
        vuln_filter=['reentrancy', 'clean'],
    )
    rw_loader = DataLoader(rw_ds, batch_size=BATCH_SIZE, shuffle=False)
    print(f"  {len(rw_ds)} contracts (reentrancy + clean)")

    results = {}

    for model_key, config in MODEL_CONFIGS.items():
        ckpt = ckpt_base / model_key / 'best_model.pt'
        if not ckpt.exists():
            print(f"\n[{config['name']}] checkpoint not found at {ckpt}, skipping")
            continue

        if model_key == 'bsgvd':
            print(f"\n[BSGVD] loading checkpoint {ckpt}")
            try:
                ft_path = ROOT / 'models' / 'fasttext' / 'solidity.model'
                bsgvd_syn_ds = BSGVDDataset(
                    graph_dir=ROOT / 'data' / 'synthetic-split' / 'test' / 'graph_data',
                    ast_data_dir=ROOT / 'data' / 'synthetic' / 'ast_dataset' / 'ast_data',
                    fasttext_model_path=ft_path if ft_path.exists() else None,
                    use_fasttext=ft_path.exists(),
                )
                re_types_idx = {VULN_TO_IDX['reentrancy'], VULN_TO_IDX['cross-function'], VULN_TO_IDX['clean']}
                syn_idx = [i for i in range(len(bsgvd_syn_ds))
                           if bsgvd_syn_ds.get(i).graph_y.item() in re_types_idx]
                bsgvd_syn_loader = DataLoader(Subset(bsgvd_syn_ds, syn_idx), batch_size=BATCH_SIZE)

                bsgvd_rw_ds = BSGVDDataset(
                    graph_dir=ROOT / 'data' / 'realworld' / 'ast_dataset' / 'graph_data',
                    ast_data_dir=ROOT / 'data' / 'realworld' / 'ast_dataset' / 'ast_data',
                    fasttext_model_path=ft_path if ft_path.exists() else None,
                    use_fasttext=ft_path.exists(),
                )
                re_rw_idx = {VULN_TO_IDX['reentrancy'], VULN_TO_IDX['clean']}
                rw_idx = [i for i in range(len(bsgvd_rw_ds))
                          if bsgvd_rw_ds.get(i).graph_y.item() in re_rw_idx]
                bsgvd_rw_loader = DataLoader(Subset(bsgvd_rw_ds, rw_idx), batch_size=BATCH_SIZE)

                bsgvd_input_dim = bsgvd_syn_ds.feature_dim
                print(f"  BSGVD feature_dim: {bsgvd_input_dim}")
                bsgvd_model = MODEL_CONFIGS['bsgvd']['factory'](
                    input_dim=bsgvd_input_dim, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS,
                    heads=HEADS, num_classes=2, dropout=DROPOUT, num_edge_types=None,
                )
                state = torch.load(ckpt, map_location=device)
                bsgvd_model.load_state_dict(state)
                bsgvd_model = bsgvd_model.to(device).eval()

                syn_m = evaluate_on_loader(bsgvd_model, bsgvd_syn_loader, device)
                rw_m  = evaluate_on_loader(bsgvd_model, bsgvd_rw_loader,  device)
                print(f"  Synthetic  P={syn_m['precision']:.4f} R={syn_m['recall']:.4f} F1={syn_m['f1']:.4f}")
                print(f"  Real-world P={rw_m['precision']:.4f}  R={rw_m['recall']:.4f}  F1={rw_m['f1']:.4f}")
                results['bsgvd'] = {'name': 'BSGVD', 'synthetic_reentrancy': syn_m, 'realworld_reentrancy': rw_m}
            except Exception as e:
                print(f"  ERROR: {e}")
                results['bsgvd'] = {'name': 'BSGVD', 'error': str(e)}
            continue

        print(f"\n[{config['name']}] loading {ckpt}")
        try:
            model = load_model(model_key, ckpt, device, input_dim, num_edge_types)
        except Exception as e:
            print(f"  ERROR loading model: {e}")
            results[model_key] = {'name': config['name'], 'error': str(e)}
            continue

        syn_metrics = evaluate_on_loader(model, syn_loader, device)
        rw_metrics  = evaluate_on_loader(model, rw_loader,  device)

        print(f"  Synthetic  P={syn_metrics['precision']:.4f} R={syn_metrics['recall']:.4f} F1={syn_metrics['f1']:.4f}")
        print(f"  Real-world P={rw_metrics['precision']:.4f}  R={rw_metrics['recall']:.4f}  F1={rw_metrics['f1']:.4f}")

        results[model_key] = {
            'name': config['name'],
            'synthetic_reentrancy': syn_metrics,
            'realworld_reentrancy': rw_metrics,
        }

    # -------------------------------------------------------------------------
    # HCA-SGNN (function-level, aggregated to contract level with OR rule)
    # -------------------------------------------------------------------------
    hca_ckpt = ROOT / 'outputs' / 'benchmark' / 'function_level' / 'func-hca' / 'best_model.pt'
    if hca_ckpt.exists():
        print(f"\n[HCA-SGNN] loading {hca_ckpt}")
        try:
            hca_model = create_hca_sgnn_model(
                input_dim=get_input_dim(), hidden_dim=128, num_layers=3,
                num_edge_types=get_num_edge_types(), heads=8, num_classes=2,
                dropout=0.2, use_cross_attention=True, pooling='hierarchical',
                use_rgcn=False, use_dual_channel=True, use_gated_fusion=True,
            )
            hca_model.load_state_dict(torch.load(hca_ckpt, map_location=device))
            hca_model = hca_model.to(device).eval()

            def eval_hca_on_root(root_dir, vuln_types):
                ds = FunctionLevelDataset(
                    root=str(root_dir),
                    vuln_filter=list(vuln_types),
                )
                loader = DataLoader(ds, batch_size=1, shuffle=False)
                tp = fp = fn = tn = 0
                with torch.no_grad():
                    for batch in loader:
                        batch = batch.to(device)
                        for i in range(len(batch.ptr) - 1):
                            start, end = batch.ptr[i].item(), batch.ptr[i+1].item()
                            node_x = batch.x[start:end]
                            edges = batch.edge_index
                            mask = (edges[0] >= start) & (edges[0] < end)
                            graph_edge_index = edges[:, mask] - start
                            sub_edges = batch.sub_edge_index
                            smask = (sub_edges[0] >= start) & (sub_edges[0] < end)
                            graph_sub_edge_index = sub_edges[:, smask] - start
                            node_subg = batch.subg[start:end]
                            num_funcs = batch.num_functions[i].item()
                            func_logits = hca_model(node_x, graph_edge_index,
                                                    graph_sub_edge_index, node_subg, num_funcs)
                            contract_pred = 1 if func_logits.argmax(dim=-1).max().item() > 0 else 0
                            contract_label = 1 if batch.graph_y[i].item() > 0 else 0
                            if contract_label==1 and contract_pred==1: tp+=1
                            elif contract_label==0 and contract_pred==1: fp+=1
                            elif contract_label==1 and contract_pred==0: fn+=1
                            else: tn+=1
                return compute_metrics(tp, fp, fn, tn)

            hca_syn = eval_hca_on_root(ROOT / 'data' / 'synthetic-split' / 'test',
                                        {'reentrancy','cross-function','clean'})
            hca_rw  = eval_hca_on_root(ROOT / 'data' / 'realworld' / 'function_level_dataset',
                                        {'reentrancy','clean'})
            print(f"  Synthetic  P={hca_syn['precision']:.4f} R={hca_syn['recall']:.4f} F1={hca_syn['f1']:.4f}")
            print(f"  Real-world P={hca_rw['precision']:.4f}  R={hca_rw['recall']:.4f}  F1={hca_rw['f1']:.4f}")
            results['hca-sgnn'] = {'name':'HCA-SGNN (ours)', 'synthetic_reentrancy': hca_syn, 'realworld_reentrancy': hca_rw}
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()
    else:
        print(f"\n[HCA-SGNN] checkpoint not found at {hca_ckpt}")

    # Save
    out_path = ROOT / 'results' / 'baselines' / 'reentrancy_subset_eval.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")

    # Quick summary table
    print("\n=== SUMMARY (reentrancy subset) ===")
    print(f"{'Model':25s}  {'Synth F1':>10}  {'Real F1':>10}")
    print("-" * 50)
    for k, v in results.items():
        if 'error' in v:
            print(f"{v['name']:25s}  ERROR")
        else:
            print(f"{v['name']:25s}  {v['synthetic_reentrancy']['f1']:10.4f}  {v['realworld_reentrancy']['f1']:10.4f}")


if __name__ == '__main__':
    main()
