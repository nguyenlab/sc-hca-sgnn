"""
Per-Vulnerability-Template Detection Metrics Generator

This script produces fine-grained detection metrics broken down by injection
template, going beyond the coarse vulnerability_type grouping.  In particular
it separates reentrancy contracts into Sereum-identified subtypes:
  * Same-Function Reentrancy  (call.value / send / transfer point templates)
  * Cross-Function Reentrancy (coupled templates with REENTRANCY vuln_type)
  * Delegated Reentrancy      (delegate_reentrancy point template)
  * Create-Based Reentrancy   (create_reentrancy point template)

Usage:
    python generate_per_template_metrics.py \
        [--checkpoint outputs/ablation_study/run_0/full/best_model.pt] \
        [--synthetic-data data/synthetic-split/test] \
        [--realworld-data data/realworld/function_level_dataset] \
        [--output results/per_template_metrics.json]
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from models.data import get_input_dim, get_num_edge_types
from models.function_level_data import FunctionLevelDataset
from models.subgraph_level.hierarchical_cross_attention import create_hca_sgnn_model
from training.utils import setup_device

# ---------- hyper-parameters (must match training) ----------
HIDDEN_DIM = 128
NUM_LAYERS = 4
HEADS = 8
DROPOUT = 0.2
BATCH_SIZE = 32


# ============================================================================
# Reentrancy sub-template identification
# ============================================================================

def identify_reentrancy_subtemplate(source_path: str) -> str:
    """
    Inspect the Solidity source file to determine which specific reentrancy
    point-injection template was used.

    Returns one of:
        call_value_legacy, call_value_050, call_value_modern,
        send_reentrancy,
        transfer_reentrancy_legacy, transfer_reentrancy,
        withdraw_reentrancy_legacy, withdraw_reentrancy_050,
        delegate_reentrancy, create_reentrancy,
        unknown_reentrancy
    """
    if not os.path.exists(source_path):
        return "unknown_reentrancy"

    with open(source_path, encoding="utf-8", errors="replace") as f:
        content = f.read()

    # --- Solidity version ---
    m = re.search(r"pragma solidity\s+[^;]*?(\d+)\.(\d+)\.(\d+)", content)
    if m:
        major, minor, patch = int(m.group(1)), int(m.group(2)), int(m.group(3))
        version = (major, minor, patch)
    else:
        version = (0, 4, 0)  # default

    has_delegatecall = "delegatecall" in content
    has_call_value = ".call.value(" in content or ".call{value:" in content
    has_send = ".send(" in content
    has_transfer = ".transfer(" in content

    # Priority: delegatecall > create > send > transfer > call.value
    if has_delegatecall:
        return "delegate_reentrancy"
    if has_send and not has_call_value and not has_transfer:
        return "send_reentrancy"
    if has_transfer and not has_call_value:
        if version < (0, 6, 0):
            return "transfer_reentrancy_legacy"
        else:
            return "transfer_reentrancy"
    if has_call_value:
        if version < (0, 5, 0):
            return "call_value_legacy"
        elif version < (0, 7, 0):
            return "call_value_050"
        else:
            return "call_value_modern"

    return "unknown_reentrancy"


def map_to_sereum_category(subtemplate: str) -> str:
    """Map a reentrancy sub-template to a Sereum category."""
    SEREUM_MAP = {
        "call_value_legacy": "Same-Function",
        "call_value_050": "Same-Function",
        "call_value_modern": "Same-Function",
        "send_reentrancy": "Same-Function",
        "transfer_reentrancy_legacy": "Same-Function",
        "transfer_reentrancy": "Same-Function",
        "withdraw_reentrancy_legacy": "Same-Function",
        "withdraw_reentrancy_050": "Same-Function",
        "delegate_reentrancy": "Delegated",
        "create_reentrancy": "Create-Based",
        "unknown_reentrancy": "Same-Function",  # default
        # Coupled reentrancy templates
        "coupled_reentrancy": "Cross-Function",
    }
    return SEREUM_MAP.get(subtemplate, "Unknown")


# ============================================================================
# Extract injection template from filename / contract_path
# ============================================================================

def extract_injection_template(filename: str, contract_path: str = "",
                                vuln_type: str = "clean") -> Tuple[str, str]:
    """
    Return (template_category, template_detail) by parsing the filename.

    template_category  – broad group (clean, point_reentrancy, coupled, etc.)
    template_detail    – finer detail when available
    """
    name = filename.replace(".json", "")

    if "_point_" in name:
        parts = name.split("_point_")[1]
        # Remove the numeric ID
        parts2 = parts.split("_", 1)
        if len(parts2) > 1:
            vtype = parts2[1]
        else:
            vtype = vuln_type
        category = f"point_{vtype}"
        return category, category

    if "_coupled_" in name:
        # All coupled files have vuln_type == "cross-function"
        return "coupled", vuln_type

    return "clean", "clean"


# ============================================================================
# Model loading
# ============================================================================

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
        pooling="mean",
        use_rgcn=False,
        use_dual_channel=True,
        use_gated_fusion=True,
        use_node_attention=False,
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model


# ============================================================================
# Inference
# ============================================================================

@torch.no_grad()
def predict_per_contract(model: nn.Module, loader: DataLoader,
                          device: torch.device) -> List[Dict[str, Any]]:
    """Run model on each contract and return per-contract predictions
    with vulnerability type metadata."""
    results = []

    for batch in loader:
        batch = batch.to(device)

        if hasattr(batch, "ptr") and batch.ptr is not None and len(batch.ptr) > 2:
            for i in range(len(batch.ptr) - 1):
                start, end = batch.ptr[i].item(), batch.ptr[i + 1].item()

                node_x = batch.x[start:end]
                node_subg = batch.subg[start:end]

                edge_mask = (batch.edge_index[0] >= start) & (batch.edge_index[0] < end)
                graph_edge_index = batch.edge_index[:, edge_mask] - start

                sub_edge_mask = (batch.sub_edge_index[0] >= start) & (
                    batch.sub_edge_index[0] < end
                )
                graph_sub_edge_index = batch.sub_edge_index[:, sub_edge_mask] - start

                num_funcs = batch.num_functions[i].item()

                func_logits = model(
                    node_x, graph_edge_index, graph_sub_edge_index,
                    node_subg, num_funcs,
                )

                func_probs = torch.softmax(func_logits, dim=-1)[:, 1]
                func_preds = func_logits.argmax(dim=-1)

                contract_pred = 1 if func_preds.max().item() > 0 else 0
                contract_prob = func_probs.max().item()
                contract_label = batch.graph_y[i].item()

                # Function labels
                func_start = sum(batch.num_functions[:i].tolist()) if i > 0 else 0
                func_end = func_start + num_funcs
                func_labels = batch.y_func[func_start:func_end]

                vuln_type = (
                    batch.vuln_type[i]
                    if hasattr(batch, "vuln_type") and isinstance(batch.vuln_type, list)
                    else "unknown"
                )

                func_names = (
                    batch.function_names[i]
                    if hasattr(batch, "function_names")
                    and isinstance(batch.function_names, list)
                    else [f"func_{j}" for j in range(num_funcs)]
                )

                contract_path = ""
                if hasattr(batch, "contract_path"):
                    if isinstance(batch.contract_path, list):
                        contract_path = batch.contract_path[i]
                    elif isinstance(batch.contract_path, str):
                        contract_path = batch.contract_path

                functions = []
                for j in range(num_funcs):
                    functions.append({
                        "name": func_names[j] if j < len(func_names) else f"func_{j}",
                        "label": func_labels[j].item(),
                        "pred": func_preds[j].item(),
                        "prob": round(func_probs[j].item(), 4),
                    })

                results.append({
                    "contract_path": contract_path,
                    "vuln_type": vuln_type,
                    "contract_label": contract_label,
                    "contract_pred": contract_pred,
                    "contract_prob": round(contract_prob, 4),
                    "functions": functions,
                })
        else:
            num_funcs = batch.num_functions
            if isinstance(num_funcs, torch.Tensor):
                num_funcs = num_funcs.sum().item() if num_funcs.dim() > 0 else num_funcs.item()

            func_logits = model(
                batch.x, batch.edge_index, batch.sub_edge_index,
                batch.subg, num_funcs,
            )

            func_probs = torch.softmax(func_logits, dim=-1)[:, 1]
            func_preds = func_logits.argmax(dim=-1)

            contract_pred = 1 if func_preds.max().item() > 0 else 0
            contract_prob = func_probs.max().item()
            contract_label = (
                batch.graph_y.item() if batch.graph_y.dim() == 0
                else batch.graph_y[0].item()
            )

            vuln_type = (
                batch.vuln_type
                if hasattr(batch, "vuln_type") and isinstance(batch.vuln_type, str)
                else "unknown"
            )

            contract_path = ""
            if hasattr(batch, "contract_path"):
                if isinstance(batch.contract_path, str):
                    contract_path = batch.contract_path
                elif isinstance(batch.contract_path, list):
                    contract_path = batch.contract_path[0] if batch.contract_path else ""

            func_names_data = (
                batch.function_names
                if hasattr(batch, "function_names")
                else [f"func_{j}" for j in range(num_funcs)]
            )
            if (isinstance(func_names_data, list) and len(func_names_data) > 0
                    and isinstance(func_names_data[0], list)):
                func_names_data = func_names_data[0]

            func_labels = batch.y_func

            functions = []
            for j in range(num_funcs):
                functions.append({
                    "name": func_names_data[j] if j < len(func_names_data) else f"func_{j}",
                    "label": func_labels[j].item(),
                    "pred": func_preds[j].item(),
                    "prob": round(func_probs[j].item(), 4),
                })

            results.append({
                "contract_path": contract_path,
                "vuln_type": vuln_type,
                "contract_label": contract_label,
                "contract_pred": contract_pred,
                "contract_prob": round(contract_prob, 4),
                "functions": functions,
            })

    return results


# ============================================================================
# Per-template metrics computation
# ============================================================================

def safe_div(a: float, b: float) -> float:
    return round(a / b, 4) if b > 0 else 0.0


def compute_metrics_for_group(contracts: List[Dict]) -> Dict[str, Any]:
    """Compute contract-level and function-level metrics for a group."""
    n = len(contracts)

    # Contract-level
    c_tp = sum(1 for c in contracts if c["contract_label"] == 1 and c["contract_pred"] == 1)
    c_fp = sum(1 for c in contracts if c["contract_label"] == 0 and c["contract_pred"] == 1)
    c_fn = sum(1 for c in contracts if c["contract_label"] == 1 and c["contract_pred"] == 0)
    c_tn = sum(1 for c in contracts if c["contract_label"] == 0 and c["contract_pred"] == 0)

    c_prec = safe_div(c_tp, c_tp + c_fp)
    c_rec = safe_div(c_tp, c_tp + c_fn)
    c_f1 = safe_div(2 * c_prec * c_rec, c_prec + c_rec)

    # Function-level
    f_tp = f_fp = f_fn = f_tn = 0
    for c in contracts:
        for fun in c["functions"]:
            if fun["label"] == 1 and fun["pred"] == 1:
                f_tp += 1
            elif fun["label"] == 0 and fun["pred"] == 1:
                f_fp += 1
            elif fun["label"] == 1 and fun["pred"] == 0:
                f_fn += 1
            else:
                f_tn += 1

    f_prec = safe_div(f_tp, f_tp + f_fp)
    f_rec = safe_div(f_tp, f_tp + f_fn)
    f_f1 = safe_div(2 * f_prec * f_rec, f_prec + f_rec)

    return {
        "num_contracts": n,
        "contract_level": {
            "TP": c_tp, "FP": c_fp, "FN": c_fn, "TN": c_tn,
            "precision": c_prec, "recall": c_rec, "f1": c_f1,
        },
        "function_level": {
            "TP": f_tp, "FP": f_fp, "FN": f_fn, "TN": f_tn,
            "precision": f_prec, "recall": f_rec, "f1": f_f1,
        },
    }


def compute_per_template_metrics(
    predictions: List[Dict],
    data_root: str = "",
) -> Dict[str, Any]:
    """
    Compute detection metrics grouped by injection template.

    For reentrancy contracts, we additionally identify the specific
    sub-template by inspecting source files.
    """
    by_template: Dict[str, List[Dict]] = defaultdict(list)
    by_reentrancy_sub: Dict[str, List[Dict]] = defaultdict(list)
    by_sereum: Dict[str, List[Dict]] = defaultdict(list)

    for pred in predictions:
        contract_path = pred.get("contract_path", "")
        filename = os.path.basename(contract_path).replace(".sol", "")
        vuln_type = pred["vuln_type"]

        category, detail = extract_injection_template(filename, contract_path, vuln_type)
        by_template[category].append(pred)

        # --- Reentrancy-specific sub-template analysis ---
        if category == "point_reentrancy":
            src_path = os.path.join(data_root, contract_path) if data_root else contract_path
            if not os.path.exists(src_path):
                # try relative to cwd
                src_path = contract_path
            sub = identify_reentrancy_subtemplate(src_path)
            by_reentrancy_sub[sub].append(pred)
            sereum = map_to_sereum_category(sub)
            by_sereum[sereum].append(pred)

        elif category == "coupled" and vuln_type == "cross-function":
            # All coupled cross-function templates are reentrancy-related
            # (the coupled templates include reentrancy variants + TOD/DOS/etc.)
            # We'll mark all coupled as "Cross-Function" in the Sereum sense
            by_reentrancy_sub["coupled_reentrancy"].append(pred)
            by_sereum["Cross-Function"].append(pred)

    # Compute metrics for each template group
    template_metrics = {}
    for tpl, contracts in sorted(by_template.items(), key=lambda x: -len(x[1])):
        template_metrics[tpl] = compute_metrics_for_group(contracts)

    # Reentrancy sub-template metrics
    reentrancy_sub_metrics = {}
    for sub, contracts in sorted(by_reentrancy_sub.items(), key=lambda x: -len(x[1])):
        reentrancy_sub_metrics[sub] = compute_metrics_for_group(contracts)

    # Sereum category metrics
    sereum_metrics = {}
    for cat, contracts in sorted(by_sereum.items()):
        sereum_metrics[cat] = compute_metrics_for_group(contracts)

    return {
        "by_template": template_metrics,
        "reentrancy_subtemplates": reentrancy_sub_metrics,
        "sereum_categories": sereum_metrics,
    }


# ============================================================================
# Pretty printing
# ============================================================================

def print_template_results(metrics: Dict, dataset_name: str):
    print(f"\n{'=' * 90}")
    print(f"  Per-Template Detection Metrics: {dataset_name}")
    print(f"{'=' * 90}")

    # --- By injection template ---
    by_tpl = metrics["by_template"]
    print(f"\n  {'Template':<28} {'#C':>4} {'C-TP':>5} {'C-FP':>5} {'C-FN':>5} "
          f"{'C-Prec':>7} {'C-Rec':>6} {'C-F1':>6} | "
          f"{'F-TP':>5} {'F-FP':>5} {'F-FN':>5} {'F-Prec':>7} {'F-Rec':>6} {'F-F1':>6}")
    print(f"  {'-' * 28} {'-' * 4} {'-' * 5} {'-' * 5} {'-' * 5} "
          f"{'-' * 7} {'-' * 6} {'-' * 6}   "
          f"{'-' * 5} {'-' * 5} {'-' * 5} {'-' * 7} {'-' * 6} {'-' * 6}")

    for tpl, m in sorted(by_tpl.items(), key=lambda x: -x[1]["num_contracts"]):
        c = m["contract_level"]
        f = m["function_level"]
        print(
            f"  {tpl:<28} {m['num_contracts']:>4} "
            f"{c['TP']:>5} {c['FP']:>5} {c['FN']:>5} "
            f"{c['precision']:>7.4f} {c['recall']:>6.4f} {c['f1']:>6.4f} | "
            f"{f['TP']:>5} {f['FP']:>5} {f['FN']:>5} "
            f"{f['precision']:>7.4f} {f['recall']:>6.4f} {f['f1']:>6.4f}"
        )

    # --- Reentrancy sub-templates ---
    reen_sub = metrics.get("reentrancy_subtemplates", {})
    if reen_sub:
        print(f"\n  --- Reentrancy Sub-Template Breakdown ---")
        print(f"  {'Sub-Template':<28} {'#C':>4} {'C-Rec':>6} {'C-F1':>6} | {'F-TP':>5} {'F-FN':>5} {'F-Rec':>6} {'F-F1':>6}")
        print(f"  {'-' * 28} {'-' * 4} {'-' * 6} {'-' * 6}   {'-' * 5} {'-' * 5} {'-' * 6} {'-' * 6}")
        for sub, m in sorted(reen_sub.items(), key=lambda x: -x[1]["num_contracts"]):
            c = m["contract_level"]
            f = m["function_level"]
            print(
                f"  {sub:<28} {m['num_contracts']:>4} "
                f"{c['recall']:>6.4f} {c['f1']:>6.4f} | "
                f"{f['TP']:>5} {f['FN']:>5} {f['recall']:>6.4f} {f['f1']:>6.4f}"
            )

    # --- Sereum categories ---
    sereum = metrics.get("sereum_categories", {})
    if sereum:
        print(f"\n  --- Sereum Reentrancy Category Summary ---")
        print(f"  {'Category':<24} {'#C':>4} {'C-Prec':>7} {'C-Rec':>6} {'C-F1':>6} | {'F-Prec':>7} {'F-Rec':>6} {'F-F1':>6}")
        print(f"  {'-' * 24} {'-' * 4} {'-' * 7} {'-' * 6} {'-' * 6}   {'-' * 7} {'-' * 6} {'-' * 6}")
        for cat, m in sorted(sereum.items()):
            c = m["contract_level"]
            f = m["function_level"]
            print(
                f"  {cat:<24} {m['num_contracts']:>4} "
                f"{c['precision']:>7.4f} {c['recall']:>6.4f} {c['f1']:>6.4f} | "
                f"{f['precision']:>7.4f} {f['recall']:>6.4f} {f['f1']:>6.4f}"
            )


def print_realworld_results(metrics: Dict, dataset_name: str):
    """Print real-world results (no sub-template analysis)."""
    print(f"\n{'=' * 90}")
    print(f"  Per-Type Detection Metrics: {dataset_name}")
    print(f"{'=' * 90}")

    by_tpl = metrics["by_template"]
    print(f"\n  {'Type':<25} {'#C':>4} {'C-TP':>5} {'C-FP':>5} {'C-FN':>5} "
          f"{'C-Prec':>7} {'C-Rec':>6} {'C-F1':>6} | "
          f"{'F-TP':>5} {'F-FP':>5} {'F-FN':>5} {'F-Prec':>7} {'F-Rec':>6} {'F-F1':>6}")
    print(f"  {'-' * 25} {'-' * 4} {'-' * 5} {'-' * 5} {'-' * 5} "
          f"{'-' * 7} {'-' * 6} {'-' * 6}   "
          f"{'-' * 5} {'-' * 5} {'-' * 5} {'-' * 7} {'-' * 6} {'-' * 6}")

    for tpl, m in sorted(by_tpl.items(), key=lambda x: -x[1]["num_contracts"]):
        c = m["contract_level"]
        f = m["function_level"]
        print(
            f"  {tpl:<25} {m['num_contracts']:>4} "
            f"{c['TP']:>5} {c['FP']:>5} {c['FN']:>5} "
            f"{c['precision']:>7.4f} {c['recall']:>6.4f} {c['f1']:>6.4f} | "
            f"{f['TP']:>5} {f['FP']:>5} {f['FN']:>5} "
            f"{f['precision']:>7.4f} {f['recall']:>6.4f} {f['f1']:>6.4f}"
        )


# ============================================================================
# Real-world: group by vuln_type directly
# ============================================================================

def compute_realworld_per_type_metrics(predictions: List[Dict]) -> Dict[str, Any]:
    """For real-world data, group by vuln_type directly (no template encoding)."""
    by_type: Dict[str, List[Dict]] = defaultdict(list)
    for pred in predictions:
        vtype = pred["vuln_type"]
        by_type[vtype].append(pred)

    metrics = {}
    for vtype, contracts in sorted(by_type.items(), key=lambda x: -len(x[1])):
        metrics[vtype] = compute_metrics_for_group(contracts)

    return {"by_template": metrics}


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate per-vulnerability-template detection metrics.",
    )
    parser.add_argument(
        "--checkpoint", type=str,
        default="outputs/ablation_study/run_0/full/best_model.pt",
    )
    parser.add_argument(
        "--synthetic-data", type=str,
        default="data/synthetic-split/test",
    )
    parser.add_argument(
        "--realworld-data", type=str,
        default="data/realworld/function_level_dataset",
    )
    parser.add_argument(
        "--output", type=str,
        default="results/per_template_metrics.json",
    )
    args = parser.parse_args()

    device = setup_device()
    print(f"Device: {device}")

    # Load model
    print(f"Loading model from {args.checkpoint} ...")
    model = load_model(args.checkpoint, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    all_results: Dict[str, Any] = {}

    # ---- Synthetic test set ----
    syn_path = args.synthetic_data
    print(f"\nLoading synthetic test set from {syn_path} ...")
    syn_dataset = FunctionLevelDataset(root=syn_path, bidirectional=True)
    syn_loader = DataLoader(syn_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"  Contracts: {len(syn_dataset)}")

    syn_preds = predict_per_contract(model, syn_loader, device)
    syn_metrics = compute_per_template_metrics(syn_preds, data_root="")
    print_template_results(syn_metrics, "Synthetic Test Set")
    all_results["synthetic"] = syn_metrics

    # ---- Real-world test set ----
    irl_path = args.realworld_data
    if Path(irl_path).exists():
        print(f"\nLoading real-world test set from {irl_path} ...")
        irl_dataset = FunctionLevelDataset(root=irl_path, bidirectional=True)
        irl_loader = DataLoader(irl_dataset, batch_size=BATCH_SIZE, shuffle=False)
        print(f"  Contracts: {len(irl_dataset)}")

        irl_preds = predict_per_contract(model, irl_loader, device)
        irl_metrics = compute_realworld_per_type_metrics(irl_preds)
        print_realworld_results(irl_metrics, "Real-World Test Set")
        all_results["realworld"] = irl_metrics
    else:
        print(f"\nReal-world data not found at {irl_path}, skipping.")

    # ---- Save ----
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
