import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# Edge type names (matching dataset_builder.py / function_level_builder.py)
EDGE_TYPE_NAMES = {
    0: "AST",
    1: "REF",
    2: "CFG_NEXT",
    3: "CFG_TRUE",
    4: "CFG_FALSE",
    5: "CALL",
    6: "INHERIT",
    7: "GUARD",
}

NUM_EDGE_TYPES = 8


def compute_distribution_stats(values: List[float]) -> Dict[str, float]:
    """Compute min, max, mean, median, std, Q1, Q3 for a list of values."""
    if not values:
        return {"count": 0, "min": 0, "max": 0, "mean": 0, "median": 0,
                "std": 0, "q1": 0, "q3": 0, "sum": 0}
    arr = np.array(values, dtype=float)
    return {
        "count": len(arr),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "q1": float(np.percentile(arr, 25)),
        "q3": float(np.percentile(arr, 75)),
        "sum": float(np.sum(arr)),
    }


def count_source_loc(source_path: Path) -> Optional[int]:
    """Count non-empty, non-comment lines of Solidity source code."""
    if not source_path.exists():
        return None
    try:
        with open(source_path, "r", errors="ignore") as f:
            lines = f.readlines()

        loc = 0
        in_block_comment = False
        for line in lines:
            stripped = line.strip()

            # Handle block comments
            if in_block_comment:
                if "*/" in stripped:
                    in_block_comment = False
                continue

            if stripped.startswith("/*"):
                if "*/" not in stripped:
                    in_block_comment = True
                continue

            # Skip empty lines and single-line comments
            if not stripped or stripped.startswith("//"):
                continue

            loc += 1
        return loc
    except Exception:
        return None


def count_total_lines(source_path: Path) -> Optional[int]:
    """Count total lines (including empty/comments) of source file."""
    if not source_path.exists():
        return None
    try:
        with open(source_path, "r", errors="ignore") as f:
            return sum(1 for _ in f)
    except Exception:
        return None


def count_function_loc(source_bytes: bytes, start_byte: int, end_byte: int) -> int:
    """Count lines of code within a function's byte range."""
    snippet = source_bytes[start_byte:end_byte]
    try:
        text = snippet.decode("utf-8", errors="ignore")
    except Exception:
        return 0
    lines = text.split("\n")
    loc = 0
    in_block_comment = False
    for line in lines:
        stripped = line.strip()
        if in_block_comment:
            if "*/" in stripped:
                in_block_comment = False
            continue
        if stripped.startswith("/*"):
            if "*/" not in stripped:
                in_block_comment = True
            continue
        if not stripped or stripped.startswith("//"):
            continue
        loc += 1
    return loc


def resolve_source_path(
    contract_path: str, source_dirs: List[Path], project_root: Path
) -> Optional[Path]:
    """
    Resolve the source file path from the contract_path stored in the dataset.

    The contract_path in the dataset is relative (e.g., 'data/synthetic/sc-source/clean/xxx.sol').
    We try multiple resolution strategies.
    """
    # Strategy 1: Try relative to project root
    candidate = project_root / contract_path
    if candidate.exists():
        return candidate

    # Strategy 2: Try the basename in each source_dir sub-directory
    basename = Path(contract_path).name
    for sdir in source_dirs:
        for subdir in [sdir, sdir / "clean", sdir / "vulnerable"]:
            candidate = subdir / basename
            if candidate.exists():
                return candidate

    return None


def collect_stats(
    dataset_dir: Path,
    ast_dataset_dir: Optional[Path] = None,
    source_dir: Optional[Path] = None,
    dataset_name: str = "Dataset",
    project_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Collect comprehensive statistics from a function-level dataset.

    Args:
        dataset_dir: Path to function_level_dataset (contains function_graphs/)
        ast_dataset_dir: Path to ast_dataset (contains ast_data/, graph_data/)
        source_dir: Path to sc-source directory (contains .sol files)
        dataset_name: Label for the dataset
        project_root: Project root for resolving relative paths

    Returns:
        Dictionary of detailed statistics
    """
    func_graph_dir = dataset_dir / "function_graphs"
    if not func_graph_dir.exists():
        print(f"Error: function_graphs/ not found in {dataset_dir}")
        sys.exit(1)

    graph_files = sorted(func_graph_dir.glob("*.json"))
    print(f"Found {len(graph_files)} function-level graph files in {dataset_dir}")

    if project_root is None:
        project_root = Path.cwd()

    source_dirs = []
    if source_dir:
        source_dirs.append(Path(source_dir))

    # ================================================================
    # Per-contract accumulators
    # ================================================================
    contract_num_nodes = []
    contract_num_edges = []
    contract_num_functions = []
    contract_num_vuln_functions = []
    contract_num_clean_functions = []
    contract_num_intra_edges = []
    contract_num_inter_edges = []
    contract_total_lines = []
    contract_loc = []  # lines of code (non-empty, non-comment)

    # ================================================================
    # Per-function accumulators
    # ================================================================
    func_num_nodes = []
    func_loc = []
    func_byte_length = []
    func_is_vulnerable = []  # 0/1

    # ================================================================
    # Edge type counters (global)
    # ================================================================
    edge_type_total = Counter()
    edge_type_intra = Counter()
    edge_type_inter = Counter()

    # ================================================================
    # Node type counter (global)
    # ================================================================
    node_type_counter = Counter()

    # ================================================================
    # Vulnerability type distribution (contract-level)
    # ================================================================
    vuln_type_counter = Counter()

    # ================================================================
    # Track if source files are available
    # ================================================================
    source_found = 0
    source_not_found = 0

    # ================================================================
    # Process each graph
    # ================================================================
    for idx, gf in enumerate(graph_files):
        if (idx + 1) % 500 == 0:
            print(f"  [{idx+1}/{len(graph_files)}] Processing...")

        with open(gf, "r") as f:
            data = json.load(f)

        num_nodes = data["num_nodes"]
        edges = data["edges"]
        intra_edges = data["intra_function_edges"]
        inter_edges = data["inter_function_edges"]
        functions = data["functions"]
        func_labels = data["function_labels"]
        vuln_type = data.get("vulnerability_type", "unknown")
        contract_path = data.get("contract_path", "")

        # Contract-level counts
        contract_num_nodes.append(num_nodes)
        contract_num_edges.append(len(edges))
        contract_num_functions.append(len(functions))
        contract_num_intra_edges.append(len(intra_edges))
        contract_num_inter_edges.append(len(inter_edges))

        num_vuln_func = sum(func_labels)
        contract_num_vuln_functions.append(num_vuln_func)
        contract_num_clean_functions.append(len(functions) - num_vuln_func)

        vuln_type_counter[vuln_type] += 1

        # Node types
        for nt in data.get("node_types", []):
            node_type_counter[nt] += 1

        # Edge types (from stored edge_counts if available, else count manually)
        edge_counts = data.get("edge_counts", {})
        if edge_counts:
            for etype_name, count in edge_counts.items():
                if etype_name in ("intra_function", "inter_function"):
                    continue
                edge_type_total[etype_name] += count
        else:
            for _, _, etype in edges:
                ename = EDGE_TYPE_NAMES.get(etype, f"type_{etype}")
                edge_type_total[ename.lower()] += 1

        # Count edge types in intra vs inter
        for _, _, etype in intra_edges:
            ename = EDGE_TYPE_NAMES.get(etype, f"type_{etype}")
            edge_type_intra[ename.lower()] += 1
        for _, _, etype in inter_edges:
            ename = EDGE_TYPE_NAMES.get(etype, f"type_{etype}")
            edge_type_inter[ename.lower()] += 1

        # Per-function stats
        for i, func in enumerate(functions):
            fn_nodes = func.get("num_nodes", 0)
            fn_start = func.get("start_byte", 0)
            fn_end = func.get("end_byte", 0)
            fn_is_vuln = func_labels[i] if i < len(func_labels) else 0

            func_num_nodes.append(fn_nodes)
            func_byte_length.append(fn_end - fn_start)
            func_is_vulnerable.append(fn_is_vuln)

        # Source LOC (contract-level)
        src_path = resolve_source_path(contract_path, source_dirs, project_root)
        if src_path and src_path.exists():
            source_found += 1
            loc = count_source_loc(src_path)
            total_lines = count_total_lines(src_path)
            if loc is not None:
                contract_loc.append(loc)
            if total_lines is not None:
                contract_total_lines.append(total_lines)

            # Per-function LOC
            try:
                raw_bytes = src_path.read_bytes()
                for func in functions:
                    fn_start = func.get("start_byte", 0)
                    fn_end = func.get("end_byte", 0)
                    fn_loc = count_function_loc(raw_bytes, fn_start, fn_end)
                    func_loc.append(fn_loc)
            except Exception:
                pass
        else:
            source_not_found += 1

    # ================================================================
    # Build statistics dictionary
    # ================================================================
    total_contracts = len(graph_files)
    total_functions = len(func_num_nodes)
    total_vuln_functions = sum(func_is_vulnerable)
    total_clean_functions = total_functions - total_vuln_functions

    stats: Dict[str, Any] = {
        "dataset_name": dataset_name,
        "dataset_path": str(dataset_dir),
        # ---- Overview ----
        "overview": {
            "total_contracts": total_contracts,
            "total_functions": total_functions,
            "total_vulnerable_functions": total_vuln_functions,
            "total_clean_functions": total_clean_functions,
            "vulnerability_ratio_functions": (
                round(total_vuln_functions / total_functions, 4)
                if total_functions > 0
                else 0
            ),
            "total_nodes": int(sum(contract_num_nodes)),
            "total_edges": int(sum(contract_num_edges)),
            "total_intra_edges": int(sum(contract_num_intra_edges)),
            "total_inter_edges": int(sum(contract_num_inter_edges)),
        },
        # ---- Contract-level distributions ----
        "contract_level": {
            "nodes_per_contract": compute_distribution_stats(contract_num_nodes),
            "edges_per_contract": compute_distribution_stats(contract_num_edges),
            "functions_per_contract": compute_distribution_stats(contract_num_functions),
            "vuln_functions_per_contract": compute_distribution_stats(contract_num_vuln_functions),
            "intra_edges_per_contract": compute_distribution_stats(contract_num_intra_edges),
            "inter_edges_per_contract": compute_distribution_stats(contract_num_inter_edges),
        },
        # ---- Function-level distributions ----
        "function_level": {
            "nodes_per_function": compute_distribution_stats(func_num_nodes),
            "byte_length_per_function": compute_distribution_stats(func_byte_length),
        },
        # ---- Edge type breakdown ----
        "edge_types": {
            "total": dict(edge_type_total.most_common()),
            "intra_function": dict(edge_type_intra.most_common()),
            "inter_function": dict(edge_type_inter.most_common()),
        },
        # ---- Node type distribution (top 30) ----
        "node_types": {
            "unique_types": len(node_type_counter),
            "top_30": dict(node_type_counter.most_common(30)),
        },
        # ---- Vulnerability type distribution ----
        "vulnerability_types": dict(vuln_type_counter.most_common()),
    }

    # ---- Source LOC stats (if available) ----
    if contract_loc:
        stats["source_code"] = {
            "source_files_found": source_found,
            "source_files_not_found": source_not_found,
            "total_lines_per_contract": compute_distribution_stats(contract_total_lines),
            "loc_per_contract": compute_distribution_stats(contract_loc),
        }
        if func_loc:
            stats["source_code"]["loc_per_function"] = compute_distribution_stats(func_loc)
    else:
        stats["source_code"] = {
            "source_files_found": source_found,
            "source_files_not_found": source_not_found,
            "note": "No source files found. Provide --source-dir to enable LOC stats.",
        }

    return stats


def print_stats(stats: Dict[str, Any]) -> None:
    """Pretty-print dataset statistics to stdout."""
    name = stats["dataset_name"]
    print()
    print("=" * 72)
    print(f"  Function-Level Dataset Statistics: {name}")
    print("=" * 72)

    ov = stats["overview"]
    print(f"\n{'─'*40}")
    print(f"  OVERVIEW")
    print(f"{'─'*40}")
    print(f"  Total contracts:              {ov['total_contracts']:>8,}")
    print(f"  Total functions:              {ov['total_functions']:>8,}")
    print(f"    Vulnerable functions:        {ov['total_vulnerable_functions']:>8,}"
          f"  ({ov['vulnerability_ratio_functions']*100:.2f}%)")
    print(f"    Clean functions:             {ov['total_clean_functions']:>8,}"
          f"  ({(1-ov['vulnerability_ratio_functions'])*100:.2f}%)")
    print(f"  Total nodes:                  {ov['total_nodes']:>8,}")
    print(f"  Total edges:                  {ov['total_edges']:>8,}")
    print(f"    Intra-function edges:        {ov['total_intra_edges']:>8,}")
    print(f"    Inter-function edges:        {ov['total_inter_edges']:>8,}")

    # Contract-level
    cl = stats["contract_level"]
    print(f"\n{'─'*40}")
    print(f"  CONTRACT-LEVEL DISTRIBUTIONS")
    print(f"{'─'*40}")
    _print_dist("Nodes per contract", cl["nodes_per_contract"])
    _print_dist("Edges per contract", cl["edges_per_contract"])
    _print_dist("Functions per contract", cl["functions_per_contract"])
    _print_dist("Vuln funcs per contract", cl["vuln_functions_per_contract"])
    _print_dist("Intra-edges per contract", cl["intra_edges_per_contract"])
    _print_dist("Inter-edges per contract", cl["inter_edges_per_contract"])

    # Function-level
    fl = stats["function_level"]
    print(f"\n{'─'*40}")
    print(f"  FUNCTION-LEVEL DISTRIBUTIONS")
    print(f"{'─'*40}")
    _print_dist("Nodes per function", fl["nodes_per_function"])
    _print_dist("Byte length per function", fl["byte_length_per_function"])

    # Source LOC
    sc = stats.get("source_code", {})
    if sc.get("loc_per_contract"):
        print(f"\n{'─'*40}")
        print(f"  SOURCE CODE (Lines of Code)")
        print(f"{'─'*40}")
        print(f"  Source files found:           {sc['source_files_found']:>8,}")
        print(f"  Source files not found:        {sc['source_files_not_found']:>8,}")
        _print_dist("Total lines per contract", sc["total_lines_per_contract"])
        _print_dist("LOC per contract", sc["loc_per_contract"])
        if sc.get("loc_per_function"):
            _print_dist("LOC per function", sc["loc_per_function"])

    # Edge types
    et = stats["edge_types"]
    print(f"\n{'─'*40}")
    print(f"  EDGE TYPE BREAKDOWN")
    print(f"{'─'*40}")
    print(f"  {'Edge Type':<16} {'Total':>10} {'Intra':>10} {'Inter':>10}")
    print(f"  {'─'*46}")
    all_types = sorted(
        set(list(et["total"].keys()) + list(et["intra_function"].keys()) + list(et["inter_function"].keys()))
    )
    for etype in all_types:
        total = et["total"].get(etype, 0)
        intra = et["intra_function"].get(etype, 0)
        inter = et["inter_function"].get(etype, 0)
        print(f"  {etype:<16} {total:>10,} {intra:>10,} {inter:>10,}")

    # Node types (top 15)
    nt = stats["node_types"]
    print(f"\n{'─'*40}")
    print(f"  NODE TYPES (top 15 of {nt['unique_types']})")
    print(f"{'─'*40}")
    for i, (ntype, count) in enumerate(list(nt["top_30"].items())[:15]):
        print(f"  {ntype:<35} {count:>10,}")

    # Vulnerability types
    vt = stats["vulnerability_types"]
    print(f"\n{'─'*40}")
    print(f"  VULNERABILITY TYPES (contract-level)")
    print(f"{'─'*40}")
    for vtype, count in vt.items():
        pct = count / ov["total_contracts"] * 100
        print(f"  {vtype:<25} {count:>6,}  ({pct:>5.1f}%)")

    print(f"\n{'='*72}")


def _print_dist(label: str, d: Dict[str, float]) -> None:
    """Print formatted distribution stats."""
    print(f"\n  {label}:")
    print(f"    Min={d['min']:.0f}  Q1={d['q1']:.0f}  "
          f"Median={d['median']:.0f}  Q3={d['q3']:.0f}  Max={d['max']:.0f}")
    print(f"    Mean={d['mean']:.2f}  Std={d['std']:.2f}  "
          f"Sum={d['sum']:.0f}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect detailed statistics of function-level datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Synthetic dataset
  python function_level_stats.py \\
      --dataset data/synthetic/function_level_dataset \\
      --ast-dataset data/synthetic/ast_dataset \\
      --source-dir data/synthetic/sc-source \\
      --dataset-name "Synthetic"

  # Real-world dataset
  python function_level_stats.py \\
      --dataset data/realworld/function_level_dataset \\
      --ast-dataset data/realworld/ast_dataset \\
      --source-dir data/realworld/sc-source \\
      --dataset-name "Real-World"

  # Save output to JSON
  python function_level_stats.py \\
      --dataset data/synthetic/function_level_dataset \\
      --source-dir data/synthetic/sc-source \\
      --output results/function_level_stats.json

  # Collect stats for both datasets at once
  python function_level_stats.py --all
""",
    )
    parser.add_argument(
        "--dataset", "-d",
        help="Path to function_level_dataset directory",
    )
    parser.add_argument(
        "--ast-dataset",
        help="Path to ast_dataset directory (optional, for extra node info)",
    )
    parser.add_argument(
        "--source-dir", "-s",
        help="Path to sc-source directory (for LOC statistics)",
    )
    parser.add_argument(
        "--dataset-name",
        default="Dataset",
        help="Label for the dataset (default: Dataset)",
    )
    parser.add_argument(
        "--output", "-o",
        help="Output JSON file path (optional, prints to stdout by default)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Collect stats for both synthetic and realworld datasets",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output (only print final stats)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()

    project_root = Path(__file__).parent

    if args.all:
        # Process both synthetic and realworld datasets
        all_stats = {}
        for dname, dpath, apath, spath in [
            (
                "Synthetic",
                project_root / "data" / "synthetic" / "function_level_dataset",
                project_root / "data" / "synthetic" / "ast_dataset",
                project_root / "data" / "synthetic" / "sc-source",
            ),
            (
                "Real-World",
                project_root / "data" / "realworld" / "function_level_dataset",
                project_root / "data" / "realworld" / "ast_dataset",
                project_root / "data" / "realworld" / "sc-source",
            ),
        ]:
            if not dpath.exists():
                print(f"Skipping {dname}: {dpath} not found")
                continue
            print(f"\nCollecting stats for {dname} dataset...")
            stats = collect_stats(
                dataset_dir=dpath,
                ast_dataset_dir=apath if apath.exists() else None,
                source_dir=spath if spath.exists() else None,
                dataset_name=dname,
                project_root=project_root,
            )
            print_stats(stats)
            all_stats[dname.lower().replace("-", "_")] = stats

        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(all_stats, f, indent=2)
            print(f"\nStats saved to {output_path}")

        return 0

    if not args.dataset:
        print("Error: --dataset is required (or use --all for both datasets)")
        return 1

    dataset_dir = Path(args.dataset)
    if not dataset_dir.exists():
        print(f"Error: Dataset directory not found: {dataset_dir}")
        return 1

    ast_dataset_dir = Path(args.ast_dataset) if args.ast_dataset else None
    source_dir = Path(args.source_dir) if args.source_dir else None

    stats = collect_stats(
        dataset_dir=dataset_dir,
        ast_dataset_dir=ast_dataset_dir,
        source_dir=source_dir,
        dataset_name=args.dataset_name,
        project_root=project_root,
    )

    print_stats(stats)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"\nStats saved to {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
