import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ============================================================================
# HuggingFace dataset constants
# ============================================================================
DATASET_REPO = "mwritescode/slither-audited-smart-contracts"
PARQUET_REVISION = "refs/convert/parquet"
PARQUET_BASE_URL = (
    "https://huggingface.co/datasets/"
    "mwritescode/slither-audited-smart-contracts/"
    "resolve/refs%2Fconvert%2Fparquet"
)

# Available configs
AVAILABLE_CONFIGS = [
    "all-multilabel", "all-plain-text",
    "big-multilabel", "big-plain-text",
    "small-multilabel", "small-plain-text",
]

# Solidity function definition regex (handles visibility, mutability, modifiers, returns)
# Matches: function name(...) [visibility] [mutability] [modifiers] [returns (...)] {
FUNC_DEF_PATTERN = re.compile(
    r'\bfunction\s+'           # keyword
    r'(\w+)\s*'                # function name
    r'\([^)]*\)',              # parameters
    re.MULTILINE,
)

# Fallback/receive/constructor patterns
SPECIAL_FUNC_PATTERNS = [
    re.compile(r'\bconstructor\s*\([^)]*\)', re.MULTILINE),
    re.compile(r'\bfallback\s*\(\s*\)', re.MULTILINE),
    re.compile(r'\breceive\s*\(\s*\)', re.MULTILINE),
]

# Pragma pattern: pragma solidity ^0.8.0;
PRAGMA_PATTERN = re.compile(r'pragma\s+solidity\s+([^;]+);', re.MULTILINE)

# Contract/Library/Interface definitions
CONTRACT_DEF_PATTERN = re.compile(
    r'\b(contract|library|interface|abstract\s+contract)\s+(\w+)', re.MULTILINE
)

# Import statements
IMPORT_PATTERN = re.compile(r'^\s*import\s+', re.MULTILINE)

# Modifier definitions
MODIFIER_DEF_PATTERN = re.compile(r'\bmodifier\s+(\w+)\s*\(', re.MULTILINE)

# Event definitions
EVENT_DEF_PATTERN = re.compile(r'\bevent\s+(\w+)\s*\(', re.MULTILINE)

# Struct definitions
STRUCT_DEF_PATTERN = re.compile(r'\bstruct\s+(\w+)\s*\{', re.MULTILINE)

# Enum definitions
ENUM_DEF_PATTERN = re.compile(r'\benum\s+(\w+)\s*\{', re.MULTILINE)

# Mapping declarations
MAPPING_PATTERN = re.compile(r'\bmapping\s*\(', re.MULTILINE)


# ============================================================================
# Source code analysis
# ============================================================================

def count_loc(source: str) -> Tuple[int, int]:
    """
    Count total lines and lines of code (non-empty, non-comment).
    
    Returns:
        (total_lines, loc)
    """
    lines = source.split('\n')
    total = len(lines)
    loc = 0
    in_block_comment = False
    
    for line in lines:
        stripped = line.strip()
        
        if in_block_comment:
            if '*/' in stripped:
                in_block_comment = False
            continue
        
        if stripped.startswith('/*'):
            if '*/' not in stripped:
                in_block_comment = True
            continue
        
        if not stripped or stripped.startswith('//'):
            continue
        
        loc += 1
    
    return total, loc


def count_functions(source: str) -> Tuple[int, List[str]]:
    """
    Count function definitions in Solidity source code.
    
    Returns:
        (count, list_of_function_names)
    """
    func_names = []
    
    # Regular functions
    for match in FUNC_DEF_PATTERN.finditer(source):
        func_names.append(match.group(1))
    
    # Special functions
    for pattern in SPECIAL_FUNC_PATTERNS:
        for match in pattern.finditer(source):
            func_names.append(match.group(0).split('(')[0].strip())
    
    return len(func_names), func_names


def extract_function_bodies(source: str) -> List[Tuple[str, str]]:
    """
    Extract function bodies with their names for per-function LOC analysis.
    Uses brace-counting to find function boundaries.
    
    Returns:
        List of (function_name, function_body) tuples
    """
    functions = []
    
    # Find all function-like definitions and their start positions
    func_starts = []
    
    for match in FUNC_DEF_PATTERN.finditer(source):
        func_starts.append((match.group(1), match.start()))
    
    for pattern in SPECIAL_FUNC_PATTERNS:
        for match in pattern.finditer(source):
            name = match.group(0).split('(')[0].strip()
            func_starts.append((name, match.start()))
    
    # Sort by position
    func_starts.sort(key=lambda x: x[1])
    
    for name, start_pos in func_starts:
        # Find the opening brace after the function signature
        brace_pos = source.find('{', start_pos)
        if brace_pos == -1:
            # Could be an interface function (no body), skip
            continue
        
        # Check if there's a semicolon before the brace (abstract/interface function)
        segment = source[start_pos:brace_pos]
        if ';' in segment:
            continue
        
        # Count braces to find the matching closing brace
        depth = 0
        end_pos = brace_pos
        in_string = False
        string_char = None
        
        for i in range(brace_pos, len(source)):
            ch = source[i]
            
            if in_string:
                if ch == string_char and (i == 0 or source[i-1] != '\\'):
                    in_string = False
                continue
            
            if ch in ('"', "'"):
                in_string = True
                string_char = ch
                continue
            
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    end_pos = i + 1
                    break
        
        body = source[start_pos:end_pos]
        functions.append((name, body))
    
    return functions


def extract_pragma_version(source: str) -> Optional[str]:
    """Extract the Solidity pragma version string."""
    match = PRAGMA_PATTERN.search(source)
    if match:
        return match.group(1).strip()
    return None


def analyze_contract(source: str) -> Dict[str, Any]:
    """
    Analyze a single smart contract source code.
    
    Returns dict with all computed metrics.
    """
    total_lines, loc = count_loc(source)
    num_functions, func_names = count_functions(source)
    char_count = len(source)
    byte_count = len(source.encode('utf-8', errors='ignore'))
    
    # Contract/library/interface definitions
    contract_defs = CONTRACT_DEF_PATTERN.findall(source)
    num_contracts = len([d for d in contract_defs if d[0] == 'contract'])
    num_libraries = len([d for d in contract_defs if d[0] == 'library'])
    num_interfaces = len([d for d in contract_defs if d[0] == 'interface'])
    num_abstract = len([d for d in contract_defs if 'abstract' in d[0]])
    
    # Other counts
    num_imports = len(IMPORT_PATTERN.findall(source))
    num_modifiers = len(MODIFIER_DEF_PATTERN.findall(source))
    num_events = len(EVENT_DEF_PATTERN.findall(source))
    num_structs = len(STRUCT_DEF_PATTERN.findall(source))
    num_enums = len(ENUM_DEF_PATTERN.findall(source))
    num_mappings = len(MAPPING_PATTERN.findall(source))
    
    # Pragma version
    pragma = extract_pragma_version(source)
    
    # Per-function LOC
    func_bodies = extract_function_bodies(source)
    func_locs = []
    func_byte_lengths = []
    for fname, body in func_bodies:
        _, floc = count_loc(body)
        func_locs.append(floc)
        func_byte_lengths.append(len(body.encode('utf-8', errors='ignore')))
    
    return {
        'total_lines': total_lines,
        'loc': loc,
        'char_count': char_count,
        'byte_count': byte_count,
        'num_functions': num_functions,
        'num_contracts': num_contracts,
        'num_libraries': num_libraries,
        'num_interfaces': num_interfaces,
        'num_abstract': num_abstract,
        'num_imports': num_imports,
        'num_modifiers': num_modifiers,
        'num_events': num_events,
        'num_structs': num_structs,
        'num_enums': num_enums,
        'num_mappings': num_mappings,
        'pragma_version': pragma,
        'func_locs': func_locs,
        'func_byte_lengths': func_byte_lengths,
    }


# ============================================================================
# Distribution stats helper
# ============================================================================

def compute_distribution_stats(values: List[float]) -> Dict[str, float]:
    """Compute min, max, mean, median, std, Q1, Q3 for a list of values."""
    if not values:
        return {"count": 0, "min": 0, "max": 0, "mean": 0, "median": 0,
                "std": 0, "q1": 0, "q3": 0, "sum": 0}
    arr = np.array(values, dtype=float)
    return {
        "count": int(len(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "q1": float(np.percentile(arr, 25)),
        "q3": float(np.percentile(arr, 75)),
        "sum": float(np.sum(arr)),
    }


# ============================================================================
# Main collection logic
# ============================================================================

def load_dataset_parquet(
    config: str = "all-multilabel",
    cache_dir: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load the dataset by downloading parquet shards via huggingface_hub.
    
    Args:
        config: Dataset configuration name
        cache_dir: Optional directory to cache downloaded parquet files
        
    Returns:
        Concatenated DataFrame
    """
    from huggingface_hub import HfApi, hf_hub_download
    
    if config not in AVAILABLE_CONFIGS:
        print(f"Unknown config '{config}'. Available: {AVAILABLE_CONFIGS}")
        sys.exit(1)
    
    # List all parquet files for the config
    api = HfApi()
    all_files = api.list_repo_files(
        DATASET_REPO, repo_type="dataset", revision=PARQUET_REVISION,
    )
    parquet_files = sorted(
        f for f in all_files
        if f.startswith(f"{config}/") and f.endswith(".parquet")
    )
    
    if not parquet_files:
        print(f"No parquet files found for config '{config}'")
        sys.exit(1)
    
    print(f"  Found {len(parquet_files)} parquet shards for '{config}'")
    
    dfs = []
    for i, pf in enumerate(parquet_files):
        print(f"  Downloading shard {i+1}/{len(parquet_files)}: {pf}...")
        local_path = hf_hub_download(
            DATASET_REPO,
            filename=pf,
            repo_type="dataset",
            revision=PARQUET_REVISION,
            cache_dir=cache_dir,
        )
        df = pd.read_parquet(local_path)
        dfs.append(df)
        print(f"    → {len(df):,} rows")
    
    combined = pd.concat(dfs, ignore_index=True)
    print(f"\n  Total rows: {len(combined):,}")
    return combined


def collect_stats(
    df: pd.DataFrame,
    dataset_name: str = "Slither-Audited",
) -> Dict[str, Any]:
    """
    Collect statistics from the loaded dataset.
    
    Args:
        df: DataFrame with columns [address, source_code, bytecode, slither]
        dataset_name: Label for the dataset
        
    Returns:
        Statistics dictionary
    """
    total = len(df)
    
    # Filter to contracts with actual source code
    has_source = df['source_code'].notna() & (df['source_code'] != '') & (df['source_code'] != '0x')
    df_with_source = df[has_source]
    num_with_source = len(df_with_source)
    num_without_source = total - num_with_source
    
    # Has bytecode
    has_bytecode = df['bytecode'].notna() & (df['bytecode'] != '') & (df['bytecode'] != '0x')
    num_with_bytecode = has_bytecode.sum()
    
    print(f"\n  Contracts with source code: {num_with_source:,} / {total:,}")
    print(f"  Contracts with bytecode:   {num_with_bytecode:,} / {total:,}")
    
    # ================================================================
    # Accumulators
    # ================================================================
    contract_total_lines = []
    contract_loc = []
    contract_char_count = []
    contract_byte_count = []
    contract_num_functions = []
    contract_num_contracts = []
    contract_num_libraries = []
    contract_num_interfaces = []
    contract_num_imports = []
    contract_num_modifiers = []
    contract_num_events = []
    contract_num_structs = []
    contract_num_enums = []
    contract_num_mappings = []
    
    pragma_counter = Counter()
    
    # Per-function accumulators (across all contracts)
    all_func_locs = []
    all_func_byte_lengths = []
    
    total_to_process = len(df_with_source)
    
    print(f"\n  Analyzing {total_to_process:,} contracts...")
    
    for idx, (_, row) in enumerate(df_with_source.iterrows()):
        if (idx + 1) % 10000 == 0:
            print(f"    [{idx+1:,}/{total_to_process:,}] Processing...")
        
        source = row['source_code']
        metrics = analyze_contract(source)
        
        contract_total_lines.append(metrics['total_lines'])
        contract_loc.append(metrics['loc'])
        contract_char_count.append(metrics['char_count'])
        contract_byte_count.append(metrics['byte_count'])
        contract_num_functions.append(metrics['num_functions'])
        contract_num_contracts.append(metrics['num_contracts'])
        contract_num_libraries.append(metrics['num_libraries'])
        contract_num_interfaces.append(metrics['num_interfaces'])
        contract_num_imports.append(metrics['num_imports'])
        contract_num_modifiers.append(metrics['num_modifiers'])
        contract_num_events.append(metrics['num_events'])
        contract_num_structs.append(metrics['num_structs'])
        contract_num_enums.append(metrics['num_enums'])
        contract_num_mappings.append(metrics['num_mappings'])
        
        if metrics['pragma_version']:
            # Normalize pragma version for grouping
            pragma = metrics['pragma_version']
            # Extract major.minor (e.g., "^0.8.0" → "0.8", ">=0.5.0 <0.7.0" → "0.5-0.7")
            version_nums = re.findall(r'(\d+\.\d+)', pragma)
            if version_nums:
                pragma_counter[version_nums[0]] += 1
        
        all_func_locs.extend(metrics['func_locs'])
        all_func_byte_lengths.extend(metrics['func_byte_lengths'])
    
    # ================================================================
    # Build statistics
    # ================================================================
    total_functions = sum(contract_num_functions)
    
    stats: Dict[str, Any] = {
        "dataset_name": dataset_name,
        "dataset_source": "huggingface: mwritescode/slither-audited-smart-contracts",
        
        # ---- Overview ----
        "overview": {
            "total_contracts": total,
            "contracts_with_source": num_with_source,
            "contracts_without_source": num_without_source,
            "contracts_with_bytecode": int(num_with_bytecode),
            "total_functions": total_functions,
            "total_definition_units": int(sum(contract_num_contracts)
                                         + sum(contract_num_libraries)
                                         + sum(contract_num_interfaces)),
        },
        
        # ---- Contract-level distributions ----
        "contract_level": {
            "total_lines_per_contract": compute_distribution_stats(contract_total_lines),
            "loc_per_contract": compute_distribution_stats(contract_loc),
            "char_count_per_contract": compute_distribution_stats(contract_char_count),
            "byte_count_per_contract": compute_distribution_stats(contract_byte_count),
            "functions_per_contract": compute_distribution_stats(contract_num_functions),
            "contract_defs_per_file": compute_distribution_stats(contract_num_contracts),
            "library_defs_per_file": compute_distribution_stats(contract_num_libraries),
            "interface_defs_per_file": compute_distribution_stats(contract_num_interfaces),
            "imports_per_contract": compute_distribution_stats(contract_num_imports),
            "modifiers_per_contract": compute_distribution_stats(contract_num_modifiers),
            "events_per_contract": compute_distribution_stats(contract_num_events),
            "structs_per_contract": compute_distribution_stats(contract_num_structs),
            "enums_per_contract": compute_distribution_stats(contract_num_enums),
            "mappings_per_contract": compute_distribution_stats(contract_num_mappings),
        },
        
        # ---- Function-level distributions ----
        "function_level": {
            "loc_per_function": compute_distribution_stats(all_func_locs),
            "byte_length_per_function": compute_distribution_stats(all_func_byte_lengths),
        },
        
        # ---- Solidity version distribution (top 15) ----
        "solidity_versions": dict(pragma_counter.most_common(15)),
    }
    
    return stats


def print_stats(stats: Dict[str, Any]) -> None:
    """Pretty-print dataset statistics."""
    name = stats["dataset_name"]
    print()
    print("=" * 72)
    print(f"  Dataset Statistics: {name}")
    print("=" * 72)
    
    ov = stats["overview"]
    print(f"\n{'─'*40}")
    print(f"  OVERVIEW")
    print(f"{'─'*40}")
    print(f"  Total contracts:              {ov['total_contracts']:>10,}")
    print(f"  With source code:             {ov['contracts_with_source']:>10,}")
    print(f"  Without source code:          {ov['contracts_without_source']:>10,}")
    print(f"  With bytecode:                {ov['contracts_with_bytecode']:>10,}")
    print(f"  Total functions:              {ov['total_functions']:>10,}")
    print(f"  Total definition units:       {ov['total_definition_units']:>10,}"
          f"  (contracts + libraries + interfaces)")
    
    cl = stats["contract_level"]
    print(f"\n{'─'*40}")
    print(f"  CONTRACT-LEVEL DISTRIBUTIONS")
    print(f"{'─'*40}")
    _print_dist("Total lines per contract", cl["total_lines_per_contract"])
    _print_dist("LOC per contract", cl["loc_per_contract"])
    _print_dist("Bytes per contract", cl["byte_count_per_contract"])
    _print_dist("Functions per contract", cl["functions_per_contract"])
    _print_dist("Contract defs per file", cl["contract_defs_per_file"])
    _print_dist("Library defs per file", cl["library_defs_per_file"])
    _print_dist("Interface defs per file", cl["interface_defs_per_file"])
    _print_dist("Imports per contract", cl["imports_per_contract"])
    _print_dist("Modifiers per contract", cl["modifiers_per_contract"])
    _print_dist("Events per contract", cl["events_per_contract"])
    _print_dist("Structs per contract", cl["structs_per_contract"])
    _print_dist("Enums per contract", cl["enums_per_contract"])
    _print_dist("Mappings per contract", cl["mappings_per_contract"])
    
    fl = stats["function_level"]
    print(f"\n{'─'*40}")
    print(f"  FUNCTION-LEVEL DISTRIBUTIONS")
    print(f"{'─'*40}")
    _print_dist("LOC per function", fl["loc_per_function"])
    _print_dist("Byte length per function", fl["byte_length_per_function"])
    
    sv = stats.get("solidity_versions", {})
    if sv:
        print(f"\n{'─'*40}")
        print(f"  SOLIDITY VERSIONS (top 15)")
        print(f"{'─'*40}")
        total_c = stats["overview"]["contracts_with_source"]
        for ver, count in sv.items():
            pct = count / total_c * 100 if total_c > 0 else 0
            print(f"  {ver:<12} {count:>10,}  ({pct:>5.1f}%)")
    
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
        description="Collect statistics from HuggingFace Slither-Audited Smart Contracts dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full dataset (all 113K contracts)
  python slither_dataset_stats.py --output results/slither_dataset_stats.json

  # Smaller subset
  python slither_dataset_stats.py --config small-multilabel --output results/slither_small_stats.json

  # With local caching
  python slither_dataset_stats.py --cache-dir data/slither_cache --output results/slither_dataset_stats.json
""",
    )
    parser.add_argument(
        "--config", "-c",
        default="all-multilabel",
        choices=AVAILABLE_CONFIGS,
        help="Dataset configuration (default: all-multilabel, ~113K contracts)",
    )
    parser.add_argument(
        "--output", "-o",
        help="Output JSON file path",
    )
    parser.add_argument(
        "--cache-dir",
        help="Directory to cache downloaded parquet files",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    
    print(f"Loading dataset: {DATASET_REPO} (config: {args.config})")
    df = load_dataset_parquet(config=args.config, cache_dir=args.cache_dir)
    
    print(f"\nCollecting statistics...")
    stats = collect_stats(df, dataset_name=f"Slither-Audited ({args.config})")
    
    print_stats(stats)
    
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"\nStats saved to {output_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
