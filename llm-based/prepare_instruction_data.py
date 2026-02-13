"""
Prepare instruction fine-tuning data for llama-factory.

Converts smart contract vulnerability detection dataset into ShareGPT format
for instruction-based fine-tuning.

Training data: one sample per contract (contract-level classification).
Test data:     one sample per function, grouped by contract, for both
               micro (function-level) and macro (contract-level) evaluation.

ShareGPT format (train):
[
    {
        "conversations": [
            {"from": "human", "value": "user message"},
            {"from": "gpt", "value": "assistant response"}
        ]
    },
    ...
]

Function-level test format:
[
    {
        "contract_id": "filename",
        "contract_path": "path/to/contract.sol",
        "vulnerability_type": "reentrancy",
        "contract_label": 1,
        "functions": [
            {
                "function_name": "withdraw",
                "function_label": 1,
                "conversations": [{"from": "human", ...}, {"from": "gpt", ...}]
            },
            ...
        ]
    },
    ...
]

Usage:
    # Run from project root directory
    python llm-based/prepare_instruction_data.py
    
    # Limit training samples
    python llm-based/prepare_instruction_data.py --max-train-samples 1000
    
    # Custom paths
    python llm-based/prepare_instruction_data.py \
        --synthetic-data data/synthetic/ast_dataset \
        --realworld-data data/realworld/ast_dataset \
        --output-dir llm-based/data
"""
import json
import os
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm
import argparse


INSTRUCTION_TEMPLATE = "Is the following Solidity smart contract vulnerable? Answer with 'vulnerable' or 'clean'."


# Single-token responses
VULNERABLE_RESPONSE = "vulnerable"
CLEAN_RESPONSE = "clean"


def create_vulnerable_response(vuln_type: str, code_snippet: Optional[str] = None) -> str:
    """Create a response for a vulnerable contract (single token)."""
    return VULNERABLE_RESPONSE


def load_contract_source(contract_path: str, root_dir: Path) -> Optional[str]:
    """Load Solidity source code from file."""
    # Resolve contract_path (might be relative to project root)
    if not os.path.isabs(contract_path):
        # contract_path is like "data/synthetic/sc-source/vulnerable/xxx.sol"
        # Find project root
        current = root_dir.resolve()
        while current.name and current.name != 'sc-vuln-detection':
            current = current.parent
            if current.parent == current:  # Reached filesystem root
                break
        
        abs_contract_path = current / contract_path
        if not abs_contract_path.exists():
            return None
        contract_path = str(abs_contract_path)
    elif not os.path.exists(contract_path):
        return None
    
    try:
        with open(contract_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading {contract_path}: {e}")
        return None


def truncate_code(code: str, max_tokens: int = 1800) -> str:
    """Truncate code to fit within token limit (rough estimate: 1 token ≈ 4 chars)."""
    max_chars = max_tokens * 4
    if len(code) <= max_chars:
        return code
    
    # Try to truncate at function boundaries
    lines = code.split('\n')
    truncated = []
    char_count = 0
    
    for line in lines:
        if char_count + len(line) > max_chars:
            truncated.append("// ... (code truncated) ...")
            break
        truncated.append(line)
        char_count += len(line) + 1
    
    return '\n'.join(truncated)


def create_conversation(
    source_code: str,
    label: int,
    vuln_type: str,
) -> Dict:
    """Create a single conversation in ShareGPT format."""
    # Truncate code if too long
    code = truncate_code(source_code)
    
    # Create user message
    user_message = f"{INSTRUCTION_TEMPLATE}\n\n```solidity\n{code}\n```"
    
    # Create assistant response (single token)
    if label == 0:  # Clean
        assistant_message = CLEAN_RESPONSE
    else:  # Vulnerable
        assistant_message = VULNERABLE_RESPONSE
    
    return {
        "conversations": [
            {"from": "human", "value": user_message},
            {"from": "gpt", "value": assistant_message}
        ]
    }


def prepare_dataset(
    root: str,
    output_file: str,
    split: str = "train",
    max_samples: Optional[int] = None,
    seed: int = 42
):
    """
    Prepare instruction tuning dataset from AST dataset (contract-level).
    
    Args:
        root: Path to ast_dataset (e.g., 'data/synthetic/ast_dataset')
        output_file: Output JSON file path
        split: 'train', 'val', 'test', or 'all'
        max_samples: Maximum number of samples (None = all)
        seed: Random seed for reproducibility
    """
    root_dir = Path(root)
    graph_data_dir = root_dir / "graph_data"
    
    if not graph_data_dir.exists():
        raise ValueError(f"Graph data directory not found: {graph_data_dir}")
    
    # Load all graph JSON files
    all_files = sorted(graph_data_dir.glob("*.json"))
    
    if len(all_files) == 0:
        raise ValueError(f"No JSON files found in {graph_data_dir}")
    
    # Split using same approach as GNN models (70/15/15 with seed 42)
    # Use deterministic shuffle based on seed
    indices = list(range(len(all_files)))
    # Sort indices by hash of (index + seed) for deterministic shuffle
    indices.sort(key=lambda x: hash((x, seed)))
    
    train_end = int(0.7 * len(all_files))
    val_end = int(0.85 * len(all_files))
    
    if split == 'train':
        selected_indices = indices[:train_end]
    elif split == 'val':
        selected_indices = indices[train_end:val_end]
    elif split == 'test':
        selected_indices = indices[val_end:]
    else:  # 'all'
        selected_indices = indices
    
    print(f"Processing {len(selected_indices)} {split} samples from {root}")
    
    # Limit samples if requested
    if max_samples and len(selected_indices) > max_samples:
        selected_indices = random.sample(selected_indices, max_samples)
        print(f"Limited to {max_samples} samples")
    
    conversations = []
    skipped = 0
    
    for idx in tqdm(selected_indices, desc=f"Preparing {split} data"):
        json_file = all_files[idx]
        
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Get contract path and load source
        contract_path = data.get('contract_path')
        if not contract_path:
            skipped += 1
            continue
        
        source_code = load_contract_source(contract_path, root_dir)
        if not source_code:
            skipped += 1
            continue
        
        # Get vulnerability info
        vuln_type = data.get('vulnerability_type', 'clean')
        label = 0 if vuln_type == 'clean' else 1
        
        # Create conversation
        conv = create_conversation(source_code, label, vuln_type)
        conversations.append(conv)
    
    print(f"Created {len(conversations)} conversations ({skipped} skipped)")
    
    # Save to JSON
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(conversations, f, indent=2, ensure_ascii=False)
    
    print(f"Saved to {output_path}")
    
    # Print stats
    vulnerable_count = sum(1 for c in conversations if c["conversations"][1]["value"] == "vulnerable")
    print(f"\nDataset statistics:")
    print(f"  Total samples: {len(conversations)}")
    print(f"  Vulnerable: {vulnerable_count} ({vulnerable_count/len(conversations)*100:.1f}%)")
    print(f"  Clean: {len(conversations) - vulnerable_count} ({(len(conversations)-vulnerable_count)/len(conversations)*100:.1f}%)")


def prepare_function_level_test(
    ast_dataset_root: str,
    func_dataset_root: str,
    output_file: str,
    split: str = "test",
    seed: int = 42
):
    """
    Prepare function-level test data from function_level_dataset.

    For each contract in the test split, extracts every function's source code
    and creates a per-function classification sample.  Results are grouped by
    contract so that both micro (function-level) and macro (contract-level)
    metrics can be computed at evaluation time.

    Args:
        ast_dataset_root: Path to ast_dataset (for graph_data splitting reference)
        func_dataset_root: Path to function_level_dataset (for function info)
        output_file: Output JSON file path
        split: 'test' or 'all'
        seed: Random seed (must match training split)
    """
    ast_root = Path(ast_dataset_root)
    func_root = Path(func_dataset_root)
    func_graph_dir = func_root / "function_graphs"
    graph_data_dir = ast_root / "graph_data"

    if not func_graph_dir.exists():
        raise ValueError(f"Function graph directory not found: {func_graph_dir}")

    # Determine which contracts belong to the test split
    # Use same splitting logic as prepare_dataset (graph_data files)
    if split == "all":
        # Use all function graph files (e.g. realworld)
        selected_files = sorted(func_graph_dir.glob("*.json"))
    else:
        # Replicate the train/val/test split from graph_data
        all_graph_files = sorted(graph_data_dir.glob("*.json"))
        indices = list(range(len(all_graph_files)))
        indices.sort(key=lambda x: hash((x, seed)))

        train_end = int(0.7 * len(all_graph_files))
        val_end = int(0.85 * len(all_graph_files))

        if split == "test":
            selected_indices = indices[val_end:]
        elif split == "val":
            selected_indices = indices[train_end:val_end]
        else:
            selected_indices = indices[:train_end]

        # Map indices to filenames, then find matching function graph files
        selected_names = {all_graph_files[i].name for i in selected_indices}
        selected_files = [
            func_graph_dir / name
            for name in sorted(selected_names)
            if (func_graph_dir / name).exists()
        ]

    print(f"Processing {len(selected_files)} contracts for function-level {split} set")

    # Find project root for resolving contract_path
    project_root = ast_root.resolve()
    while project_root.name and project_root.name != "sc-vuln-detection":
        project_root = project_root.parent
        if project_root.parent == project_root:
            break

    contract_samples = []
    total_functions = 0
    total_vuln_funcs = 0
    skipped_contracts = 0

    for fg_path in tqdm(selected_files, desc=f"Preparing function-level {split}"):
        with open(fg_path, "r") as f:
            fg = json.load(f)

        contract_path = fg.get("contract_path", "")
        functions = fg.get("functions", [])
        function_labels = fg.get("function_labels", [])
        vuln_type = fg.get("vulnerability_type", "clean")

        if not functions:
            skipped_contracts += 1
            continue

        # Load source code
        abs_contract_path = project_root / contract_path
        if not abs_contract_path.exists():
            # Try relative from cwd
            if not os.path.exists(contract_path):
                skipped_contracts += 1
                continue
            abs_contract_path = Path(contract_path)

        try:
            with open(abs_contract_path, "r", encoding="utf-8", errors="ignore") as f:
                source_code = f.read()
        except Exception:
            skipped_contracts += 1
            continue

        source_bytes = source_code.encode("utf-8")

        # Extract each function
        func_entries = []
        for i, func_info in enumerate(functions):
            start = func_info["start_byte"]
            end = func_info["end_byte"]
            func_src = source_bytes[start:end].decode("utf-8", errors="ignore")

            if not func_src.strip():
                continue

            label = function_labels[i] if i < len(function_labels) else 0
            func_name = func_info.get("name", f"function_{i}")
            if not func_name:
                func_name = f"unnamed_{i}"

            code = truncate_code(func_src)
            user_message = f"{INSTRUCTION_TEMPLATE}\n\n```solidity\n{code}\n```"
            assistant_message = VULNERABLE_RESPONSE if label == 1 else CLEAN_RESPONSE

            func_entries.append({
                "function_name": func_name,
                "function_label": label,
                "conversations": [
                    {"from": "human", "value": user_message},
                    {"from": "gpt", "value": assistant_message},
                ],
            })

        if not func_entries:
            skipped_contracts += 1
            continue

        contract_label = 1 if any(fl == 1 for fl in function_labels) else 0

        contract_samples.append({
            "contract_id": fg_path.stem,
            "contract_path": contract_path,
            "vulnerability_type": vuln_type,
            "contract_label": contract_label,
            "functions": func_entries,
        })

        total_functions += len(func_entries)
        total_vuln_funcs += sum(fe["function_label"] for fe in func_entries)

    # Save
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(contract_samples, f, indent=2, ensure_ascii=False)

    print(f"Saved to {output_path}")
    print(f"\nFunction-level test statistics:")
    print(f"  Contracts:          {len(contract_samples)} ({skipped_contracts} skipped)")
    print(f"  Total functions:    {total_functions}")
    print(f"  Vulnerable funcs:   {total_vuln_funcs} ({total_vuln_funcs/max(total_functions,1)*100:.1f}%)")
    print(f"  Clean funcs:        {total_functions - total_vuln_funcs} ({(total_functions-total_vuln_funcs)/max(total_functions,1)*100:.1f}%)")
    vuln_contracts = sum(1 for c in contract_samples if c["contract_label"] == 1)
    print(f"  Vulnerable contracts: {vuln_contracts} ({vuln_contracts/max(len(contract_samples),1)*100:.1f}%)")
    print(f"  Clean contracts:      {len(contract_samples) - vuln_contracts}")


def main():
    parser = argparse.ArgumentParser(description="Prepare instruction tuning data for llama-factory")
    parser.add_argument(
        '--synthetic-data',
        type=str,
        default='data/synthetic/ast_dataset',
        help='Path to synthetic ast_dataset (relative to project root)'
    )
    parser.add_argument(
        '--realworld-data',
        type=str,
        default='data/realworld/ast_dataset',
        help='Path to realworld ast_dataset (for IRL test)'
    )
    parser.add_argument(
        '--synthetic-func-data',
        type=str,
        default='data/synthetic/function_level_dataset',
        help='Path to synthetic function_level_dataset'
    )
    parser.add_argument(
        '--realworld-func-data',
        type=str,
        default='data/realworld/function_level_dataset',
        help='Path to realworld function_level_dataset'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='llm-based/data',
        help='Output directory for JSON files'
    )
    parser.add_argument(
        '--max-train-samples',
        type=int,
        default=None,
        help='Maximum training samples (None = all)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Preparing Instruction Fine-tuning Data for llama-factory")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Synthetic data:       {args.synthetic_data}")
    print(f"  Synthetic func data:  {args.synthetic_func_data}")
    print(f"  Real-world data:      {args.realworld_data}")
    print(f"  Real-world func data: {args.realworld_func_data}")
    print(f"  Output directory:     {args.output_dir}")
    print(f"  Seed:                 {args.seed}")
    
    # Prepare synthetic train/val (contract-level, for training)
    for split in ['train', 'val']:
        max_samples = args.max_train_samples if split == 'train' else None
        print(f"\n{'='*70}")
        print(f"Preparing SYNTHETIC {split.upper()} set (contract-level)")
        print(f"{'='*70}")
        prepare_dataset(
            root=args.synthetic_data,
            output_file=f"{args.output_dir}/sc_synthetic_{split}.json",
            split=split,
            max_samples=max_samples,
            seed=args.seed
        )
    
    # Prepare function-level test sets
    # Synthetic test (function-level)
    if os.path.exists(args.synthetic_func_data):
        print(f"\n{'='*70}")
        print("Preparing SYNTHETIC function-level test set")
        print(f"{'='*70}")
        prepare_function_level_test(
            ast_dataset_root=args.synthetic_data,
            func_dataset_root=args.synthetic_func_data,
            output_file=f"{args.output_dir}/sc_synthetic_test.json",
            split='test',
            seed=args.seed,
        )
    else:
        print(f"\n⚠️  Skipping synthetic function-level data (not found: {args.synthetic_func_data})")

    # Real-world test (function-level, use all)
    if os.path.exists(args.realworld_func_data):
        print(f"\n{'='*70}")
        print("Preparing REAL-WORLD function-level test set")
        print(f"{'='*70}")
        prepare_function_level_test(
            ast_dataset_root=args.realworld_data,
            func_dataset_root=args.realworld_func_data,
            output_file=f"{args.output_dir}/sc_realworld_test.json",
            split='all',  # Use all real-world samples
            seed=args.seed,
        )
    else:
        print(f"\n⚠️  Skipping real-world function-level data (not found: {args.realworld_func_data})")

    print("\n" + "=" * 70)
    print("DATA PREPARATION COMPLETE!")
    print("=" * 70)
    print("\nGenerated files in llm-based/data/:")
    print("  - sc_synthetic_train.json  (contract-level, for training)")
    print("  - sc_synthetic_val.json    (contract-level, for validation)")
    print("  - sc_synthetic_test.json   (function-level, grouped by contract)")
    if os.path.exists(args.realworld_func_data):
        print("  - sc_realworld_test.json   (function-level, grouped by contract)")
    
    print("\nNext steps:")
    print("  1. Verify the dataset_info.json is updated")
    print("  2. Run fine-tuning:")
    print("     cd llm-based")
    print("     bash quickstart_instruct.sh")
    print("\nOr use llama-factory directly:")
    print("  llamafactory-cli train llm-based/config/sft_config.yaml")


if __name__ == "__main__":
    main()
