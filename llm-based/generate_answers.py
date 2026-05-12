"""
Generate answers from fine-tuned instruction models on test sets.

Loads a fine-tuned LoRA adapter and generates predictions for each *function*
in the function-level test data.  Reports both:
  - Micro (function-level) metrics: each function is a sample.
  - Macro (contract-level) metrics: a contract is predicted vulnerable if
    ANY of its functions is predicted vulnerable.

Test data format (produced by prepare_instruction_data.py):
[
    {
        "contract_id": "...",
        "contract_path": "...",
        "vulnerability_type": "...",
        "contract_label": 0 | 1,
        "functions": [
            {
                "function_name": "...",
                "function_label": 0 | 1,
                "conversations": [{"from": "human", ...}, {"from": "gpt", ...}]
            }, ...
        ]
    }, ...
]

Usage:
    python generate_answers.py \
        --model-path outputs/instruct_sft/qwen2_5-coder-7b/lora/sft

    # See run_generate_answers.sh for running all models
"""
import json
import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_test_data(json_file: str) -> List[Dict]:
    """Load function-level test data (list of contract dicts)."""
    with open(json_file, "r") as f:
        return json.load(f)


def extract_prediction(response: str) -> int:
    """Extract binary prediction from model response.  1=vulnerable, 0=clean."""
    response_lower = response.strip().lower()

    if response_lower in ("vulnerable", "vulnerable."):
        return 1
    if response_lower in ("clean", "clean."):
        return 0

    if "vulnerable" in response_lower:
        return 1

    if any(w in response_lower for w in
           ["clean", "secure", "no vulnerabilities", "no security issues", "safe"]):
        return 0

    # Conservative default
    return 1


def compute_metrics(predictions: List[int], ground_truths: List[int]) -> Dict:
    """Compute binary classification metrics."""
    tp = sum(1 for p, g in zip(predictions, ground_truths) if p == 1 and g == 1)
    fp = sum(1 for p, g in zip(predictions, ground_truths) if p == 1 and g == 0)
    tn = sum(1 for p, g in zip(predictions, ground_truths) if p == 0 and g == 0)
    fn = sum(1 for p, g in zip(predictions, ground_truths) if p == 0 and g == 1)

    total = len(predictions)
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "total": total,
    }


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict_single(model, tokenizer, user_message: str,
                   max_new_tokens: int, temperature: float, device: str) -> str:
    """Run a single inference and return the decoded response string."""
    messages = [{"role": "user", "content": user_message}]
    try:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        prompt = user_message + "\n\nAnswer:"

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=temperature > 0,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.1,
    )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )
    return response.strip()


# ---------------------------------------------------------------------------
# Evaluation driver
# ---------------------------------------------------------------------------

def run_evaluation(
    model, tokenizer,
    test_file: str, output_file: str, test_name: str,
    max_new_tokens: int, temperature: float, device: str,
) -> Optional[Dict]:
    """
    Run function-level inference on *test_file* and compute both
    micro (function-level) and macro (contract-level) metrics.
    """
    test_path = Path(test_file)
    if not test_path.exists():
        print(f"  WARNING: Test file not found: {test_file}")
        return None

    contracts = load_test_data(test_file)
    total_funcs = sum(len(c["functions"]) for c in contracts)
    print(f"\n  {test_name}: {len(contracts)} contracts, {total_funcs} functions")

    model.eval()
    start_time = time.time()

    # --- per-function predictions ---
    all_func_preds: List[int] = []
    all_func_gts: List[int] = []

    # --- per-contract aggregation ---
    contract_preds: List[int] = []
    contract_gts: List[int] = []

    output_contracts = []  # detailed results

    with torch.no_grad():
        for contract in tqdm(contracts, desc=f"  {test_name}"):
            contract_id = contract["contract_id"]
            contract_label = contract["contract_label"]
            func_results = []
            any_func_predicted_vuln = False

            for func in contract["functions"]:
                user_msg = func["conversations"][0]["value"]
                gt_response = func["conversations"][1]["value"]
                gt_label = func["function_label"]

                response = predict_single(
                    model, tokenizer, user_msg,
                    max_new_tokens, temperature, device,
                )
                pred_label = extract_prediction(response)

                if pred_label == 1:
                    any_func_predicted_vuln = True

                all_func_preds.append(pred_label)
                all_func_gts.append(gt_label)

                func_results.append({
                    "function_name": func["function_name"],
                    "ground_truth_label": gt_label,
                    "predicted_label": pred_label,
                    "model_response": response,
                    "ground_truth_response": gt_response,
                })

            # Contract-level: vulnerable if ANY function predicted vulnerable
            contract_pred = 1 if any_func_predicted_vuln else 0
            contract_preds.append(contract_pred)
            contract_gts.append(contract_label)

            output_contracts.append({
                "contract_id": contract_id,
                "contract_path": contract.get("contract_path", ""),
                "vulnerability_type": contract.get("vulnerability_type", ""),
                "contract_label": contract_label,
                "contract_predicted": contract_pred,
                "functions": func_results,
            })

    elapsed = time.time() - start_time

    # Compute metrics
    micro_metrics = compute_metrics(all_func_preds, all_func_gts)
    macro_metrics = compute_metrics(contract_preds, contract_gts)

    output = {
        "test_set": test_name,
        "test_file": test_file,
        "num_contracts": len(contracts),
        "num_functions": total_funcs,
        "elapsed_seconds": round(elapsed, 1),
        "micro_metrics": micro_metrics,
        "macro_metrics": macro_metrics,
        "contracts": output_contracts,
    }

    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n  {'Micro (function-level)':30s}  {'Macro (contract-level)':30s}")
    print(f"  {'─'*30}  {'─'*30}")
    print(f"  Accuracy:  {micro_metrics['accuracy']:.4f}"
          f"                    Accuracy:  {macro_metrics['accuracy']:.4f}")
    print(f"  Precision: {micro_metrics['precision']:.4f}"
          f"                    Precision: {macro_metrics['precision']:.4f}")
    print(f"  Recall:    {micro_metrics['recall']:.4f}"
          f"                    Recall:    {macro_metrics['recall']:.4f}")
    print(f"  F1 Score:  {micro_metrics['f1']:.4f}"
          f"                    F1 Score:  {macro_metrics['f1']:.4f}")
    print(f"  Samples:   {micro_metrics['total']}"
          f"                       Samples:   {macro_metrics['total']}")
    print(f"  Time: {elapsed:.1f}s  |  Saved to: {out_path}")

    return output


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate function-level answers from fine-tuned models on test sets"
    )
    parser.add_argument(
        "--model-path", type=str, required=True,
        help="Path to fine-tuned LoRA adapter directory",
    )
    parser.add_argument(
        "--base-model", type=str, default=None,
        help="Base model name (auto-detected from adapter_config.json if omitted)",
    )
    parser.add_argument(
        "--synthetic-test", type=str, default="data/sc_synthetic_test.json",
        help="Path to synthetic test data (function-level format)",
    )
    parser.add_argument(
        "--realworld-test", type=str, default="data/sc_realworld_test.json",
        help="Path to real-world test data (function-level format)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: <model-path>/eval_results/)",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=128,
        help="Maximum tokens to generate per function",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.1,
        help="Sampling temperature (lower = more deterministic)",
    )
    parser.add_argument(
        "--test-set", type=str, choices=["synthetic", "realworld", "both"],
        default="both", help="Which test set(s) to evaluate on",
    )

    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Auto-detect base model
    if args.base_model is None:
        cfg_path = Path(args.model_path) / "adapter_config.json"
        if cfg_path.exists():
            with open(cfg_path) as f:
                args.base_model = json.load(f).get("base_model_name_or_path")
            print(f"Auto-detected base model: {args.base_model}")
        else:
            raise ValueError(
                "Cannot auto-detect base model. Provide --base-model or ensure "
                "adapter_config.json exists in --model-path."
            )

    if args.output_dir is None:
        args.output_dir = str(Path(args.model_path) / "eval_results")

    print("=" * 70)
    print("Generate Function-Level Answers from Fine-tuned Model")
    print("=" * 70)
    print(f"  Base model:     {args.base_model}")
    print(f"  LoRA adapter:   {args.model_path}")
    print(f"  Device:         {device}")
    print(f"  Temperature:    {args.temperature}")
    print(f"  Max new tokens: {args.max_new_tokens}")
    print(f"  Output dir:     {args.output_dir}")

    # Load model
    print(f"\nLoading base model: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model, trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )

    print(f"Loading LoRA adapter: {args.model_path}")
    model = PeftModel.from_pretrained(base_model, args.model_path)
    model = model.merge_and_unload()
    print("Model loaded and merged.\n")

    all_results: Dict[str, Dict] = {}

    # Synthetic test
    if args.test_set in ("synthetic", "both"):
        syn = run_evaluation(
            model, tokenizer,
            args.synthetic_test,
            f"{args.output_dir}/synthetic_test_answers.json",
            "Synthetic Test",
            args.max_new_tokens, args.temperature, device,
        )
        if syn:
            all_results["synthetic_test"] = {
                "micro": syn["micro_metrics"],
                "macro": syn["macro_metrics"],
            }

    # Real-world test
    if args.test_set in ("realworld", "both"):
        rw = run_evaluation(
            model, tokenizer,
            args.realworld_test,
            f"{args.output_dir}/realworld_test_answers.json",
            "Real-World Test",
            args.max_new_tokens, args.temperature, device,
        )
        if rw:
            all_results["realworld_test"] = {
                "micro": rw["micro_metrics"],
                "macro": rw["macro_metrics"],
            }

    # Save summary
    summary_path = Path(args.output_dir) / "metrics_summary.json"
    summary = {
        "model": args.base_model,
        "adapter": args.model_path,
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
        "results": all_results,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'=' * 70}")
    print("Summary")
    print(f"{'=' * 70}")
    for name, res in all_results.items():
        mi, ma = res["micro"], res["macro"]
        print(f"  {name}:")
        print(f"    Micro (func):     Acc={mi['accuracy']:.4f}  P={mi['precision']:.4f}  "
              f"R={mi['recall']:.4f}  F1={mi['f1']:.4f}")
        print(f"    Macro (contract): Acc={ma['accuracy']:.4f}  P={ma['precision']:.4f}  "
              f"R={ma['recall']:.4f}  F1={ma['f1']:.4f}")
    print(f"\n  Summary saved to: {summary_path}")
    print("Done.")


if __name__ == "__main__":
    main()
