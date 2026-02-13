"""
Evaluate instruction-tuned model on test sets.

Loads a fine-tuned model and evaluates on synthetic test and IRL test sets.
"""
import json
import argparse
from pathlib import Path
from typing import Dict, List
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm
import re


def load_test_data(json_file: str) -> List[Dict]:
    """Load test data from ShareGPT format JSON."""
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data


def extract_prediction(response: str) -> int:
    """
    Extract binary prediction from model response.
    
    Returns:
        1 if vulnerable, 0 if clean
    """
    response_lower = response.lower()
    
    # Check for explicit vulnerable markers
    if "vulnerable" in response_lower or "⚠️" in response:
        return 1
    
    # Check for clean/secure markers
    if any(word in response_lower for word in ["clean", "secure", "no vulnerabilities", "no security issues"]):
        return 0
    
    # Default to vulnerable if uncertain (conservative)
    return 1


def evaluate_model(
    model,
    tokenizer,
    test_data: List[Dict],
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    device: str = "cuda"
) -> Dict:
    """
    Evaluate model on test data.
    
    Returns:
        Dictionary with metrics: accuracy, precision, recall, F1
    """
    predictions = []
    ground_truths = []
    
    model.eval()
    
    with torch.no_grad():
        for sample in tqdm(test_data, desc="Evaluating"):
            # Get user message (contract code)
            user_message = sample["conversations"][0]["value"]
            ground_truth_response = sample["conversations"][1]["value"]
            
            # Determine ground truth label
            if "VULNERABLE" in ground_truth_response or "⚠️" in ground_truth_response:
                ground_truth = 1
            else:
                ground_truth = 0
            
            # Generate prediction
            inputs = tokenizer(user_message, return_tensors="pt", truncation=True, max_length=2048)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )
            
            # Decode response
            response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            
            # Extract prediction
            prediction = extract_prediction(response)
            
            predictions.append(prediction)
            ground_truths.append(ground_truth)
    
    # Compute metrics
    tp = sum(1 for p, g in zip(predictions, ground_truths) if p == 1 and g == 1)
    fp = sum(1 for p, g in zip(predictions, ground_truths) if p == 1 and g == 0)
    tn = sum(1 for p, g in zip(predictions, ground_truths) if p == 0 and g == 0)
    fn = sum(1 for p, g in zip(predictions, ground_truths) if p == 0 and g == 1)
    
    accuracy = (tp + tn) / len(predictions) if len(predictions) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "total": len(predictions)
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate instruction-tuned model")
    parser.add_argument(
        '--model-path',
        type=str,
        required=True,
        help='Path to fine-tuned model checkpoint (LoRA adapter)'
    )
    parser.add_argument(
        '--base-model',
        type=str,
        default='Qwen/Qwen2.5-Coder-7B-Instruct',
        help='Base model name'
    )
    parser.add_argument(
        '--synthetic-test',
        type=str,
        default='data/sc_synthetic_test.json',
        help='Path to synthetic test data'
    )
    parser.add_argument(
        '--realworld-test',
        type=str,
        default='data/sc_realworld_test.json',
        help='Path to real-world test data'
    )
    parser.add_argument(
        '--max-new-tokens',
        type=int,
        default=256,
        help='Maximum tokens to generate'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.7,
        help='Sampling temperature'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='outputs/instruct_sft/eval_results.json',
        help='Output file for results'
    )
    
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load base model and tokenizer
    print(f"Loading base model: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    
    # Load LoRA adapter
    print(f"Loading LoRA adapter: {args.model_path}")
    model = PeftModel.from_pretrained(base_model, args.model_path)
    model = model.merge_and_unload()  # Merge for faster inference
    
    results = {}
    
    # Evaluate on synthetic test
    if Path(args.synthetic_test).exists():
        print(f"\nEvaluating on synthetic test set...")
        syn_test_data = load_test_data(args.synthetic_test)
        syn_metrics = evaluate_model(model, tokenizer, syn_test_data, args.max_new_tokens, args.temperature, device)
        results["synthetic_test"] = syn_metrics
        
        print(f"\n{'='*60}")
        print("Synthetic Test Results:")
        print(f"{'='*60}")
        print(f"Accuracy:  {syn_metrics['accuracy']:.4f}")
        print(f"Precision: {syn_metrics['precision']:.4f}")
        print(f"Recall:    {syn_metrics['recall']:.4f}")
        print(f"F1 Score:  {syn_metrics['f1']:.4f}")
        print(f"Samples:   {syn_metrics['total']}")
    
    # Evaluate on real-world test
    if Path(args.realworld_test).exists():
        print(f"\nEvaluating on real-world (IRL) test set...")
        irl_test_data = load_test_data(args.realworld_test)
        irl_metrics = evaluate_model(model, tokenizer, irl_test_data, args.max_new_tokens, args.temperature, device)
        results["realworld_test"] = irl_metrics
        
        print(f"\n{'='*60}")
        print("Real-World Test Results:")
        print(f"{'='*60}")
        print(f"Accuracy:  {irl_metrics['accuracy']:.4f}")
        print(f"Precision: {irl_metrics['precision']:.4f}")
        print(f"Recall:    {irl_metrics['recall']:.4f}")
        print(f"F1 Score:  {irl_metrics['f1']:.4f}")
        print(f"Samples:   {irl_metrics['total']}")
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to {output_path}")


if __name__ == "__main__":
    main()
