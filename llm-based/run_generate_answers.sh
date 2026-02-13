#!/bin/bash
# Generate function-level answers from all fine-tuned models on test sets.
#
# For each contract, classifies every function individually, then reports:
#   - Micro (function-level) metrics: P, R, F1 per function
#   - Macro (contract-level) metrics: contract is vulnerable if ANY function is
#
# Usage:
#   cd sc-vuln-detection/llm-based
#   bash run_generate_answers.sh
#
# Each model's results are saved under:
#   outputs/instruct_sft/<model>/lora/sft/eval_results/
#     ├── synthetic_test_answers.json   (per-function predictions + metrics)
#     ├── realworld_test_answers.json   (per-function predictions + metrics)
#     └── metrics_summary.json          (micro + macro metrics)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================"
echo " Generate Answers for All Trained Models"
echo "========================================"
echo ""

# ---------------------------------------------------------------
# Model 1: Qwen2.5-Coder-7B-Instruct
# ---------------------------------------------------------------
echo "=========================================="
echo " [1/3] Qwen2.5-Coder-7B-Instruct (LoRA)"
echo "=========================================="
python generate_answers.py \
    --model-path outputs/instruct_sft/qwen2_5-coder-7b/lora/sft \
    --synthetic-test data/sc_synthetic_test.json \
    --realworld-test data/sc_realworld_test.json \
    --temperature 0.1 \
    --max-new-tokens 128
echo ""

# ---------------------------------------------------------------
# Model 2: Qwen2.5-7B-Instruct
# ---------------------------------------------------------------
echo "=========================================="
echo " [2/3] Qwen2.5-7B-Instruct (LoRA)"
echo "=========================================="
python generate_answers.py \
    --model-path outputs/instruct_sft/qwen2_5-7b/lora/sft \
    --synthetic-test data/sc_synthetic_test.json \
    --realworld-test data/sc_realworld_test.json \
    --temperature 0.1 \
    --max-new-tokens 128
echo ""

# ---------------------------------------------------------------
# Model 3: Llama-3.1-8B-Instruct
# ---------------------------------------------------------------
echo "=========================================="
echo " [3/3] Llama-3.1-8B-Instruct (LoRA)"
echo "=========================================="
python generate_answers.py \
    --model-path outputs/instruct_sft/llama3_1_8b_instruct/lora/sft \
    --synthetic-test data/sc_synthetic_test.json \
    --realworld-test data/sc_realworld_test.json \
    --temperature 0.1 \
    --max-new-tokens 128
echo ""

# ---------------------------------------------------------------
# Summary
# ---------------------------------------------------------------
echo "========================================"
echo " All Done! Results saved under:"
echo "========================================"
echo "  outputs/instruct_sft/qwen2_5-coder-7b/lora/sft/eval_results/"
echo "  outputs/instruct_sft/qwen2_5-7b/lora/sft/eval_results/"
echo "  outputs/instruct_sft/llama3_1_8b_instruct/lora/sft/eval_results/"
echo ""
echo "Each directory contains:"
echo "  - synthetic_test_answers.json   (632 samples)"
echo "  - realworld_test_answers.json   (242 samples)"
echo "  - metrics_summary.json"
