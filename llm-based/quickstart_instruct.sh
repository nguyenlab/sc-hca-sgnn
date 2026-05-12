#!/bin/bash
# Quick start script for instruction fine-tuning with llama-factory

set -e  # Exit on error

echo "========================================"
echo "Instruction Fine-tuning Quick Start"
echo "========================================"
echo ""

# Step 1: Prepare data
echo "Step 1: Preparing instruction tuning data..."
cd /home/minhnn/kagayaki/git/sc-vuln-detection/llm-based
python prepare_instruction_data.py \
    --synthetic-data ../data/synthetic/ast_dataset \
    --realworld-data ../data/realworld/ast_dataset \
    --output-dir data \
    --max-train-samples 1000
echo ""

# Step 2: Verify data files
echo "Step 2: Verifying data files..."
for file in data/sc_synthetic_train.json data/sc_synthetic_val.json data/sc_synthetic_test.json; do
    if [ -f "$file" ]; then
        count=$(jq '. | length' "$file")
        echo "  ✓ $file: $count samples"
    else
        echo "  ✗ $file: NOT FOUND"
    fi
done
echo ""

# Step 3: Train model
echo "Step 3: Starting fine-tuning..."
echo "  Model: Qwen/Qwen2.5-Coder-7B-Instruct"
echo "  Method: LoRA (rank=8)"
echo "  Epochs: 3"
echo ""
echo "Run the training script:"
echo "  bash run_instruct_sft.sh"
echo ""
echo "Or customize parameters:"
echo "  llamafactory-cli train --help"
echo ""

# Step 4: Evaluation (after training)
echo "Step 4: After training completes, evaluate with:"
echo "  python evaluate_instruct_model.py \\"
echo "    --model-path outputs/instruct_sft/qwen2_5-coder-7b/lora/sft \\"
echo "    --synthetic-test data/sc_synthetic_test.json \\"
echo "    --realworld-test data/sc_realworld_test.json"
echo ""

echo "========================================"
echo "Setup complete! Ready to train."
echo "========================================"
