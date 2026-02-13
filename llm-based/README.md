# LLM-Based Vulnerability Detection

This directory contains LLM-based approaches for smart contract vulnerability detection, completely separate from the GNN models in the parent directory.

## Overview

We fine-tune large language models (CodeLlama) for vulnerability detection at:
- **Contract-level**: Classify entire smart contracts as vulnerable or clean
- **Function-level**: Classify individual functions (future work)

Two fine-tuning approaches:
1. **Classification Head** (`contract_level_finetune.py`): Direct binary classification
2. **Instruction Tuning** (`contract_level_instruct_finetune.py`): Natural language prompts ⭐ **NEW**

Uses the **same datasets and train/val/test splits** as GNN approaches for fair comparison.

## Setup

### Prerequisites
First ensure the main conda environment is set up (see parent README):
```bash
conda activate sc-dev
```

### Install LLM Dependencies
```bash
pip install -r llm-based/requirements.txt
```

This installs:
- `transformers` - HuggingFace models
- `peft` - LoRA for efficient finetuning
- `accelerate` - Distributed training
- `bitsandbytes` - Quantization support

## Contract-Level Detection

### Two Approaches

#### 1. Classification Head (Original)
Direct binary classification with a classification head:

```bash
conda activate sc-dev

python llm-based/contract_level_finetune.py \
    --data data/synthetic/ast_dataset \
    --irl-data data/realworld/ast_dataset \
    --output llm-based/outputs/contract_level_codellama \
    --epochs 10
```

#### 2. Instruction Tuning ⭐ **Recommended**
Fine-tune with natural language prompts (works with instruct models):

```bash
# Simple prompt (most efficient)
python llm-based/contract_level_instruct_finetune.py \
    --data data/synthetic/ast_dataset \
    --irl-data data/realworld/ast_dataset \
    --output llm-based/outputs/instruct_simple \
    --prompt-type simple \
    --epochs 10

# Compare all prompt types
python llm-based/compare_prompts.py \
    --data data/synthetic/ast_dataset \
    --epochs 5
```

**See [INSTRUCTION_TUNING.md](INSTRUCTION_TUNING.md) for detailed guide on instruction-based fine-tuning.**

### Quick Start

```bash
# Activate environment
conda activate sc-dev

# Fine-tune CodeLlama on synthetic data
python llm-based/contract_level_finetune.py \
    --data data/synthetic/ast_dataset \
    --output llm-based/outputs/contract_level_codellama \
    --epochs 10

# With real-world (IRL) test set for two-dataset evaluation
python llm-based/contract_level_finetune.py \
    --data data/synthetic/ast_dataset \
    --irl-data data/realworld/ast_dataset \
    --output llm-based/outputs/contract_level_codellama \
    --epochs 10
```

### Arguments

- `--data`: Path to synthetic ast_dataset (required)
- `--irl-data`: Path to real-world dataset for IRL testing (optional)
- `--output`: Output directory for model and results
- `--epochs`: Number of training epochs (default: 10)
- `--batch-size`: Batch size per device (default: 4)
- `--lr`: Learning rate (default: 2e-5)
- `--no-lora`: Disable LoRA for full finetuning (not recommended)

### How It Works

1. **Data Loading**: Loads Solidity source code from the same `ast_dataset` used by GNN models
2. **Preprocessing**: Tokenizes source code with CodeLlama tokenizer (max 2048 tokens)
3. **Fine-tuning**: Uses LoRA for efficient training (only ~0.8% of parameters trained)
4. **Evaluation**: Reports accuracy, precision, recall, F1, AUC on both synthetic and IRL test sets

### Output Structure

```
llm-based/outputs/contract_level_codellama/
├── adapter_model.bin          # LoRA adapter weights
├── adapter_config.json        # LoRA configuration
├── pytorch_model.bin          # Full model (if not using LoRA)
├── config.json                # Model config
├── tokenizer.json             # Tokenizer
├── results.json               # Training metrics
└── logs/                      # Training logs
    └── events.out.tfevents.*
```

### Results Format

`results.json` contains:
```json
{
  "timestamp": "2026-02-01T...",
  "config": {
    "model_name": "codellama/CodeLlama-7b-hf",
    "epochs": 10,
    "batch_size": 4,
    "learning_rate": 2e-5,
    ...
  },
  "synthetic_test": {
    "accuracy": 0.85,
    "precision": 0.83,
    "recall": 0.87,
    "f1": 0.85,
    "auc": 0.92
  },
  "irl_test": {
    "accuracy": 0.78,
    "precision": 0.76,
    "recall": 0.81,
    "f1": 0.78,
    "auc": 0.86
  }
}
```

## Architecture

### Contract-Level Pipeline

```
Solidity Source Code
    ↓
Tokenization (CodeLlama tokenizer, max 2048 tokens)
    ↓
CodeLlama-7B Encoder (with LoRA adapters)
    ↓
Classification Head (2 classes: clean/vulnerable)
    ↓
Prediction: 0 (clean) or 1 (vulnerable)
```

### Key Components

- **`config.py`**: Configuration classes for contract/function-level models
- **`contract_level_data.py`**: Data loader for contract-level task
- **`contract_level_finetune.py`**: Training script for contract-level model
- **`__init__.py`**: Module exports

### LoRA Configuration

Default LoRA settings for efficient finetuning:
- `r=16`: Rank of adaptation matrices
- `alpha=32`: Scaling factor
- `dropout=0.05`: Dropout probability
- Target modules: `q_proj`, `v_proj`, `k_proj`, `o_proj` (attention layers)

This trains only ~55M parameters instead of 7B (0.8% of total).

## Function-Level Detection (Future Work)

Function-level implementation will follow the same pattern:
- `function_level_data.py`: Load individual functions with labels
- `function_level_finetune.py`: Training script
- Use shorter context (1024 tokens) since functions are smaller

## Comparison with GNN Models

| Aspect | GNN Models | LLM Models |
|--------|-----------|------------|
| Input | AST graph structure | Raw source code |
| Model Size | 100K - 10M params | 7B params |
| Training Time | Minutes | Hours |
| Memory | 2-8 GB GPU | 16+ GB GPU |
| Interpretability | Graph attention | Token attention |

## Tips

### Memory Optimization
- Use LoRA (enabled by default) to reduce memory
- Reduce batch size if OOM: `--batch-size 2`
- Enable gradient checkpointing for larger models
- Use 8-bit quantization (requires `bitsandbytes`)

### Training Speed
- Use mixed precision (fp16, enabled by default)
- Increase gradient accumulation for larger effective batch size
- Use multiple GPUs with accelerate

### Common Issues

**"CUDA out of memory"**
- Reduce `--batch-size` to 2 or 1
- Enable 8-bit quantization in config
- Use smaller model (CodeLlama-1B)

**"ModuleNotFoundError: No module named 'transformers'"**
- Install LLM dependencies: `pip install -r llm-based/requirements.txt`
- Make sure conda environment is activated: `conda activate sc-dev`

**"No source code found" warnings**
- Rebuild dataset with source code preservation
- Check that ast_dataset contains source_code field in JSON files

## Future Extensions

- [ ] Function-level vulnerability detection
- [ ] Multi-task learning (binary + multiclass)
- [ ] Larger models (CodeLlama-13B, 34B)
- [ ] Few-shot learning for rare vulnerability types
- [ ] Ensemble with GNN models
- [ ] Explanation generation using LLM
