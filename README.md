# Smart Contract Vulnerability Detection

A comprehensive framework for detecting vulnerabilities in Solidity smart contracts using **Graph Neural Networks (GNNs)** on Abstract Syntax Trees (ASTs), with **LLM-based** approaches for comparison.

## Overview

This project transforms Solidity source code into rich multi-edge AST graphs and applies various GNN architectures to detect vulnerabilities at both the **contract level** (graph classification) and **function level** (subgraph classification).

### Key Features

- **Multi-edge AST representation** — 8 edge types: AST structure, reference/data flow, control flow (next/true/false branches), function calls, inheritance, and guard conditions
- **10+ GNN architectures** — from standard GCN/GAT to specialized models (DR-GCN, TMP, BugSweeper, SCVHunter, MLAGNN, BSGVD, HCA-SGNN)
- **Multi-level analysis** — graph-level (whole contract) and subgraph-level (function-level) detection
- **Binary & multiclass classification** — detect whether a contract is vulnerable, or identify the specific vulnerability type
- **LLM comparison** — CodeLlama-based detection with LoRA fine-tuning for fair benchmarking
- **Comprehensive evaluation** — benchmarking, ablation studies, grid search, and cross-dataset testing (synthetic → real-world)

### Supported Vulnerability Types

| Synthetic | Real-World (SmartBugs) |
|-----------|----------------------|
| Overflow / Underflow | Access Control |
| Reentrancy | Bad Randomness |
| Timestamp Dependency | Denial of Service |
| `tx.origin` | Front Running |
| Unchecked Send | Short Addresses |
| Unhandled Exception | Other |
| Cross-function Reentrancy | |

## Architecture

```
Solidity Source Code (.sol)
    │
    ▼
┌──────────────────────────┐
│   Solidity Compiler (solc)│  ← py-solc-x manages versions
│   AST Extraction          │
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│   Dataset Builder         │  ← dataset_builder.py
│   • Node type encoding    │     (90+ Solidity AST node types)
│   • Multi-edge extraction │     (AST + CFG + data flow + calls)
│   • Vulnerability labeling│
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│   PyTorch Geometric       │
│   Graph Dataset           │  ← models/data.py
└──────────┬───────────────┘
           │
     ┌─────┴─────┐
     ▼           ▼
 Graph-Level  Subgraph-Level
  Models       (Function)
```

## Models

### Graph-Level Models (`models/graph_level/`)

| Model | Description | Reference |
|-------|-------------|-----------|
| **GCN / RGCN** | Standard and relational graph convolution | Baseline |
| **Multi-Edge GCN** | Hierarchical GCN with edge-type awareness | — |
| **DR-GCN** | Dual-regularized GCN for imbalanced classification | IJCAI 2020 |
| **TMP** | Temporal Message Passing with attention | IJCAI 2020 |
| **GAT** | Graph Attention Network | — |
| **Transformer GNN** | Graph Transformer with optional positional encoding | — |
| **BugSweeper** | Two-stage code graph neural network with pooling | AAAI 2026 |
| **SCVHunter** | Heterogeneous attention with node importance | 2024 |
| **MLAGNN** | Multi-Level Adaptive Attention GNN | 2024 |
| **BSGVD** | Bimodal detection with FastText + GATv2 | — |

### Subgraph-Level Models (`models/subgraph_level/`)

| Model | Description |
|-------|-------------|
| **Function-Level GNN** | Dual-path encoder with function pooling |
| **HCA-SGNN** | Hierarchical Cross-Attention Subgraph GNN with gated local-global fusion |
| + Function-level variants of DR-GCN, TMP, GAT, Transformer, SCVHunter, MLAGNN |

## Installation

### Prerequisites

- Python 3.9+
- CUDA 11.8 or 12.1 (recommended for GPU acceleration)

### Setup

```bash
# 1. Create conda environment
conda create -n sc-dev python=3.11
conda activate sc-dev

# 2. Install Solidity compiler wrapper
pip install py-solc-x

# 3. Install PyTorch (adjust for your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 4. Install PyTorch Geometric with pre-built wheels
pip install torch-geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
  -f https://data.pyg.org/whl/torch-2.0.0+cu118.html

# 5. Install remaining dependencies
pip install -r requirements.txt
```

> [!NOTE]
> Adjust the PyG wheel URL to match your PyTorch version. Check with: `python -c "import torch; print(torch.__version__)"`. For CPU-only: use `torch-2.0.0+cpu.html`.

## Usage

### 1. Build Dataset

Extract AST graphs from Solidity contracts:

```bash
python main.py build \
  --input data/synthetic/contracts \
  --output data/synthetic/ast_dataset \
  --clean data/synthetic/clean_contracts
```

### 2. Train Models

**Binary classification** (clean vs. vulnerable):

```bash
python main.py train binary \
  --data data/synthetic/ast_dataset \
  --model-type hierarchical \
  --mode rgcn \
  --hidden-dim 128 \
  --num-layers 4 \
  --epochs 100
```

**Multiclass classification** (identify vulnerability type):

```bash
python main.py train multiclass \
  --data data/synthetic/ast_dataset \
  --model-type gat \
  --heads 4 \
  --hidden-dim 128
```

#### Available Model Types

```
graph, hierarchical    — RGCN-based models
dr-gcn, tmp            — IJCAI 2020 models
gat, transformer       — Attention-based models
bugsweeper             — AAAI 2026 BugSweeper
bugsweeper-light       — Lightweight variant
scvhunter, mlagnn      — 2024 specialized models
```

#### Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model-type` | `hierarchical` | GNN architecture |
| `--mode` | `rgcn` | GNN mode (`gcn`, `rgcn`, `gat`) |
| `--hidden-dim` | `128` | Hidden dimension size |
| `--num-layers` | `4` | Number of GNN layers |
| `--heads` | `4` | Attention heads (for GAT/Transformer/TMP) |
| `--dropout` | `0.1` | Dropout rate |
| `--pooling` | `mean` | Graph pooling (`mean`, `max`, `both`) |
| `--epochs` | `100` | Maximum training epochs |
| `--lr` | `0.001` | Learning rate |
| `--batch-size` | `32` | Batch size |
| `--patience` | `20` | Early stopping patience |
| `--seed` | `42` | Random seed |

### 3. Benchmark All Models

Run a comprehensive benchmark across all model architectures:

```bash
python scripts/benchmark.py \
  --synthetic-data data/synthetic/ast_dataset \
  --realworld-data data/realworld/ast_dataset \
  --output results/benchmark
```

### 4. Ablation Study

Evaluate the contribution of each component in HCA-SGNN:

```bash
python scripts/ablation_study.py \
  --synthetic-data data/synthetic/function_level_dataset \
  --realworld-data data/realworld/function_level_dataset \
  --output results/ablation_study.json
```

### 5. Grid Search

Find optimal hyperparameters:

```bash
bash scripts/run_grid_search.sh
```

### 6. LLM-Based Comparison

Fine-tune CodeLlama for vulnerability detection:

```bash
# Install LLM dependencies
pip install -r llm-based/requirements.txt

# Fine-tune with instruction tuning
python llm-based/contract_level_instruct_finetune.py \
  --data data/synthetic/ast_dataset \
  --irl-data data/realworld/ast_dataset \
  --output llm-based/outputs/instruct_simple \
  --prompt-type simple \
  --epochs 10
```

See [llm-based/README.md](llm-based/README.md) for details.

## Project Structure

```
sc-vuln-detection/
├── main.py                     # CLI entry point (build, train, view)
├── dataset_builder.py          # Solidity → AST graph conversion
├── requirements.txt            # Python dependencies
│
├── scripts/                    # Standalone tool scripts
│   ├── benchmark.py            # Multi-model benchmarking
│   ├── function_level_benchmark.py # Function-level model benchmark
│   ├── ablation_study.py       # HCA-SGNN component analysis
│   ├── grid_search.py          # Hyperparameter optimization
│   ├── run_grid_search.sh      # Grid search runner
│   ├── function_level_builder.py  # Build function-level dataset
│   ├── split_by_template.py    # Split dataset by template
│   ├── split_function_level.py # Split function-level dataset
│   ├── function_level_stats.py # Function-level dataset statistics
│   ├── slither_dataset_stats.py # Slither dataset statistics
│   ├── analyze_template_overlap.py # Template overlap analysis
│   ├── generate_graph_size_analysis.py
│   ├── generate_per_template_metrics.py
│   ├── generate_per_type_predictions.py
│   ├── test_all_models.py      # Smoke test all model architectures
│   ├── validate_paths.py       # Validate data folder structure
│   ├── verify_split.py         # Verify dataset split integrity
│   ├── view_injected.py        # View injected vulnerabilities
│   └── visualize_graph.py      # Visualize AST graphs
│
├── models/
│   ├── data.py                 # Dataset class & feature extraction
│   ├── function_level_data.py  # Function-level dataset
│   ├── bsgvd_data.py           # BSGVD-specific data loader
│   ├── graph_level/            # Graph-level GNN models
│   │   ├── gcn.py              # GCN / RGCN baseline
│   │   ├── multi_edge_gcn.py   # Hierarchical multi-edge GCN
│   │   ├── dr_gcn.py           # DR-GCN & TMP
│   │   ├── attention_models.py # GAT & Transformer GNN
│   │   ├── bugsweeper.py       # BugSweeper & BugSweeper-Light
│   │   ├── scvhunter.py        # SCVHunter
│   │   ├── mlagnn.py           # MLAGNN
│   │   ├── bsgvd.py            # BSGVD bimodal model
│   │   └── fasttext_embedding.py
│   └── subgraph_level/         # Function-level GNN models
│       ├── function_level_gnn.py
│       ├── hierarchical_cross_attention.py  # HCA-SGNN
│       ├── dr_gcn.py
│       ├── attention_models.py
│       ├── scvhunter.py
│       └── mlagnn.py
│
├── training/                   # Training & evaluation utilities
├── testing/                    # Compilation & validation tools
├── src/                        # Core utilities (AST extraction)
│
├── data/
│   ├── synthetic/              # Synthetic vulnerability datasets
│   └── realworld/              # Real-world SmartBugs datasets
│
└── llm-based/                  # LLM comparison (CodeLlama)
    ├── README.md
    ├── requirements.txt
    └── ...
```

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{sc-vuln-detection,
  title={Smart Contract Vulnerability Detection with Graph Neural Networks},
  url={https://github.com/your-username/sc-vuln-detection}
}
```

## License

This project is for research purposes.
