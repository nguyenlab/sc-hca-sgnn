import re
import json
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple
from dataclasses import dataclass
import warnings

# Try to import gensim (optional dependency)
try:
    from gensim.models import FastText
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False
    warnings.warn(
        "gensim not installed. FastText embeddings will not be available. "
        "Install with: pip install gensim"
    )


# ============================================================================
# Tokenization
# ============================================================================

def solidity_tokenizer(code_string: str) -> List[str]:
    # First split by whitespace and common delimiters
    tokens = re.split(r'[\s\(\)\{\}\[\]\.;=,<>+\-\*/&\|!?:\^%~]+', code_string)
    tokens = [t for t in tokens if t]
    
    # Further split CamelCase and snake_case
    expanded_tokens = []
    for token in tokens:
        # Split CamelCase: "bankBalance" -> ["bank", "Balance"]
        camel_split = re.sub(r'([a-z])([A-Z])', r'\1 \2', token)
        # Split snake_case and numbers
        parts = re.split(r'[_\d]+', camel_split)
        expanded_tokens.extend([p.lower() for p in parts if p])
    
    return expanded_tokens


def get_node_label(node: Dict) -> str:
    attrs = node.get('attributes', {})
    
    # Try to get name (most semantic information)
    name = attrs.get('name')
    if name:
        return name
    
    # Try value (for literals)
    value = attrs.get('value')
    if value is not None:
        return str(value)
    
    # Try member name (for MemberAccess)
    member_name = attrs.get('member_name') or attrs.get('memberName')
    if member_name:
        return member_name
    
    # Try operator (for operations)
    operator = attrs.get('operator')
    if operator:
        return operator
    
    # Fallback to node type
    return node.get('node_type', 'Unknown')


def extract_node_labels(ast_json: Dict) -> List[str]:
    nodes = ast_json.get('nodes', [])
    return [get_node_label(node) for node in nodes]


# ============================================================================
# FastText Model Training
# ============================================================================

@dataclass
class FastTextConfig:
    vector_size: int = 100  # Embedding dimension
    window: int = 5         # Context window
    min_count: int = 1      # Minimum word frequency
    epochs: int = 10        # Training epochs
    workers: int = 4        # Parallel workers
    sg: int = 1             # Skip-gram (1) vs CBOW (0)


def train_fasttext_on_solidity(
    corpus_path: Union[str, Path],
    output_path: Union[str, Path],
    config: Optional[FastTextConfig] = None,
    include_ast_labels: bool = True,
    verbose: bool = True,
) -> 'FastText':
    if not GENSIM_AVAILABLE:
        raise ImportError("gensim is required for FastText training. Install with: pip install gensim")
    
    config = config or FastTextConfig()
    corpus_path = Path(corpus_path)
    output_path = Path(output_path)
    
    sentences = []
    
    # Collect source code
    sol_files = list(corpus_path.rglob('*.sol'))
    if verbose:
        print(f"Found {len(sol_files)} Solidity files")
    
    for sol_file in sol_files:
        try:
            with open(sol_file, 'r', encoding='utf-8', errors='ignore') as f:
                code = f.read()
            tokens = solidity_tokenizer(code)
            if tokens:
                sentences.append(tokens)
        except Exception as e:
            if verbose:
                print(f"Warning: Failed to process {sol_file}: {e}")
    
    # Optionally include AST node labels
    if include_ast_labels:
        ast_files = list(corpus_path.rglob('*.json'))
        for ast_file in ast_files:
            try:
                with open(ast_file, 'r') as f:
                    ast_data = json.load(f)
                if 'nodes' in ast_data:
                    labels = extract_node_labels(ast_data)
                    # Tokenize each label
                    for label in labels:
                        tokens = solidity_tokenizer(label)
                        if tokens:
                            sentences.append(tokens)
            except Exception:
                pass
    
    if verbose:
        print(f"Training FastText on {len(sentences)} sentences...")
    
    # Train FastText
    model = FastText(
        sentences=sentences,
        vector_size=config.vector_size,
        window=config.window,
        min_count=config.min_count,
        epochs=config.epochs,
        workers=config.workers,
        sg=config.sg,
    )
    
    # Save model
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(output_path))
    
    if verbose:
        print(f"Model saved to {output_path}")
        print(f"Vocabulary size: {len(model.wv)}")
    
    return model


def load_fasttext_model(model_path: Union[str, Path]) -> 'FastText':
    if not GENSIM_AVAILABLE:
        raise ImportError("gensim is required to load FastText models")
    return FastText.load(str(model_path))


# ============================================================================
# Embedding Generation
# ============================================================================

class FastTextEmbedder:
    
    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        model: Optional['FastText'] = None,
        vector_size: int = 100,
        combine_with_type: bool = True,
        num_node_types: int = 97,
    ):
        self.vector_size = vector_size
        self.combine_with_type = combine_with_type
        self.num_node_types = num_node_types
        
        if model is not None:
            self.model = model
        elif model_path is not None:
            self.model = load_fasttext_model(model_path)
        else:
            self.model = None
            warnings.warn("No FastText model provided. Will use zero vectors.")
        
        if self.model is not None:
            self.vector_size = self.model.wv.vector_size
    
    @property
    def output_dim(self) -> int:
        if self.combine_with_type:
            return self.vector_size + self.num_node_types
        return self.vector_size
    
    def embed_node(self, node: Dict) -> torch.Tensor:
        # Get node label and tokenize
        label = get_node_label(node)
        tokens = solidity_tokenizer(label)
        
        # Get FastText embedding (average over tokens)
        if self.model is not None and tokens:
            vectors = []
            for token in tokens:
                try:
                    vec = self.model.wv[token]
                    vectors.append(vec)
                except KeyError:
                    # FastText handles OOV internally, but just in case
                    pass
            
            if vectors:
                ft_embedding = np.mean(vectors, axis=0)
            else:
                ft_embedding = np.zeros(self.vector_size)
        else:
            ft_embedding = np.zeros(self.vector_size)
        
        ft_tensor = torch.tensor(ft_embedding, dtype=torch.float32)
        
        # Optionally combine with node type one-hot
        if self.combine_with_type:
            node_type_idx = node.get('node_type_idx', 0)
            type_onehot = torch.zeros(self.num_node_types)
            if 0 <= node_type_idx < self.num_node_types:
                type_onehot[node_type_idx] = 1.0
            return torch.cat([ft_tensor, type_onehot])
        
        return ft_tensor
    
    def embed_nodes(self, nodes: List[Dict]) -> torch.Tensor:
        embeddings = [self.embed_node(node) for node in nodes]
        return torch.stack(embeddings)
    
    def embed_graph(self, ast_json: Dict) -> torch.Tensor:
        nodes = ast_json.get('nodes', [])
        if not nodes:
            return torch.zeros((0, self.output_dim))
        return self.embed_nodes(nodes)


# ============================================================================
# Dataset Integration
# ============================================================================

class FastTextFeatureExtractor:
    
    def __init__(
        self,
        embedder: FastTextEmbedder,
        ast_data_dir: Union[str, Path],
        cache_dir: Optional[Union[str, Path]] = None,
    ):
        self.embedder = embedder
        self.ast_data_dir = Path(ast_data_dir)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def extract_features(self, filename: str) -> Optional[torch.Tensor]:
        # Check cache first
        if self.cache_dir:
            cache_path = self.cache_dir / f"{filename}.pt"
            if cache_path.exists():
                return torch.load(cache_path)
        
        # Load AST data
        ast_path = self.ast_data_dir / filename
        if not ast_path.exists():
            return None
        
        with open(ast_path, 'r') as f:
            ast_json = json.load(f)
        
        # Extract features
        features = self.embedder.embed_graph(ast_json)
        
        # Cache if enabled
        if self.cache_dir:
            torch.save(features, cache_path)
        
        return features
    
    def precompute_all(self, verbose: bool = True) -> int:
        if not self.cache_dir:
            raise ValueError("cache_dir must be set to precompute features")
        
        ast_files = list(self.ast_data_dir.glob('*.json'))
        
        if verbose:
            from tqdm import tqdm
            ast_files = tqdm(ast_files, desc="Computing FastText features")
        
        count = 0
        for ast_file in ast_files:
            self.extract_features(ast_file.name)
            count += 1
        
        return count


# ============================================================================
# CLI Interface
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='FastText utilities for smart contract analysis')
    parser.add_argument('--train', action='store_true', help='Train a new FastText model')
    parser.add_argument('--embed', action='store_true', help='Generate embeddings for AST data')
    parser.add_argument('--corpus', type=str, help='Path to Solidity corpus (for training)')
    parser.add_argument('--ast-data', type=str, help='Path to AST data directory')
    parser.add_argument('--model', type=str, default='models/fasttext/solidity.model',
                       help='Path to FastText model')
    parser.add_argument('--output', type=str, help='Output path')
    parser.add_argument('--vector-size', type=int, default=100, help='Embedding dimension')
    parser.add_argument('--epochs', type=int, default=10, help='Training epochs')
    
    args = parser.parse_args()
    
    if args.train:
        if not args.corpus:
            parser.error("--corpus is required for training")
        
        output = args.output or args.model
        config = FastTextConfig(vector_size=args.vector_size, epochs=args.epochs)
        train_fasttext_on_solidity(args.corpus, output, config)
    
    elif args.embed:
        if not args.ast_data:
            parser.error("--ast-data is required for embedding")
        if not args.output:
            parser.error("--output is required for embedding")
        
        embedder = FastTextEmbedder(model_path=args.model)
        extractor = FastTextFeatureExtractor(embedder, args.ast_data, args.output)
        count = extractor.precompute_all()
        print(f"Processed {count} files")
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
