import argparse
import sys
from pathlib import Path


def cmd_build(args):
    """Build AST dataset from smart contracts."""
    from dataset_builder import DatasetBuilder
    
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    clean_dir = Path(args.clean) if args.clean else None
    
    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        return 1
    
    if clean_dir and not clean_dir.exists():
        print(f"Error: Clean contracts directory not found: {clean_dir}")
        return 1
    
    builder = DatasetBuilder(input_dir, output_dir, clean_dir, args.max_clean)
    stats = builder.build_dataset(verbose=not args.quiet)
    
    return 1 if stats['failed'] > 0 else 0


def cmd_train_binary(args):
    """Train binary (clean vs vulnerable) classifier."""
    import torch
    from models.data import (
        ContractGraphDataset,
        create_data_loaders,
        get_input_dim,
        get_num_edge_types,
        NUM_NODE_TYPES,
        NUM_EDGE_TYPES,
    )
    from models.graph_level.multi_edge_gcn import create_model
    from training.utils import (
        setup_seed,
        setup_device,
        train_with_early_stopping,
        evaluate_binary,
        print_binary_results,
        save_results,
        compute_binary_weights,
    )
    
    setup_seed(args.seed)
    device = setup_device()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Device: {device}")
    print(f"Task: Binary Classification (Clean vs Vulnerable)")
    print(f"Configuration:")
    print(f"  Model: {args.model_type} ({args.mode})")
    print(f"  Hidden dim: {args.hidden_dim}, Layers: {args.num_layers}")
    print(f"  Node types: {NUM_NODE_TYPES}, Edge types: {NUM_EDGE_TYPES}")
    
    # Load dataset
    print(f"\nLoading dataset from {args.data}...")
    dataset = ContractGraphDataset(root=args.data, use_edge_attr=True, bidirectional=True)
    stats = dataset.get_stats()
    
    clean_count = stats.vuln_distribution.get('clean', 0)
    vuln_count = sum(stats.vuln_distribution.values()) - clean_count
    print(f"  Graphs: {stats.num_graphs} (clean: {clean_count}, vulnerable: {vuln_count})")
    
    class_weights = compute_binary_weights(clean_count, vuln_count)
    print(f"  Class weights: clean={class_weights[0]:.2f}, vuln={class_weights[1]:.2f}")
    
    train_loader, val_loader, test_loader = create_data_loaders(
        dataset, batch_size=args.batch_size, seed=args.seed
    )
    print(f"  Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")
    
    # Create model
    if args.model_type == 'dr-gcn':
        from models.graph_level.dr_gcn import create_dr_gcn_model
        model = create_dr_gcn_model(
            input_dim=get_input_dim(),
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_classes=2,
            dropout=args.dropout,
            pooling=args.pooling,
        ).to(device)
    elif args.model_type == 'tmp':
        from models.graph_level.dr_gcn import create_tmp_model
        model = create_tmp_model(
            input_dim=get_input_dim(),
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            heads=args.heads,
            num_classes=2,
            dropout=args.dropout,
            pooling=args.pooling,
        ).to(device)
    elif args.model_type == 'gat':
        from models.graph_level.attention_models import create_gat_model
        model = create_gat_model(
            input_dim=get_input_dim(),
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            heads=args.heads,
            num_classes=2,
            dropout=args.dropout,
            pooling=args.pooling,
        ).to(device)
    elif args.model_type == 'transformer':
        from models.graph_level.attention_models import create_transformer_gnn_model
        model = create_transformer_gnn_model(
            input_dim=get_input_dim(),
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            heads=args.heads,
            num_classes=2,
            dropout=args.dropout,
            pooling=args.pooling,
            use_positional_encoding=args.use_pe,
        ).to(device)
    elif args.model_type == 'bugsweeper':
        from models.graph_level.bugsweeper import create_bugsweeper_model
        model = create_bugsweeper_model(
            input_dim=get_input_dim(),
            num_classes=2,
            dropout=args.dropout,
        ).to(device)
    elif args.model_type == 'bugsweeper-light':
        from models.graph_level.bugsweeper import create_bugsweeper_light_model
        model = create_bugsweeper_light_model(
            input_dim=get_input_dim(),
            hidden_dim=args.hidden_dim,
            num_classes=2,
            dropout=args.dropout,
        ).to(device)
    elif args.model_type == 'scvhunter':
        from models.graph_level.scvhunter import create_scvhunter_model
        model = create_scvhunter_model(
            input_dim=get_input_dim(),
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_edge_types=get_num_edge_types(),
            heads=args.heads,
            num_classes=2,
            dropout=args.dropout,
        ).to(device)
    elif args.model_type == 'mlagnn':
        from models.graph_level.mlagnn import create_mlagnn_model
        model = create_mlagnn_model(
            input_dim=get_input_dim(),
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_edge_types=get_num_edge_types(),
            heads=args.heads,
            num_classes=2,
            dropout=args.dropout,
        ).to(device)
    else:
        model = create_model(
            model_type=args.model_type,
            input_dim=get_input_dim(),
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            num_classes=2,
            num_edge_types=get_num_edge_types(),
            mode=args.mode,
        ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {num_params:,}")
    
    # Train
    print("\nTraining...")
    _, history = train_with_early_stopping(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
        output_dir=output_dir,
        class_weights=class_weights,
        binary=True,
    )
    
    # Evaluate
    import torch.nn as nn
    criterion = nn.CrossEntropyLoss()
    test_metrics = evaluate_binary(model, test_loader, criterion, device)
    print_binary_results(test_metrics)
    
    save_results(output_dir, vars(args), test_metrics, num_params)
    return 0


def cmd_train_multiclass(args):
    """Train multiclass vulnerability classifier."""
    import torch
    from models.data import (
        ContractGraphDataset,
        create_data_loaders,
        get_input_dim,
        get_num_classes,
        get_num_edge_types,
        IDX_TO_VULN,
    )
    from models.graph_level.multi_edge_gcn import create_model
    from training.utils import (
        setup_seed,
        setup_device,
        train_with_early_stopping,
        evaluate_multiclass,
        print_multiclass_results,
        save_results,
    )
    
    setup_seed(args.seed)
    device = setup_device()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Device: {device}")
    print(f"Task: Multiclass Classification")
    print(f"Configuration:")
    print(f"  Model: {args.model_type} ({args.mode})")
    print(f"  Hidden dim: {args.hidden_dim}, Layers: {args.num_layers}")
    
    # Load dataset
    print(f"\nLoading dataset from {args.data}...")
    dataset = ContractGraphDataset(root=args.data, use_edge_attr=True, bidirectional=True)
    stats = dataset.get_stats()
    print(f"  Graphs: {stats.num_graphs}")
    print(f"  Vulnerability distribution:")
    for vtype, count in stats.vuln_distribution.items():
        print(f"    {vtype}: {count}")
    
    train_loader, val_loader, test_loader = create_data_loaders(
        dataset, batch_size=args.batch_size, seed=args.seed
    )
    print(f"  Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")
    
    # Create model
    if args.model_type == 'dr-gcn':
        from models.graph_level.dr_gcn import create_dr_gcn_model
        model = create_dr_gcn_model(
            input_dim=get_input_dim(),
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_classes=get_num_classes(),
            dropout=args.dropout,
            pooling=args.pooling,
        ).to(device)
    elif args.model_type == 'tmp':
        from models.graph_level.dr_gcn import create_tmp_model
        model = create_tmp_model(
            input_dim=get_input_dim(),
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            heads=args.heads,
            num_classes=get_num_classes(),
            dropout=args.dropout,
            pooling=args.pooling,
        ).to(device)
    elif args.model_type == 'gat':
        from models.graph_level.attention_models import create_gat_model
        model = create_gat_model(
            input_dim=get_input_dim(),
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            heads=args.heads,
            num_classes=get_num_classes(),
            dropout=args.dropout,
            pooling=args.pooling,
        ).to(device)
    elif args.model_type == 'transformer':
        from models.graph_level.attention_models import create_transformer_gnn_model
        model = create_transformer_gnn_model(
            input_dim=get_input_dim(),
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            heads=args.heads,
            num_classes=get_num_classes(),
            dropout=args.dropout,
            pooling=args.pooling,
            use_positional_encoding=args.use_pe,
        ).to(device)
    elif args.model_type == 'bugsweeper':
        from models.graph_level.bugsweeper import create_bugsweeper_model
        model = create_bugsweeper_model(
            input_dim=get_input_dim(),
            num_classes=get_num_classes(),
            dropout=args.dropout,
        ).to(device)
    elif args.model_type == 'bugsweeper-light':
        from models.graph_level.bugsweeper import create_bugsweeper_light_model
        model = create_bugsweeper_light_model(
            input_dim=get_input_dim(),
            hidden_dim=args.hidden_dim,
            num_classes=get_num_classes(),
            dropout=args.dropout,
        ).to(device)
    elif args.model_type == 'scvhunter':
        from models.graph_level.scvhunter import create_scvhunter_model
        model = create_scvhunter_model(
            input_dim=get_input_dim(),
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_edge_types=get_num_edge_types(),
            heads=args.heads,
            num_classes=get_num_classes(),
            dropout=args.dropout,
        ).to(device)
    elif args.model_type == 'mlagnn':
        from models.graph_level.mlagnn import create_mlagnn_model
        model = create_mlagnn_model(
            input_dim=get_input_dim(),
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_edge_types=get_num_edge_types(),
            heads=args.heads,
            num_classes=get_num_classes(),
            dropout=args.dropout,
        ).to(device)
    else:
        model = create_model(
            model_type=args.model_type,
            input_dim=get_input_dim(),
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            num_classes=get_num_classes(),
            num_edge_types=get_num_edge_types(),
            mode=args.mode,
        ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {num_params:,}")
    
    # Train
    print("\nTraining...")
    _, history = train_with_early_stopping(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
        output_dir=output_dir,
        binary=False,
    )
    
    # Evaluate
    import torch.nn as nn
    criterion = nn.CrossEntropyLoss()
    test_metrics = evaluate_multiclass(model, test_loader, criterion, device)
    print_multiclass_results(test_metrics, IDX_TO_VULN)
    
    save_results(output_dir, vars(args), test_metrics, num_params)
    return 0


def cmd_view_contract(args):
    """View a smart contract and its AST."""
    from view_injected import view_contract
    view_contract(args.path)
    return 0


def add_common_train_args(parser):
    """Add common training arguments."""
    parser.add_argument('--data', '-d', default='data/synthetic/ast_dataset', help='Dataset directory')
    parser.add_argument('--output', '-o', default='outputs', help='Output directory')
    parser.add_argument('--model-type', '-t', default='hierarchical',
                        choices=['graph', 'hierarchical', 'dr-gcn', 'tmp', 'gat', 'transformer', 
                                 'bugsweeper', 'bugsweeper-light', 'scvhunter', 'mlagnn'], 
                        help='Model type (graph/hierarchical=RGCN, dr-gcn/tmp=IJCAI2020, gat/transformer=attention, bugsweeper=AAAI2026, scvhunter/mlagnn=2024)')
    parser.add_argument('--mode', '-m', default='rgcn',
                        choices=['gcn', 'rgcn', 'gat'], help='GNN mode (for graph/hierarchical models)')
    parser.add_argument('--pooling', '-p', default='mean',
                        choices=['mean', 'max', 'both'], help='Graph pooling method')
    parser.add_argument('--heads', type=int, default=4, help='Number of attention heads (for TMP/GAT/Transformer/SCVHUNTER/MLAGNN)')
    parser.add_argument('--use-pe', action='store_true', help='Use positional encoding (for Transformer)')
    parser.add_argument('--hidden-dim', type=int, default=128, help='Hidden dimension')
    parser.add_argument('--num-layers', type=int, default=4, help='Number of GNN layers')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--epochs', type=int, default=100, help='Max epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')


def main():
    parser = argparse.ArgumentParser(
        description='Smart Contract Vulnerability Detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Build command
    build_parser = subparsers.add_parser('build', help='Build AST dataset')
    build_parser.add_argument('--input', '-i', required=True, help='Input contracts directory')
    build_parser.add_argument('--output', '-o', required=True, help='Output dataset directory')
    build_parser.add_argument('--clean', '-c', default=None, help='Clean contracts directory')
    build_parser.add_argument('--max-clean', type=int, default=None, help='Max clean contracts')
    build_parser.add_argument('--quiet', '-q', action='store_true', help='Quiet mode')
    
    # Train command with subcommands
    train_parser = subparsers.add_parser('train', help='Train models')
    train_subparsers = train_parser.add_subparsers(dest='train_type', help='Training type')
    
    # Train binary
    binary_parser = train_subparsers.add_parser('binary', help='Binary classification')
    add_common_train_args(binary_parser)
    binary_parser.set_defaults(output='outputs/binary')
    
    # Train multiclass
    multi_parser = train_subparsers.add_parser('multiclass', help='Multiclass classification')
    add_common_train_args(multi_parser)
    multi_parser.set_defaults(output='outputs/multiclass')
    
    # View command
    view_parser = subparsers.add_parser('view', help='View contracts/data')
    view_subparsers = view_parser.add_subparsers(dest='view_type', help='View type')
    
    contract_parser = view_subparsers.add_parser('contract', help='View a contract')
    contract_parser.add_argument('path', help='Path to contract file')
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return 0
    
    if args.command == 'build':
        return cmd_build(args)
    elif args.command == 'train':
        if args.train_type == 'binary':
            return cmd_train_binary(args)
        elif args.train_type == 'multiclass':
            return cmd_train_multiclass(args)
        else:
            train_parser.print_help()
            return 0
    elif args.command == 'view':
        if args.view_type == 'contract':
            return cmd_view_contract(args)
        else:
            view_parser.print_help()
            return 0
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
