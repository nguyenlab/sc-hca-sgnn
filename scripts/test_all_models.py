import torch
from models import (
    create_model,
    create_dr_gcn_model,
    create_tmp_model,
    create_gat_model,
    create_transformer_gnn_model,
    create_bugsweeper_model,
    create_bugsweeper_light_model,
    create_scvhunter_model,
    create_mlagnn_model,
    ContractGraphDataset,
)
from torch_geometric.loader import DataLoader
import torch.nn as nn


def test_model(name, model, loader, device='cpu'):
    """Test a model with basic operations."""
    print(f"\n{'='*60}")
    print(f"Testing {name}")
    print(f"{'='*60}")
    
    model = model.to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params:,}")
    
    # Test forward pass
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, 
                          batch.edge_attr if hasattr(batch, 'edge_attr') else None,
                          batch.batch)
            print(f"Forward pass: {batch.num_graphs} graphs -> {logits.shape}")
            break
    
    # Test training step
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    batch = next(iter(loader)).to(device)
    optimizer.zero_grad()
    logits = model(batch.x, batch.edge_index,
                   batch.edge_attr if hasattr(batch, 'edge_attr') else None,
                   batch.batch)
    labels = batch.graph_y.squeeze()
    # Binary classification
    labels = (labels > 0).long()
    loss = criterion(logits, labels)
    loss.backward()
    optimizer.step()
    print(f"Training step: loss={loss.item():.4f}")
    
    print(f"✓ {name} passed all tests")
    return num_params


def main():
    print("="*60)
    print("GNN Model Test Suite")
    print("="*60)
    
    # Load dataset
    print("\nLoading dataset...")
    dataset = ContractGraphDataset(root='data/ast_dataset', use_edge_attr=True)
    loader = DataLoader(dataset[:20], batch_size=8, shuffle=True)
    print(f"Loaded {len(dataset)} graphs")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    results = {}
    
    # Test RGCN models
    print("\n" + "="*60)
    print("RGCN-Based Models")
    print("="*60)
    
    model = create_model('hierarchical', input_dim=100, hidden_dim=64, 
                        num_layers=3, num_classes=2, num_edge_types=8, mode='rgcn')
    results['Hierarchical RGCN'] = test_model('Hierarchical RGCN', model, loader, device)
    
    # Test DR-GCN
    print("\n" + "="*60)
    print("IJCAI 2020 Models")
    print("="*60)
    
    model = create_dr_gcn_model(input_dim=100, hidden_dim=64, num_layers=3, num_classes=2)
    results['DR-GCN'] = test_model('DR-GCN', model, loader, device)
    
    model = create_tmp_model(input_dim=100, hidden_dim=64, num_layers=3, 
                            heads=4, num_classes=2)
    results['TMP'] = test_model('TMP', model, loader, device)
    
    # Test attention models
    print("\n" + "="*60)
    print("Attention-Based Models")
    print("="*60)
    
    model = create_gat_model(input_dim=100, hidden_dim=64, num_layers=3,
                            heads=4, num_classes=2)
    results['GAT'] = test_model('GAT', model, loader, device)
    
    model = create_transformer_gnn_model(input_dim=100, hidden_dim=64, num_layers=3,
                                        heads=4, num_classes=2, use_positional_encoding=True)
    results['TransformerGNN'] = test_model('TransformerGNN (with PE)', model, loader, device)
    
    model = create_transformer_gnn_model(input_dim=100, hidden_dim=64, num_layers=3,
                                        heads=4, num_classes=2, use_positional_encoding=False)
    results['TransformerGNN (no PE)'] = test_model('TransformerGNN (no PE)', model, loader, device)
    
    # Test BugSweeper models
    print("\n" + "="*60)
    print("BugSweeper Models (AAAI 2026)")
    print("="*60)
    
    model = create_bugsweeper_model(input_dim=100, num_classes=2, dropout=0.3)
    results['BugSweeper'] = test_model('BugSweeper', model, loader, device)
    
    model = create_bugsweeper_light_model(input_dim=100, hidden_dim=64, num_classes=2, dropout=0.3)
    results['BugSweeperLight'] = test_model('BugSweeperLight', model, loader, device)
    
    # Test 2024 Models (SCVHUNTER and ML-AGNN)
    print("\n" + "="*60)
    print("2024 Models (SCVHUNTER, ML-AGNN)")
    print("="*60)
    
    model = create_scvhunter_model(input_dim=100, hidden_dim=64, num_layers=3,
                                   heads=4, num_classes=2, dropout=0.3)
    results['SCVHUNTER'] = test_model('SCVHUNTER (ICSE 2024)', model, loader, device)
    
    model = create_mlagnn_model(input_dim=100, hidden_dim=64, num_layers=3,
                                heads=4, num_classes=2, dropout=0.3)
    results['ML-AGNN'] = test_model('ML-AGNN (MSN 2024)', model, loader, device)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"\n{'Model':<30} {'Parameters':>15}")
    print("-"*60)
    for name, params in results.items():
        print(f"{name:<30} {params:>15,}")
    
    print("\n" + "="*60)
    print("✓ All tests passed!")
    print("="*60)


if __name__ == '__main__':
    main()
