import argparse
import json
import random
from pathlib import Path
import matplotlib.pyplot as plt
import networkx as nx
from collections import Counter


def load_graph_json(path: Path) -> dict:
    with open(path, 'r') as f:
        return json.load(f)


def visualize_ast_graph(data: dict, output_path: str = None, max_nodes: int = 200):
    """Visualize an AST graph."""
    nodes = data['nodes']
    edges = data['edges']
    vuln_type = data.get('vulnerability_type', 'unknown')
    contract_path = data.get('contract_path', 'unknown')
    
    # Create NetworkX graph
    G = nx.DiGraph()
    
    # Add nodes with attributes
    for i, node in enumerate(nodes):
        G.add_node(
            i,
            node_type=node.get('node_type', 'Unknown'),
            is_vulnerable=node.get('is_vulnerable', False),
        )
    
    # Add edges
    for src, dst in edges:
        if src < len(nodes) and dst < len(nodes):
            G.add_edge(src, dst)
    
    # If graph is too large, sample a subgraph
    if len(G.nodes) > max_nodes:
        print(f"Graph has {len(G.nodes)} nodes, sampling {max_nodes} nodes for visualization...")
        # Try to include vulnerable nodes
        vuln_nodes = [n for n in G.nodes if G.nodes[n]['is_vulnerable']]
        if vuln_nodes:
            # Start BFS from vulnerable nodes
            sample_nodes = set(vuln_nodes[:10])
            for vn in vuln_nodes[:10]:
                # Add predecessors and successors
                sample_nodes.update(list(G.predecessors(vn))[:5])
                sample_nodes.update(list(G.successors(vn))[:5])
                # Add 2-hop neighbors
                for neighbor in list(G.neighbors(vn))[:3]:
                    sample_nodes.update(list(G.neighbors(neighbor))[:3])
            # Fill remaining with random nodes
            remaining = max_nodes - len(sample_nodes)
            if remaining > 0:
                other_nodes = [n for n in G.nodes if n not in sample_nodes]
                sample_nodes.update(random.sample(other_nodes, min(remaining, len(other_nodes))))
        else:
            sample_nodes = set(random.sample(list(G.nodes), max_nodes))
        
        G = G.subgraph(sample_nodes).copy()
    
    # Count node types
    node_types = Counter(G.nodes[n]['node_type'] for n in G.nodes)
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(18, 10))
    
    # Left: Graph visualization
    ax1 = axes[0]
    
    # Node colors based on vulnerability
    node_colors = []
    for n in G.nodes:
        if G.nodes[n]['is_vulnerable']:
            node_colors.append('red')
        else:
            node_colors.append('lightblue')
    
    # Layout
    try:
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    except:
        pos = nx.random_layout(G, seed=42)
    
    # Draw
    nx.draw(
        G, pos, ax=ax1,
        node_color=node_colors,
        node_size=50,
        edge_color='gray',
        alpha=0.7,
        arrows=True,
        arrowsize=5,
        width=0.5,
    )
    
    # Add legend
    ax1.plot([], [], 'o', color='red', label='Vulnerable Node', markersize=10)
    ax1.plot([], [], 'o', color='lightblue', label='Normal Node', markersize=10)
    ax1.legend(loc='upper right')
    
    # Title
    num_vuln = sum(1 for n in G.nodes if G.nodes[n]['is_vulnerable'])
    ax1.set_title(
        f"AST Graph: {vuln_type.upper()}\n"
        f"Nodes: {len(G.nodes)} | Edges: {len(G.edges)} | Vulnerable: {num_vuln}",
        fontsize=12
    )
    
    # Right: Node type distribution
    ax2 = axes[1]
    
    # Sort by count
    sorted_types = sorted(node_types.items(), key=lambda x: x[1], reverse=True)[:15]
    types, counts = zip(*sorted_types) if sorted_types else ([], [])
    
    bars = ax2.barh(range(len(types)), counts, color='steelblue')
    ax2.set_yticks(range(len(types)))
    ax2.set_yticklabels(types)
    ax2.invert_yaxis()
    ax2.set_xlabel('Count')
    ax2.set_title('Top 15 AST Node Types')
    
    # Add count labels
    for bar, count in zip(bars, counts):
        ax2.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
                 str(count), va='center', fontsize=9)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()
    
    return G


def print_graph_info(data: dict):
    """Print detailed graph information."""
    nodes = data['nodes']
    edges = data['edges']
    
    print("\n" + "="*60)
    print("GRAPH INFORMATION")
    print("="*60)
    print(f"Contract: {data.get('contract_path', 'unknown')}")
    print(f"Vulnerability Type: {data.get('vulnerability_type', 'unknown')}")
    print(f"Template: {data.get('template_name', 'N/A')}")
    print(f"Injection Mode: {data.get('injection_mode', 'N/A')}")
    print(f"\nTotal Nodes: {len(nodes)}")
    print(f"Total Edges: {len(edges)}")
    
    # Vulnerable nodes
    vuln_nodes = [n for n in nodes if n.get('is_vulnerable', False)]
    print(f"Vulnerable Nodes: {len(vuln_nodes)}")
    
    if vuln_nodes:
        print("\nVulnerable Node Details:")
        for i, node in enumerate(vuln_nodes[:10]):  # Show first 10
            print(f"  [{i}] Type: {node['node_type']:30s} | "
                  f"Bytes: {node['start_byte']}-{node['end_byte']} | "
                  f"Component: {node.get('component', 'N/A')}")
        if len(vuln_nodes) > 10:
            print(f"  ... and {len(vuln_nodes) - 10} more")
    
    # Node type distribution
    node_types = Counter(n['node_type'] for n in nodes)
    print(f"\nNode Type Distribution (top 10):")
    for node_type, count in node_types.most_common(10):
        print(f"  {node_type:30s}: {count}")


def main():
    parser = argparse.ArgumentParser(description='Visualize AST graphs')
    parser.add_argument('--data', type=str, required=True, help='Dataset directory')
    parser.add_argument('--graph-dir', type=str, default='ast_graphs', help='Graph subdirectory')
    parser.add_argument('--index', type=int, default=None, help='Graph index to visualize')
    parser.add_argument('--vuln', type=str, default=None, help='Vulnerability type to filter')
    parser.add_argument('--output', type=str, default=None, help='Output image path')
    parser.add_argument('--max-nodes', type=int, default=200, help='Max nodes to visualize')
    
    args = parser.parse_args()
    
    graph_dir = Path(args.data) / args.graph_dir
    files = sorted(graph_dir.glob('*.json'))
    
    print(f"Found {len(files)} graph files")
    
    # Filter by vulnerability type if specified
    if args.vuln:
        filtered = []
        for f in files:
            data = load_graph_json(f)
            if data.get('vulnerability_type') == args.vuln:
                filtered.append(f)
        files = filtered
        print(f"Filtered to {len(files)} graphs with vuln_type={args.vuln}")
    
    if not files:
        print("No graphs found!")
        return
    
    # Select graph
    if args.index is not None:
        idx = args.index
    else:
        idx = random.randint(0, len(files) - 1)
    
    if idx >= len(files):
        print(f"Index {idx} out of range (max: {len(files)-1})")
        return
    
    graph_path = files[idx]
    print(f"\nLoading: {graph_path.name}")
    
    data = load_graph_json(graph_path)
    
    # Print info
    print_graph_info(data)
    
    # Visualize
    output_path = args.output or f"graph_viz_{data.get('vulnerability_type', 'unknown')}_{idx}.png"
    visualize_ast_graph(data, output_path, max_nodes=args.max_nodes)


if __name__ == '__main__':
    main()
