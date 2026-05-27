import torch
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.data import Data


torch.serialization.add_safe_globals([Data])

def generate_graph_image():
    print("Loading VLSI Graph Data...")
    graph = torch.load('processed_graph.pt', weights_only=False)

    #Isolate the 50-node subset
    subset_size = 50
    

    mask = (graph.edge_index[0] < subset_size) & (graph.edge_index[1] < subset_size)
    subset_edges = graph.edge_index[:, mask]

    print(f"🔬 Filtered down to {subset_size} nodes and {subset_edges.shape[1] // 2} unique connections.")


    G = nx.Graph()
    G.add_nodes_from(range(subset_size))
    edges_list = subset_edges.t().tolist()
    G.add_edges_from(edges_list)

    # 4. Set up the Matplotlib Canvas
    plt.figure(figsize=(10, 8), facecolor='white')
    

    pos = nx.spring_layout(G, k=0.5, seed=42) 
    

    nx.draw_networkx_nodes(G, pos, node_size=300, node_color='#87CEEB', edgecolors='black', linewidths=1.5)
    nx.draw_networkx_edges(G, pos, width=1.5, alpha=0.6, edge_color='#2E8B57')
    nx.draw_networkx_labels(G, pos, font_size=9, font_family='sans-serif', font_weight='bold')

    plt.title("Phase 2: VLSI Hypergraph Topology (50 Node Subset)", fontsize=16, fontweight='bold', pad=20)
    plt.axis('off') 
    

    output_file = "graph_topology.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Success! Graph topology saved to '{output_file}'")

if __name__ == "__main__":
    generate_graph_image()
