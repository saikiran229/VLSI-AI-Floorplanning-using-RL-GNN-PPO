import torch
from build_graph import build_vlsi_graph
from model_gnn import VLSI_GNN
import os
from torch_geometric.data import Data


torch.serialization.add_safe_globals([Data])

def run_perception():
    # 1. Load the processed graph
    # We load from the saved file to ensure features are identical
    print(" Loading processed graph data...")
    if not os.path.exists('processed_graph.pt'):
        print("Error: 'processed_graph.pt' not found. Run build_graph.py first.")
        return
    
    graph = torch.load('processed_graph.pt', weights_only=False)

    
    print("Initializing 3-Layer GNN Encoder...")
    model = VLSI_GNN(input_dim=3, hidden_dim=64, output_dim=64)
    model.eval() # Set to evaluation mode

    
    # WE sample a window of nodes to keep memory usage low
    print("Generating embeddings for the placement window (5000 nodes)...")
    
    subset_size = 5000
    subset_x = graph.x[:subset_size]
    
    
    mask = (graph.edge_index[0] < subset_size) & (graph.edge_index[1] < subset_size)
    subset_edge_index = graph.edge_index[:, mask]

    with torch.no_grad():
        # Generate the 64-D Embeddings
        embeddings = model(subset_x, subset_edge_index)

    # 4. Save the Embeddings for the RL Agent
    torch.save(embeddings, 'node_embeddings.pt')
    
    print(f"\n run_perception.py: Success!")
    print(f"Generated {embeddings.shape[0]} embeddings.")
    print(f"Saved 'node_embeddings.pt' for Step 4 (The RL Agent).")

if __name__ == "__main__":
    run_perception()
