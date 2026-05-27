import torch
from model_gnn import VLSI_GNN
import os
from torch_geometric.data import Data

torch.serialization.add_safe_globals([Data])

def run_perception():
    print("Loading BigBlue1 processed graph data...")
    #Load the BigBlue specific graph
    if not os.path.exists('bb_processed_graph.pt'):
        print("Error: 'bb_processed_graph.pt' not found.")
        return
    
    graph = torch.load('bb_processed_graph.pt', weights_only=False)

    print("Initializing 3-Layer GNN Encoder...")
    model = VLSI_GNN(input_dim=3, hidden_dim=64, output_dim=64)
    model.eval() 

    print("Generating embeddings for the BigBlue placement window...")
    subset_size = 5000
    subset_x = graph.x[:subset_size]
    mask = (graph.edge_index[0] < subset_size) & (graph.edge_index[1] < subset_size)
    subset_edge_index = graph.edge_index[:, mask]

    with torch.no_grad():
        embeddings = model(subset_x, subset_edge_index)

    # Save as a unique file so we don't destroy adaptec1 embeddings
    torch.save(embeddings, 'bb_node_embeddings.pt')
    
    print(f"\n run_perception_bb.py: Success! Saved 'bb_node_embeddings.pt'.")

if __name__ == "__main__":
    run_perception()
