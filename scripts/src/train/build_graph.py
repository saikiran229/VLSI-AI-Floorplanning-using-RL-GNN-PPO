import torch
import os
from torch_geometric.data import Data
from torch_geometric.utils import degree


torch.serialization.add_safe_globals([Data])

def build_vlsi_graph(data_dir):
    nodes_file = os.path.join(data_dir, "adaptec1.nodes")
    nets_file = os.path.join(data_dir, "adaptec1.nets")
    
    node_map = {}
    node_features = [] # To store [Width, Height]
    
    print("Step 1: Parsing Nodes and Dimensions...")
    with open(nodes_file, 'r') as f:
        for line in f:
            parts = line.split()
           
            if len(parts) >= 3 and parts[0].startswith('o'):
                node_name = parts[0]
                width = float(parts[1])
                height = float(parts[2])
                
                node_map[node_name] = len(node_features)
                node_features.append([width, height])
    
    raw_dims = torch.tensor(node_features, dtype=torch.float)
    num_nodes = len(node_features)
    print(f"Successfully parsed {num_nodes} nodes.")

    print("Step 2: Parsing Nets and Connectivity...")
    edge_list = []
    with open(nets_file, 'r') as f:
        current_net = []
        for line in f:
            if 'NetDegree' in line:
                
                if len(current_net) > 1:
                    for i in range(len(current_net)):
                        for j in range(i + 1, len(current_net)):
                            u, v = current_net[i], current_net[j]
                            edge_list.append([u, v])
                            edge_list.append([v, u])
                current_net = []
            else:
                parts = line.split()
                if parts and parts[0] in node_map:
                    current_net.append(node_map[parts[0]])

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    print(f"Successfully parsed {edge_index.shape[1]} edges.")

    print("Step 3: Engineering Degree Features...")
    
    d = degree(edge_index[1], num_nodes).view(-1, 1)
    
    # X features for GNN: [Normalized Width, Normalized Height, Normalized Degree]
    x_raw = torch.cat([raw_dims, d], dim=-1)
    x_norm = (x_raw - x_raw.min(0)[0]) / (x_raw.max(0)[0] - x_raw.min(0)[0])
    
    
    return Data(x=x_norm, edge_index=edge_index, raw_dims=raw_dims)

if __name__ == "__main__":
    
    data_path = os.path.expanduser("~/vlsi_ai_project/data/adaptec1")
    
    if not os.path.exists(data_path):
        print(f"❌ Error: Data directory not found at {data_path}")
    else:
        vlsi_graph = build_vlsi_graph(data_path)
        
        torch.save(vlsi_graph, 'processed_graph.pt')
        print("\nbuild_graph.py: Success!")
        print(f"Saved 'processed_graph.pt' with {vlsi_graph.num_nodes} nodes.")
