import gdstk
import torch
import os
from stable_baselines3 import PPO
from vlsi_env import VLSIPlacementEnv
from torch_geometric.data import Data

torch.serialization.add_safe_globals([Data])

def generate_gds():
    print("Loading AI Policy for GDSII Extraction...")
    model = PPO.load("vlsi_master_model")
    graph = torch.load('processed_graph.pt', weights_only=False)
    embeddings = torch.load('node_embeddings.pt', weights_only=False)

    env = VLSIPlacementEnv(
        node_embeddings=embeddings, node_dims=graph.raw_dims, 
        edge_index=graph.edge_index, canvas_size=1000
    )

    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, _ = env.step(action)

    print("Constructing GDSII Layout...")
    lib = gdstk.Library()
    main_cell = lib.new_cell("AI_TOP_LEVEL")

    boundary_path = [(0, 0), (1000, 0), (1000, 1000), (0, 1000), (0, 0)]
    frame = gdstk.FlexPath(boundary_path, width=2, layer=0, datatype=0)
    main_cell.add(frame)

    # DRAW THE CHESSBOARD GRID (Layer 4)
    grid_spacing = 20.0
    for i in range(0, 1001, int(grid_spacing)):
       
        main_cell.add(gdstk.FlexPath([(i, 0), (i, 1000)], width=0.5, layer=4, datatype=0))
       
        main_cell.add(gdstk.FlexPath([(0, i), (1000, i)], width=0.5, layer=4, datatype=0))

    placed_nodes = {n['idx']: n for n in env.history}

    # DRAW MANHATTAN ROUTING (Layer 5)
    for i in range(graph.edge_index.shape[1]):
        u, v = graph.edge_index[0, i].item(), graph.edge_index[1, i].item()
        if u in placed_nodes and v in placed_nodes:
            u_node = placed_nodes[u]
            v_node = placed_nodes[v]
            
            # Create an L-shape (Manhattan) route instead of a Euclidean route
            x1, y1 = u_node['x'], u_node['y']
            x2, y2 = v_node['x'], v_node['y']
            
            # Route X first, then Y
            net_path = [(x1, y1), (x2, y1), (x2, y2)]
            manhattan_wire = gdstk.FlexPath(net_path, width=1.0, layer=5, datatype=0)
            main_cell.add(manhattan_wire)

    # DRAW MACROS AND CELLS (Layers 1 and 2)
    for node in env.history:
        area = node['w'] * node['h']
        is_macro = area > 500
        
        layer = 1 if is_macro else 2
        
        x_min = node['x'] - (node['w'] / 2)
        y_min = node['y'] - (node['h'] / 2)
        x_max = node['x'] + (node['w'] / 2)
        y_max = node['y'] + (node['h'] / 2)
        
        rect = gdstk.rectangle((x_min, y_min), (x_max, y_max), layer=layer)
        main_cell.add(rect)
        
        label_text = f"M{node['idx']}" if is_macro else f"C{node['idx']}"
        label = gdstk.Label(label_text, (node['x'], node['y']), layer=3)
        main_cell.add(label)

    gds_filename = "ai_floorplan.gds"
    lib.write_gds(gds_filename)
    print(f"\nSuccess! Layout exported to '{gds_filename}'.")
    print("In KLayout, look at Layer 4 for the Grid and Layer 5 for Manhattan Routes!")

if __name__ == "__main__":
    generate_gds()
