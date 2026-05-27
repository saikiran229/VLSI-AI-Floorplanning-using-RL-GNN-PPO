import matplotlib.pyplot as plt
import matplotlib.patches as patches
from stable_baselines3 import PPO
from vlsi_env import VLSIPlacementEnv
from torch_geometric.data import Data
import torch
import os

torch.serialization.add_safe_globals([Data])

def visualize_results():
    print("Step 1: Loading Model and VLSI Data...")
    
    # 1. Load the trained model and graph data
    if not os.path.exists('vlsi_master_model.zip') or not os.path.exists('processed_graph.pt'):
        print("Error: Missing model or graph files. Train the agent first!")
        return

    model = PPO.load("vlsi_master_model.zip")
    graph = torch.load('processed_graph.pt', weights_only=False)
    embeddings = torch.load('node_embeddings.pt', weights_only=False)

    # 2. Setup Environment and Run Placement
    env = VLSIPlacementEnv(
        node_embeddings=embeddings, 
        node_dims=graph.raw_dims, 
        edge_index=graph.edge_index,
        canvas_size=1000
    )

    print("Running AI Inference (Placing nodes based on learned policy)...")
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(action)

    
    print("Generating Floorplan Visualization...")
    fig, ax = plt.subplots(figsize=(14, 14))
    history = env.history
    placed_ids = [n['idx'] for n in history]

    
    print("Drawing connection nets...")
    for i in range(graph.edge_index.shape[1]):
        u, v = graph.edge_index[0, i].item(), graph.edge_index[1, i].item()
        
       
        if u in placed_ids and v in placed_ids:
            u_data = next(n for n in history if n['idx'] == u)
            v_data = next(n for n in history if n['idx'] == v)
            
            ax.plot(
                [u_data['x'], v_data['x']], 
                [u_data['y'], v_data['y']], 
                color='green', alpha=0.15, linewidth=0.6, zorder=1
            )

    
    print("Drawing physical blocks...")
    for node in history:
        
        area = node['w'] * node['h']
        is_macro = area > 500 
        
        face_color = 'salmon' if is_macro else 'skyblue'
        edge_color = 'darkred' if is_macro else 'navy'
        
        
        rect = patches.Rectangle(
            (node['x'] - node['w']/2, node['y'] - node['h']/2), 
            node['w'], node['h'], 
            linewidth=1, edgecolor=edge_color, facecolor=face_color, alpha=0.85, zorder=2
        )
        ax.add_patch(rect)
        
        
        label = f"M{node['idx']}" if is_macro else f"C{node['idx']}"
        ax.text(node['x'], node['y'], label, fontsize=7, ha='center', va='center', weight='bold', zorder=3)

    
    plt.xlim(0, 1000)
    plt.ylim(0, 1000)
    plt.title("Advanced AI VLSI Floorplan (adaptec1 subset)\nRed = Macros | Blue = Standard Cells | Green = Netlist Connections", fontsize=15)
    plt.xlabel("Microns (X)", fontsize=12)
    plt.ylabel("Microns (Y)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    output_file = "advanced_layout.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✅ visualize_placement.py: Success!")
    print(f"Results saved to '{output_file}'. Use 'xdg-open {output_file}' to view it.")

if __name__ == "__main__":
    visualize_results()
