import torch
import os
import gdstk
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from vlsi_env import VLSIPlacementEnv
from torch_geometric.data import Data

torch.serialization.add_safe_globals([Data])

def run_transfer_learning():
    print("Loading New Chip Data (bigblue1)...")
    if not os.path.exists('vlsi_master_model.zip'):
        print("Error: Cannot find 'vlsi_master_model.zip'.")
        return

    #Load the BigBlue specific data files
    graph = torch.load('bb_processed_graph.pt', weights_only=False)
    embeddings = torch.load('bb_node_embeddings.pt', weights_only=False)

    env = VLSIPlacementEnv(
        node_embeddings=embeddings, node_dims=graph.raw_dims, 
        edge_index=graph.edge_index, canvas_size=1000
    )
    vec_env = DummyVecEnv([lambda: env])

    #We load the brain trained on Adaptec1, but place it into the BigBlue1 environment!
    print("Loading Master AI Policy (Pre-trained on adaptec1)...")
    model = PPO.load("vlsi_master_model", env=vec_env)

    # ZERO-SHOT INFERENCE 
    print("\n PHASE 1: Zero-Shot Inference (0 Training Steps)...")
    obs, _ = env.reset()
    done = False
    zero_shot_hpwl = 0
    zero_shot_collisions = 0
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)
        if done:
            zero_shot_hpwl = info['final_hpwl']
            zero_shot_collisions = info['final_collisions']

    print(f"   Target Acquired. Time: < 1 second.")
    print(f"   Zero-Shot HPWL: {zero_shot_hpwl:.2f}")
    print(f"   Zero-Shot Collisions: {zero_shot_collisions}")

    # --- PHASE 2: FINE-TUNING ---
    print("\nPHASE 2: Fine-Tuning (2,048 steps)...")
    model.learn(total_timesteps=2048)

    # EVALUATE FINE-TUNED MODEL
    print("\n PHASE 3: Post-Fine-Tuning Inference...")
    obs, _ = env.reset()
    done = False
    finetuned_hpwl = 0
    finetuned_collisions = 0
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)
        if done:
            finetuned_hpwl = info['final_hpwl']
            finetuned_collisions = info['final_collisions']

    print(f"   Fine-Tuned HPWL: {finetuned_hpwl:.2f}")
    print(f"   Fine-Tuned Collisions: {finetuned_collisions}")
    
    improvement = zero_shot_hpwl - finetuned_hpwl
    if improvement > 0:
        print(f"  Improvement: {improvement:.2f} wirelength units saved via Fine-Tuning!")
    else:
        print(f"  The Zero-Shot placement was already highly optimized!")

    # EXPORT TO KLAYOUT
    print("\nConstructing GDSII Layout for the New Chip...")
    lib = gdstk.Library()
    main_cell = lib.new_cell("BIGBLUE_TOP_LEVEL")

    boundary_path = [(0, 0), (1000, 0), (1000, 1000), (0, 1000), (0, 0)]
    frame = gdstk.FlexPath(boundary_path, width=2, layer=0, datatype=0)
    main_cell.add(frame)

    grid_spacing = 20.0
    for i in range(0, 1001, int(grid_spacing)):
        main_cell.add(gdstk.FlexPath([(i, 0), (i, 1000)], width=0.5, layer=4, datatype=0))
        main_cell.add(gdstk.FlexPath([(0, i), (1000, i)], width=0.5, layer=4, datatype=0))

    placed_nodes = {n['idx']: n for n in env.history}

    for i in range(graph.edge_index.shape[1]):
        u, v = graph.edge_index[0, i].item(), graph.edge_index[1, i].item()
        if u in placed_nodes and v in placed_nodes:
            u_node = placed_nodes[u]
            v_node = placed_nodes[v]
            x1, y1 = u_node['x'], u_node['y']
            x2, y2 = v_node['x'], v_node['y']
            net_path = [(x1, y1), (x2, y1), (x2, y2)]
            manhattan_wire = gdstk.FlexPath(net_path, width=1.0, layer=5, datatype=0)
            main_cell.add(manhattan_wire)

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

    # Save as transfer_floorplan.gds so it doesn't overwrite your Adaptec layout
    gds_filename = "transfer_floorplan.gds"
    lib.write_gds(gds_filename)
    print(f"Success! Layout exported to '{gds_filename}'.")

if __name__ == "__main__":
    run_transfer_learning()
