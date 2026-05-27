import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from vlsi_env import VLSIPlacementEnv
from torch_geometric.data import Data
import os
import argparse
import numpy as np

torch.serialization.add_safe_globals([Data])

class VLSIMetricsCallback(BaseCallback):
    """Custom callback to log real HPWL and Congestion improvements."""
    def __init__(self, verbose=0):
        super(VLSIMetricsCallback, self).__init__(verbose)
        self.best_hpwl = float('inf')

    def _on_step(self) -> bool:
        # Check if an episode just finished
        if self.locals['dones'][0]:
            # Safely pull the metrics from the info dictionary
            info = self.locals['infos'][0]
            
            if 'final_hpwl' in info:
                total_hpwl = abs(info['final_hpwl']) 
                total_overlap = info['final_collisions']
                
                if total_hpwl < self.best_hpwl and total_overlap == 0:
                    self.best_hpwl = total_hpwl
                
                # Log to TensorBoard
                self.logger.record("vlsi/episode_hpwl", total_hpwl)
                self.logger.record("vlsi/episode_collisions", total_overlap)
                self.logger.record("vlsi/best_legal_hpwl", self.best_hpwl if self.best_hpwl != float('inf') else 0)
        return True

def train_master_placer(is_fast_test=True):
    print("Loading GNN Embeddings and Graph Data...")
    graph = torch.load('processed_graph.pt', weights_only=False)
    embeddings = torch.load('node_embeddings.pt', weights_only=False)

    env = VLSIPlacementEnv(
        node_embeddings=embeddings, node_dims=graph.raw_dims, 
        edge_index=graph.edge_index, canvas_size=1000
    )
    env = DummyVecEnv([lambda: env])

    policy_kwargs = dict(activation_fn=torch.nn.ReLU, net_arch=dict(pi=[256, 256, 256], qf=[256, 256, 256]))

    # tensorboard_log directory here
    model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1, 
                learning_rate=3e-4, n_steps=2048, batch_size=64, n_epochs=10, gamma=0.99,
                tensorboard_log="./vlsi_tensorboard/")

    timesteps = 50000 if is_fast_test else 200000
    
    
    metrics_callback = VLSIMetricsCallback()
    
    print(f"\nStarting Training ({timesteps} steps)...")
    model.learn(total_timesteps=timesteps, callback=metrics_callback)

    model.save("vlsi_master_model")
    print("\nSuccess! Model saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--full', action='store_true')
    args = parser.parse_args()
    train_master_placer(is_fast_test=not args.full)
