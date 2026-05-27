import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from torch_geometric.data import Data

torch.serialization.add_safe_globals([Data])

class VLSIPlacementEnv(gym.Env):
    def __init__(self, node_embeddings, node_dims, edge_index, canvas_size=1000):
        super(VLSIPlacementEnv, self).__init__()
        
        self.embeddings = node_embeddings
        self.node_dims = node_dims  
        self.edge_index = edge_index
        self.canvas_size = canvas_size
        self.num_to_place = 50 
        
        # We keep the Box space so the neural network architecture remains stable
        self.action_space = spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(170,), dtype=np.float32)
        
        self.current_node_idx = 0
        self.history = []

    def _get_obs(self):
        if self.current_node_idx < self.num_to_place:
            emb = self.embeddings[self.current_node_idx].numpy()
            w, h = self.node_dims[self.current_node_idx]
        else:
            emb = np.zeros(64, dtype=np.float32)
            w, h = torch.tensor(0.0), torch.tensor(0.0)

        norm_w = w.item() / self.canvas_size
        norm_h = h.item() / self.canvas_size

        grid = np.zeros((10, 10), dtype=np.float32)
        for node in self.history:
            grid_x = int(np.clip((node['x'] / self.canvas_size) * 10, 0, 9))
            grid_y = int(np.clip((node['y'] / self.canvas_size) * 10, 0, 9))
            grid[grid_x, grid_y] += 1.0 

        grid_flat = grid.flatten() 

        if len(self.history) > 0:
            last_x = self.history[-1]['x'] / self.canvas_size
            last_y = self.history[-1]['y'] / self.canvas_size
            cx = sum(n['x'] for n in self.history) / len(self.history) / self.canvas_size
            cy = sum(n['y'] for n in self.history) / len(self.history) / self.canvas_size
        else:
            last_x, last_y = 0.5, 0.5 
            cx, cy = 0.5, 0.5

        obs = np.concatenate([
            emb,
            np.array([norm_w, norm_h, last_x, last_y, cx, cy], dtype=np.float32),
            grid_flat
        ])
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_node_idx = 0
        self.history = []
        return self._get_obs(), {}

    def step(self, action):
        w, h = self.node_dims[self.current_node_idx]
        w_val, h_val = w.item(), h.item()
        

        
        # THE CHESSBOARD GRID (50x50 slots, each 20x20 microns)
        grid_slots = 50 
        slot_size = self.canvas_size / grid_slots 
        
        # 1. Snap AI's floating point guess to the exact grid slot (0 to 49)
        grid_x = int(np.clip(action[0] * grid_slots, 0, grid_slots - 1))
        grid_y = int(np.clip(action[1] * grid_slots, 0, grid_slots - 1))
        
        # 2. Place cell perfectly in the center of that grid slot
        x_center = (grid_x * slot_size) + (slot_size / 2.0)
        y_center = (grid_y * slot_size) + (slot_size / 2.0)
        
        # 3. Handle Macro Overflows (Clamp back to board if a massive block spills over)
        x_center = np.clip(x_center, w_val/2, self.canvas_size - w_val/2)
        y_center = np.clip(y_center, h_val/2, self.canvas_size - h_val/2)
            
        current_placement = {
            'idx': self.current_node_idx, 
            'x': x_center, 'y': y_center, 
            'w': w_val, 'h': h_val
        }
        self.history.append(current_placement)
        
        hpwl_reward = self._calculate_hpwl_reward(x_center, y_center)
        overlap_penalty = self._calculate_overlap_penalty(current_placement)
        boundary_penalty = self._calculate_boundary_penalty(current_placement)
        
        if overlap_penalty == 0 and boundary_penalty == 0:
            reward = 2.0 + (hpwl_reward * 0.1) 
        else:
            reward = -(overlap_penalty * 2.0) - (boundary_penalty * 2.0) + (hpwl_reward * 0.01)
        
        reward = float(np.clip(reward, -5.0, 5.0))
        
        self.current_node_idx += 1
        done = self.current_node_idx >= self.num_to_place
        
        info = {}
        if done:
            info['final_collisions'] = overlap_penalty 
            info['final_hpwl'] = hpwl_reward * self.canvas_size 
            
        return self._get_obs(), reward, done, False, info

    def _calculate_hpwl_reward(self, cur_x, cur_y):
        # NOTE: abs(dx) + abs(dy) IS already Manhattan distance! The math was right, now the visuals match.
        if len(self.history) < 2: return 0.0
        cur_node_id = self.history[-1]['idx']
        connected_mask = (self.edge_index[0] == cur_node_id)
        connected_node_ids = self.edge_index[1][connected_mask].tolist()
        
        hpwl_total = 0
        connections_found = 0
        
        for target_id in connected_node_ids:
            placed_neighbor = [n for n in self.history if n['idx'] == target_id]
            if placed_neighbor:
                target = placed_neighbor[0]
                dist = abs(cur_x - target['x']) + abs(cur_y - target['y'])
                hpwl_total += dist
                connections_found += 1
        
        if connections_found == 0: return 0.0
        avg_hpwl = hpwl_total / connections_found
        return -float(avg_hpwl / self.canvas_size)

    def _calculate_overlap_penalty(self, cur_node):
        if len(self.history) < 2: return 0.0
        collisions = 0
        for placed_node in self.history[:-1]:
            dx = min(cur_node['x'] + cur_node['w']/2, placed_node['x'] + placed_node['w']/2) - \
                 max(cur_node['x'] - cur_node['w']/2, placed_node['x'] - placed_node['w']/2)
            dy = min(cur_node['y'] + cur_node['h']/2, placed_node['y'] + placed_node['h']/2) - \
                 max(cur_node['y'] - cur_node['h']/2, placed_node['y'] - placed_node['h']/2)
            # Threshold to prevent mathematical floating point rounding false-positives
            if dx > 0.1 and dy > 0.1:
                collisions += 1.0 
        return collisions

    def _calculate_boundary_penalty(self, cur):
        penalty = 0.0
        if cur['x'] - cur['w']/2 < 0: penalty += 1.0
        if cur['x'] + cur['w']/2 > self.canvas_size: penalty += 1.0
        if cur['y'] - cur['h']/2 < 0: penalty += 1.0
        if cur['y'] + cur['h']/2 > self.canvas_size: penalty += 1.0
        return penalty
