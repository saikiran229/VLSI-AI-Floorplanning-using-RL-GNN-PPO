import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data


torch.serialization.add_safe_globals([Data])

class VLSI_GNN(torch.nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, output_dim=64):
        """
        3-Layer GCN Architecture
        input_dim: 3 (Normalized Width, Height, Degree)
        hidden_dim: 64 (Increased for better pattern recognition)
        output_dim: 64 (The final embedding size for the RL agent)
        """
        super(VLSI_GNN, self).__init__()
        
        # Layer 1: Initial feature extraction
        self.conv1 = GCNConv(input_dim, hidden_dim)
        
        # Layer 2: Intermediate spatial relationship learning
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        
        # Layer 3: Final embedding generation
        self.conv3 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        # First Layer + Activation + Dropout (to prevent overfitting)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.1, training=self.training)
        
        # Second Layer
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        
        # Third Layer (Final Embeddings)
        x = self.conv3(x, edge_index)
        
        return x

if __name__ == "__main__":
    # Quick check of the architecture
    model = VLSI_GNN()
    print(" model_gnn.py: 3-Layer Architecture Initialized.")
    print(model)
    
    # Check total parameters 
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Trainable Parameters: {total_params}")
