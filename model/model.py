import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

class Transformer(nn.Module):
    def __init__(self, input_dim, num_heads, ff_hidden_dim, num_layers, dropout=0.1):
        super(Transformer, self).__init__()
        # Transformer Encoder Layer
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=input_dim,
                nhead=num_heads,
                dim_feedforward=ff_hidden_dim,
                dropout=dropout,
                activation="relu"
            ) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(input_dim)
        output_dim = 128
        self.output_layer = nn.Linear(input_dim, output_dim)
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, input_dim)
        Returns:
            Tensor of shape (batch_size, seq_len, input_dim)
        """
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)  
        x = self.output_layer(x)
        return x

class SynthesisPredictionModel(nn.Module):
    def __init__(self, input_dim: int=512, hidden_dim: int=1024, output_dim: int=27106):
        super(SynthesisPredictionModel, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layer = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, target_formula: np.ndarray) -> Tensor:
        x = F.relu(self.input_layer(target_formula))
        x = F.relu(self.hidden_layer(x))
        x = self.output_layer(x)
        return x