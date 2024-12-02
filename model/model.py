import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F



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