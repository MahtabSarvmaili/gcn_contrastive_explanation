import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from layers import GraphConvolution
from utils import accuracy
from sklearn.utils.class_weight import compute_class_weight
# torch.manual_seed(0)
# np.random.seed(0)


class LinkPredictor(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LinkPredictor, self).__init__()
        self.lin = nn.Linear(in_channels, out_channels)

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j):
        # x_i and x_j are both of shape (E, D)
        x = x_i * x_j
        x = self.lin(x)
        x = F.relu(x)
        return torch.sigmoid(x)