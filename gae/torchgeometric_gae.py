import os.path as osp

import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from torch_geometric.utils import negative_sampling
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv
from torch_geometric.utils import train_test_split_edges


class GAE(torch.nn.Module):
    def __init__(self, num_features, num_hidden1, num_hidden2, device='cuda'):
        super(GAE, self).__init__()
        self.conv1 = GCNConv(num_features, num_hidden1)
        self.conv2 = GCNConv(num_hidden1, num_hidden2)
        self.device = device

    def encode(self, x, pos_edge_index):
        x = self.conv1(x, pos_edge_index) # convolution 1
        x = x.relu()
        return self.conv2(x, pos_edge_index) # convolution 2

    def decode(self, z, pos_edge_index, neg_edge_index): # only pos and neg edges
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1).to(self.device)# concatenate pos and neg edges
        logits = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)  # dot product 
        return logits

    def decode_all(self, z): 
        prob_adj = z @ z.t() # get adj NxN
        return (prob_adj > 0).nonzero(as_tuple=False).t() # get predicted edge_list 