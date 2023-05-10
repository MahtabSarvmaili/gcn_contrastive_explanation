import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from layers import GraphConvolution
from utils import accuracy
from sklearn.utils.class_weight import compute_class_weight
# torch.manual_seed(0)
# np.random.seed(0)


class GCN(nn.Module):
    def __init__(self, nfeat, nhid1, nhid2, nout, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid1)
        self.gc2 = GraphConvolution(nhid1, nhid2)
        self.lin = nn.Linear(nhid2, nout)
        self.loss_func = nn.BCEWithLogitsLoss()
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        # x = F.dropout(x, self.dropout, training=False)
        x = F.relu(self.gc2(x, adj))
        # x = F.dropout(x, self.dropout, training=False)
        return x

    def link_pred(self, x_i, x_j, sigmoid=True):
        x = x_i * x_j
        x = self.lin(x)
        if sigmoid:
            x = torch.sigmoid(x)
        return x

    def loss(self, node_emb, edges, labels):
        preds = self.link_pred(node_emb[edges[0]], node_emb[edges[1]], sigmoid=False)
        loss = self.loss_func(preds, labels)
        return loss


class GraphSparseConv(nn.Module):
    def __init__(self, num_features, num_hidden1, num_hidden2, nout=1, device='cuda'):
        super(GraphSparseConv, self).__init__()
        self.gc1 = GCNConv(num_features, num_hidden1)
        self.gc2 = GCNConv(num_hidden1, num_hidden2)
        self.lin = nn.Linear(num_hidden2, nout)
        self.loss_func = nn.BCEWithLogitsLoss()
        self.device = device

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        # x = F.dropout(x, self.dropout, training=False)
        x = F.relu(self.gc2(x, adj))
        # x = F.dropout(x, self.dropout, training=False)
        return x

    def link_pred(self, x_i, x_j, sigmoid=True):
        x = x_i * x_j
        x = self.lin(x)
        if sigmoid:
            x = torch.sigmoid(x)
        return x

    def loss(self, node_emb, edges, labels):
        preds = self.link_pred(node_emb[edges[0]], node_emb[edges[1]], sigmoid=False)
        loss = self.loss_func(preds, labels)
        return loss


class simple(nn.Module):
    def __init__(self, nfeat, nhid):
        super(simple, self).__init__()
        self.gc = GraphConvolution(nfeat, nhid)
        self.lin = nn.Linear(nhid, 1)

    def forward(self, x, adj):
        x = self.gc(x, adj)
        x = F.relu(x)
        return x

    def link_pred(self, x_i, x_j):
        x = x_i * x_j
        x = self.lin(x)
        x = torch.sigmoid(x)
        return x

    def loss(self, node_emb, pos_edge, neg_edge):
        pos_pred = self.link_pred(node_emb[pos_edge[0]], node_emb[pos_edge[1]])
        neg_pred = self.link_pred(node_emb[neg_edge[0]], node_emb[neg_edge[1]])
        loss = -torch.log(pos_pred + 1e-15).mean() - torch.log(1 - neg_pred + 1e-15).mean()
        return loss

