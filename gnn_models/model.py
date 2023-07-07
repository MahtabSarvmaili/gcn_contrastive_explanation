import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, GCNConv
from layers import GraphConvolution
from utils import accuracy
from sklearn.utils.class_weight import compute_class_weight
# torch.manual_seed(0)
# np.random.seed(0)


class GCN_dep(nn.Module):
    def __init__(self, nfeat, nhid, nout, nclasses, dropout):
        super(GCN_dep, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.gc3 = GraphConvolution(nhid, nhid)
        self.lin = nn.Linear(nout, nclasses)
        self.dropout = dropout

    def forward(self, x, adj, logit=True):
        x = F.relu(self.gc1(x, adj))
        x = F.relu(self.gc2(x, adj))
        x = self.gc3(x, adj)
        x = global_mean_pool(x, None)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        return x


class GCNGraph(nn.Module):
    def __init__(self, n_features, n_hidden, n_classes):
        super(GCNGraph, self).__init__()
        self.conv1 = GCNConv(n_features, n_hidden)
        self.conv2 = GCNConv(n_hidden, n_hidden)
        self.conv3 = GCNConv(n_hidden, n_hidden)
        self.lin = nn.Linear(n_hidden, n_classes)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        return x


class GCNNode(nn.Module):
    def __init__(self, n_features, n_hidden, n_classes):
        super(GCNNode, self).__init__()
        self.conv1 = GCNConv(n_features, n_hidden)
        self.conv2 = GCNConv(n_hidden, n_hidden)
        self.conv3 = GCNConv(n_hidden, n_hidden)
        self.lin = nn.Linear(n_hidden, n_classes)

    def forward(self, x, edge_index):
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        return x


def train_graph_classifier(model, criterion, optimizer, train_loader):
    model.train()

    for data in train_loader:  # Iterate in batches over the training dataset.
        out = model(data.x.to(torch.float), data.edge_index, data.batch)  # Perform a single forward pass.
        loss = criterion(out, data.y.to(torch.int64).reshape(-1))  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.


def test_graph_classifier(model, loader):
    model.eval()
    correct = 0
    for data in loader:  # Iterate in batches over the training/test dataset.
        out = model(data.x.to(torch.float), data.edge_index, data.batch)  # Perform a single forward pass.
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct += int((pred == data.y.to(torch.int64).reshape(-1)).sum())  # Check against ground-truth labels.
    return correct / len(loader.dataset)  # Derive ratio of correct predictions.


def train_node_classifier(model, graph, optimizer, criterion, n_epochs=200):
    for epoch in range(1, n_epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(graph.x, graph.edge_index)
        loss = criterion(out[graph.train_mask], graph.y[graph.train_mask])
        loss.backward()
        optimizer.step()

        pred = out.argmax(dim=1)

        if epoch % 10 == 0:
            train_acc = test_node_classifier(model, graph, graph.val_mask)
            val_acc = test_node_classifier(model, graph, graph.val_mask)
            print(f'Epoch: {epoch:03d}, Train Loss: {loss:.3f}, Train Acc: {train_acc:.3f}, Val Acc: {val_acc:.3f}')

    return model


def test_node_classifier(model, graph, mask):
    model.eval()
    pred = model(graph.x, graph.edge_index).argmax(dim=1)
    correct = (pred[mask] == graph.y[mask]).sum()
    acc = int(correct) / int(mask.sum())
    return acc
