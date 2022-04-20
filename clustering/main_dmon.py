from dmon import DMoN

import argparse
import sys
import torch
import numpy as np
import pickle
import sklearn
import os.path as osp
import scipy.sparse as sp
import clustering.metrics as metrics
from torch_geometric.utils import to_dense_adj
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from gae.utils import preprocess_graph
from model import GCN, train

from gae.GAE import gae
torch.manual_seed(0)
np.random.seed(0)


parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda', help='torch device.')
parser.add_argument('--bb-epochs', type=int, default=500, help='Number of epochs to train the ')
parser.add_argument('--cf-epochs', type=int, default=200, help='Number of epochs to train the ')
parser.add_argument('--inputdim', type=int, default=10, help='Input dimension')
parser.add_argument('--hidden', type=int, default=20, help='Number of units in hidden layer 1.')
parser.add_argument('--n-layers', type=int, default=3, help='Number of units in hidden layer 1.')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--cf-optimizer', type=str, default='SGD', help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset-str', type=str, default='cora', help='type of dataset.')
parser.add_argument('--beta', type=float, default=0.5, help='beta variable')
parser.add_argument('--include_ae', type=bool, default=False, help='Including AutoEncoder reconstruction loss')
parser.add_argument('--n-momentum', type=float, default=0.0, help='Nesterov momentum')
explainer_args = parser.parse_args()


def main(explainer_args):

    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Planetoid')
    transform = T.Compose([
        T.NormalizeFeatures(),
        T.ToDevice(explainer_args.device),
        T.RandomNodeSplit(num_val=0.1, num_test=0.2),
    ])
    dataset = Planetoid(path, explainer_args.dataset_str, transform=transform)[0]
    train_mask = dataset.train_mask
    val_mask = dataset.val_mask
    test_mask = dataset.test_mask
    features = dataset.x
    labels = dataset.y

    adj = to_dense_adj(dataset.edge_index).squeeze(dim=0)
    adj_norm = preprocess_graph(adj.cpu(), explainer_args.device)

    n_nodes, feat_dim, num_classes = dataset.num_nodes, dataset.num_features, len(dataset.y.unique())
    # Store original adjacency matrix (without diagonal entries) for later
    adj_orig = adj.cpu().detach().numpy()
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    pos_weight = torch.Tensor(
        [float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()],

    )

    model = GCN(
        nfeat=feat_dim,
        nhid=explainer_args.hidden,
        nout=explainer_args.hidden,
        nclasses=num_classes,
        dropout=explainer_args.dropout
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=explainer_args.lr, weight_decay=5e-4)

    if explainer_args.device=='cuda':
        model = model.cuda()


    train(
        model=model,
        features=features,
        train_adj=adj_norm,
        labels=labels,
        train_mask=train_mask,
        optimizer=optimizer,
        epoch=explainer_args.bb_epochs,
        val_mask=val_mask
    )

    model.eval()
    cluster_features = model.encode(features, adj_norm).detach()
    dmon = DMoN(cluster_features.shape[1], n_clusters=16)
    optimizer = torch.optim.Adam(dmon.parameters(), lr=explainer_args.lr, weight_decay=5e-4)
    for i in range(2000):
        dmon.train()
        optimizer.zero_grad()
        loss = dmon.loss(cluster_features, adj_norm.to_dense())
        loss.backward()
        optimizer.step()
        print(f'epoch {i}, loss: {loss}')

    dmon.eval()
    a, b = dmon.forward(cluster_features[1].reshape(1, -1))
    features_pooled, assignments = dmon.forward(cluster_features)
    assignments = assignments.cpu().detach().numpy()
    clusters = assignments.argmax(axis=1)
    # Prints some metrics used in the paper.
    print('Conductance:', metrics.conductance(adj_orig, clusters))
    print('Modularity:', metrics.modularity(adj_orig, clusters))
    print(
        'NMI:',
        sklearn.metrics.normalized_mutual_info_score(
            labels, clusters, average_method='arithmetic'))
    precision = metrics.pairwise_precision(labels, clusters)
    recall = metrics.pairwise_recall(labels, clusters)
    print('F1:', 2 * precision * recall / (precision + recall))

    print('yes')


if __name__ == '__main__':
    main(explainer_args)