from __future__ import division
from __future__ import print_function
import argparse
import time
import numpy as np
import scipy.sparse as sp
import torch
from torch import optim
import os.path as osp
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.utils import to_dense_adj, train_test_split_edges
from gae.model import GCNModelVAE
from gae.optimizer import loss_function
from gae.utils import load_data, mask_test_edges, preprocess_graph, get_roc_score
torch.manual_seed(0)
np.random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda', help='torch device.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=150, help='Number of epochs to train.')
parser.add_argument('--hidden1', type=int, default=32, help='Number of units in hidden layer 1.')
parser.add_argument('--hidden2', type=int, default=16, help='Number of units in hidden layer 2.')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--dropout', type=float, default=0., help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset-str', type=str, default='cora', help='type of dataset.')

args = parser.parse_args()


def gae(args):

    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Planetoid')
    transform = T.Compose([
        T.NormalizeFeatures(),
        T.ToDevice(args.device),
    ])
    dataset = Planetoid(path, args.dataset_str, transform=transform)
    dataset = dataset[0]
    dataset.train_mask = dataset.test_mask = dataset.val_mask = None
    dataset = train_test_split_edges(dataset)
    train_adj = to_dense_adj(dataset.train_pos_edge_index).squeeze(dim=0)
    val_adj = to_dense_adj(dataset.val_pos_edge_index, max_num_nodes=dataset.num_nodes).squeeze(dim=0)

    train_adj_norm = preprocess_graph(train_adj.cpu(), device=args.device)
    val_adj_norm = preprocess_graph(val_adj.cpu(), device=args.device)

    print("Using {} dataset".format(args.dataset_str))
    # n_nodes, feat_dim = train_data.num_nodes, train_data.num_features
    n_nodes, feat_dim = dataset.num_nodes, dataset.num_features
    # Store original adjacency matrix (without diagonal entries) for later
    adj_orig = train_adj.cpu().detach().numpy()
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    pos_weight = torch.Tensor([float(train_adj.shape[0] * train_adj.shape[0] - train_adj.sum()) / train_adj.sum()])
    norm = train_adj.shape[0] * train_adj.shape[0] / float((train_adj.shape[0] * train_adj.shape[0] - train_adj.sum()) * 2)

    model = GCNModelVAE(feat_dim, args.hidden1, args.hidden2, args.dropout)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if args.device == 'cuda':
        model = model.cuda()
        train_adj_norm = train_adj_norm.cuda()
        pos_weight = pos_weight.cuda()

    hidden_emb = None
    loss_trace = []
    for epoch in range(args.epochs):
        t = time.time()
        model.train()
        optimizer.zero_grad()
        # recovered, mu, logvar = model(train_data.x, train_adj_norm)
        recovered, mu, logvar = model(dataset.x, train_adj_norm)
        loss = loss_function(preds=recovered, pos_labels=train_adj,
                             mu=mu, logvar=logvar, n_nodes=n_nodes,
                             norm=norm, pos_weight=pos_weight, neg_labels=dataset.train_neg_adj_mask)
        loss.backward()
        cur_loss = loss.item()
        loss_trace.append(cur_loss)
        optimizer.step()
        hidden_emb = mu.data.cpu().detach().numpy()
        if (epoch+1)%10 == 0:
            model.eval()
            recovered, mu, logvar = model(dataset.x, val_adj_norm)
            val_loss = loss_function(preds=recovered, pos_labels=val_adj,
                                 mu=mu, logvar=logvar, n_nodes=n_nodes,
                                 norm=norm, pos_weight=pos_weight)

            roc_curr, ap_curr = get_roc_score(
                hidden_emb, adj_orig, dataset.val_pos_edge_index.t(), dataset.val_neg_edge_index.t()
            )
            print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(cur_loss),
                  "val_loss=", "{:.5f}".format(val_loss),
                  "val_ap=", "{:.5f}".format(ap_curr),
                  "time=", "{:.5f}".format(time.time() - t)
                  )

    print("Optimization Finished!")
    roc_score, ap_score = get_roc_score(
        hidden_emb, adj_orig, dataset.test_pos_edge_index.t(), dataset.test_neg_edge_index.t()
    )
    print('Test ROC score: ' + str(roc_score))
    print('Test AP score: ' + str(ap_score))
    return model


gae(args)