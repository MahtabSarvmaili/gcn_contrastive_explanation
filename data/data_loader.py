import argparse
import sys

sys.path.append('../../..')
import torch
import numpy as np
import pickle
import os.path as osp
import scipy.sparse as sp

from torch_geometric.utils import to_dense_adj, train_test_split_edges
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from gae.utils import preprocess_graph
torch.manual_seed(0)
np.random.seed(0)


def __load__planetoid__(dataset_str, transformer):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '../..', 'data', 'Planetoid')
    dataset = Planetoid(path, dataset_str, transform=transformer)[0]
    return dataset


def load_data(args):
    transform = T.Compose([
        T.NormalizeFeatures(),
        T.ToDevice(args.device),
        T.RandomNodeSplit(num_val=0.1, num_test=0.2),
    ])
    dataset = __load__planetoid__(args.dataset_str, transform)
    train_mask = dataset.train_mask
    val_mask = dataset.val_mask
    test_mask = dataset.test_mask
    features = dataset.x
    labels = dataset.y

    adj = to_dense_adj(dataset.edge_index).squeeze(dim=0)
    adj_norm = preprocess_graph(adj.cpu(), args.device)

    n_nodes, feat_dim, num_classes = dataset.num_nodes, dataset.num_features, len(dataset.y.unique())
    # Store original adjacency matrix (without diagonal entries) for later
    adj_orig = adj.cpu().detach().numpy()
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    pos_weight = torch.Tensor(
        [float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()],

    )
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
    idx_test = np.arange(0, n_nodes)[test_mask.cpu()]

    return {
        'dataset':dataset,
        'train_mask':train_mask,
        'val_mask':val_mask,
        'test_mask':test_mask,
        'idx_test':idx_test,
        'features':features,
        'labels':labels,
        'adj':adj,
        'adj_norm':adj_norm,
        'adj_orig':adj_orig,
        'pos_weight':pos_weight,
        'norm':norm,
        'n_nodes':n_nodes,
        'feat_dim':feat_dim,
        'num_classes':num_classes
    }


def load_data_ae(args):
    transform = T.Compose([
        T.NormalizeFeatures(),
        T.ToDevice(args.device),
    ])
    dataset = __load__planetoid__(args.dataset_str, transform)
    dataset.train_mask = dataset.test_mask = dataset.val_mask = None
    dataset = train_test_split_edges(dataset)
    train_adj = to_dense_adj(dataset.train_pos_edge_index).squeeze(dim=0)
    val_adj = to_dense_adj(dataset.val_pos_edge_index, max_num_nodes=dataset.num_nodes).squeeze(dim=0)

    train_adj_norm = preprocess_graph(train_adj.cpu(), device=args.device)
    val_adj_norm = preprocess_graph(val_adj.cpu(), device=args.device)

    print("Using {} dataset".format(args.dataset_str))
    # Store original adjacency matrix (without diagonal entries) for later
    adj_orig = train_adj.cpu().detach().numpy()
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    pos_weight = torch.Tensor([float(train_adj.shape[0] * train_adj.shape[0] - train_adj.sum()) / train_adj.sum()])
    norm = train_adj.shape[0] * train_adj.shape[0] / float((train_adj.shape[0] * train_adj.shape[0] - train_adj.sum()) * 2)
    return {
        'dataset':dataset,
        'train_adj':train_adj,
        'train_adj_norm':train_adj_norm,
        'val_adj':val_adj_norm,
        'val_adj_norm':val_adj_norm,
        'features':dataset.x,
        'labels':dataset.y,
        'adj_orig':adj_orig,
        'pos_weight':pos_weight,
        'norm':norm,
        'n_nodes':dataset.num_nodes,
        'feat_dim':dataset.num_features,
        'num_classes':len(dataset.y.unique())
    }