import argparse
import sys

sys.path.append('../../..')
import torch
import numpy as np
import os.path as osp
import scipy.sparse as sp

from torch_geometric.utils import to_dense_adj, train_test_split_edges, dense_to_sparse
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.data import Data
from gae.utils import preprocess_graph, mask_test_edges
from utils import normalize, normalize_adj, sparse_mx_to_torch_sparse_tensor
from data.gengraph import gen_syn1, preprocess_input_graph
import networkx as nx
torch.manual_seed(0)
np.random.seed(0)


def __load__planetoid__(dataset_str, transformer):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '../..', 'data', 'Planetoid')
    dataset = Planetoid(path, dataset_str, transform=transformer)[0]
    return dataset


def __prepare_edge_class_dataset__(dataset, device):
    adj = nx.adjacency_matrix(nx.from_edgelist(dataset.edge_index.t().cpu().numpy()))
    # Store original adjacency matrix (without diagonal entries) for later
    adj_orig = adj
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    adj_orig.eliminate_zeros()

    adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false, edges = mask_test_edges(adj)
    adj = adj_train
    # Some preprocessing
    adj_norm = preprocess_graph(adj, device)
    adj_label = adj_train + sp.eye(adj_train.shape[0])
    # adj_label = sparse_to_tuple(adj_label)
    adj_label = torch.FloatTensor(adj_label.toarray())
    adj_train = torch.FloatTensor(adj_train.todense())
    pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
    pos_weight = torch.tensor(pos_weight, dtype=torch.float32)
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
    if device=='cuda':
        adj_train = adj_train.cuda()
        adj_label = adj_label.cuda()
        pos_weight = pos_weight.cuda()

    return {
        'train_adj':adj_train,
        'adj_norm':adj_norm,
        'train_edges':train_edges,
        'val_edges':val_edges,
        'val_neg_edge':val_edges_false,
        'test_edge':test_edges,
        'test_neg_edge':test_edges_false,
        'features':dataset.x,
        'labels':adj_label,
        'adj_orig':adj_orig,
        'edge_index':edges,
        'edge_index_':dataset.edge_index,
        'pos_weight':pos_weight,
        'norm':norm,
        'n_nodes':dataset.num_nodes,
        'feat_dim':dataset.num_features,
        'num_classes':len(dataset.y.unique())
    }


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
        'train_mask':train_mask,
        'val_mask':val_mask,
        'test_mask':test_mask,
        'idx_test':idx_test,
        'features':features,
        'labels':labels,
        'adj':adj,
        'adj_norm':adj_norm,
        'adj_orig':adj_orig,
        'edge_index':dataset.edge_index,
        'pos_weight':pos_weight,
        'norm':norm,
        'n_nodes':n_nodes,
        'feat_dim':feat_dim,
        'num_classes':num_classes
    }


def load_data_AE(args):
    transform = T.Compose([
        # T.NormalizeFeatures(),
        T.ToDevice(args.device)
    ])
    dataset = __load__planetoid__(args.dataset_str, transform)
    dataset.train_mask = dataset.test_mask = dataset.val_mask = None
    return __prepare_edge_class_dataset__(dataset, args.device)


def load_synthetic(gen_syn_func, device='cuda'):
    G, role_id, name = gen_syn_func()
    data = preprocess_input_graph(G, role_id, name)
    org_adj = torch.tensor(data['org_adj'])
    adj = torch.tensor(data['adj'], dtype=torch.float32)

    edge_index = dense_to_sparse(adj)[0]
    features = normalize(data['feat'])
    features = torch.FloatTensor(np.array(features))
    labels = torch.LongTensor(data['labels'])
    dt = Data(x=features,edge_index=edge_index,y=labels)
    transform = T.Compose([
        T.NormalizeFeatures(),
        T.ToDevice(device),
        T.RandomNodeSplit(num_val=0.1, num_test=0.2),
    ])
    dataset = transform(dt)

    if device == 'cuda':
        adj = adj.cuda()
        org_adj = org_adj.cuda()

    return {
        'train_mask':dt.train_mask,
        'val_mask':dt.val_mask,
        'test_mask':dt.test_mask,
        'idx_test':np.arange(dataset.num_nodes)[dt.test_mask],
        'features':dataset.x,
        'labels':dataset.y,
        'adj':org_adj,
        'adj_norm':adj,
        'adj_orig':np.matrix(data['adj']),
        'edge_index':dataset.edge_index,
        'n_nodes':dataset.num_nodes,
        'feat_dim':dataset.num_features,
        'num_classes':len(dataset.y.unique())
    }


def load_synthetic_AE(gen_syn_func, device='cuda'):
    G, role_id, name = gen_syn_func()
    data = preprocess_input_graph(G, role_id, name)
    adj = torch.tensor(data['adj'], dtype=torch.float32)
    edge_index = dense_to_sparse(adj)[0]
    features = data['feat']
    features = torch.FloatTensor(np.array(features))
    labels = torch.LongTensor(data['labels'])
    dt = Data(x=features, edge_index=edge_index, y=labels)
    transform = T.Compose([
        T.ToDevice(device),
        T.NormalizeFeatures()
    ])
    dataset = transform(dt)
    return __prepare_edge_class_dataset__(dataset, device)