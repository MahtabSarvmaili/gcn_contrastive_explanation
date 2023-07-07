import copy
import sys

sys.path.append('../../..')
import torch
import numpy as np
import os.path as osp
import scipy.sparse as sp
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_adj, train_test_split_edges, dense_to_sparse
from sklearn.model_selection import StratifiedKFold
from torch_geometric.datasets import Planetoid, TUDataset, MoleculeNet
import torch_geometric.transforms as T
from torch_geometric.data import Data
from gae.utils import preprocess_graph, mask_test_edges
from utils import normalize, normalize_adj, sparse_mx_to_torch_sparse_tensor
from data.gengraph import gen_syn1, gen_syn2
import networkx as nx
torch.manual_seed(0)
np.random.seed(0)


def __k_fold__(x, y, folds, device='cuda'):
    skf = StratifiedKFold(folds, shuffle=True, random_state=12345)

    test_indices, train_indices = [], []
    for _, idx in skf.split(torch.zeros(len(x)), y.cpu()):
        test_indices.append(torch.from_numpy(idx).to(torch.long).to(device))

    val_indices = [test_indices[i - 1] for i in range(folds)]

    for i in range(folds):
        train_mask = torch.ones(len(x), dtype=torch.bool, device=device)
        train_mask[test_indices[i]] = 0
        train_mask[val_indices[i]] = 0
        train_indices.append(train_mask.nonzero(as_tuple=False).view(-1))

    return train_indices, test_indices, val_indices


def __load__data__(dataset_func, dataset_str, transformer):
    function = globals()[dataset_func]
    path = osp.join(osp.dirname(osp.realpath(__file__)), dataset_func).replace("\\", "/")
    dataset = function(path, dataset_str, transform=transformer)[0]
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
    adj_norm = normalize_adj(adj, device)
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
        'train_adj': adj_train,
        'adj_norm': adj_norm,
        'train_edges': train_edges,
        'val_edges': val_edges,
        'val_neg_edge': val_edges_false,
        'test_edge': test_edges,
        'test_neg_edge': test_edges_false,
        'features': dataset.x,
        'labels': adj_label,
        'adj_orig': adj_orig,
        'edge_index': edges,
        'edge_index_': dataset.edge_index,
        'pos_weight': pos_weight,
        'norm': norm,
        'n_nodes': dataset.num_nodes,
        'feat_dim': dataset.num_features,
        'num_classes': len(dataset.y.unique())
    }


def load_data(args):
    transformer = T.Compose([
        T.NormalizeFeatures(),
        T.ToDevice(args.device),
        T.RandomNodeSplit(num_val=0.1, num_test=0.2),
    ])
    dataset = __load__data__(args.dataset_func, args.dataset_str, transformer)
    train_mask = dataset.train_mask
    val_mask = dataset.val_mask
    test_mask = dataset.test_mask
    features = dataset.x
    labels = dataset.y

    adj = to_dense_adj(dataset.edge_index).squeeze(dim=0)
    adj_norm = normalize_adj(adj, args.device)

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


def load_synthetic_node_data(args):
    function = globals()[args.dataset_func]
    g, lb, _ = function()
    edge_index = torch.Tensor(np.array(g.edges)).to(torch.int64).to(args.device).t()
    x = []
    for i in g.nodes():
        x.append(g.nodes[i]['feat'])
    x = np.array(x)
    x = torch.Tensor(x).to(torch.float32).to(args.device)
    y = torch.Tensor(np.array(lb)).to(torch.int64).to(args.device)
    data = Data(x=x, edge_index=edge_index, y=y)
    transformer = T.Compose([
        T.ToDevice(args.device),
        T.RandomNodeSplit(split='train_rest', num_val=0.05, num_test=0.10)
    ])
    data_tmp = transformer(data)
    return {
        'data': data,
        'n_classes': len(data_tmp.y.unique()),
        'n_features': data_tmp.x.shape[1],
    }


# loading data for graph classification
def load_graph_data_(args, batch_size=64):
    function = globals()[args.dataset_func]
    path = osp.join(osp.dirname(osp.realpath(__file__)), args.dataset_func).replace("\\", "/")
    transformer = T.Compose([
        T.ToDevice(args.device),
    ])
    dataset = function(path, args.dataset_str,transformer)
    dataset, indices = dataset.shuffle(return_perm=True)
    class_sample_count = np.unique(dataset.y, return_counts=True)[1]
    weight = 1. / class_sample_count


    weight = torch.FloatTensor(weight).to(args.device)
    split = int(len(dataset)*0.85)
    train = dataset[:split]
    test = dataset[split:]
    train = DataLoader(train, batch_size=batch_size, shuffle=False)
    test = DataLoader(test, batch_size=batch_size, shuffle=False)
    return {
        'train': train,
        'test': test,
        'split': split,
        'indices': indices,
        'expl_tst_dt': copy.deepcopy(test),
        'n_features': dataset.num_features,
        'n_classes': dataset.num_classes,
        'weight':weight
    }