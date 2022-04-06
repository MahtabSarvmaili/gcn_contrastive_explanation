import argparse
import sys

sys.path.append('../..')
import torch
import numpy as np
import pickle
import os.path as osp
import scipy.sparse as sp

from torch_geometric.utils import to_dense_adj
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.utils import dense_to_sparse
from utils import normalize_adj, get_neighbourhood
from model import GCN, train
from cf_explainer import CFExplainer
from gae.utils import preprocess_graph
torch.manual_seed(0)
np.random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda', help='torch device.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--hidden1', type=int, default=32, help='Number of units in hidden layer 1.')
parser.add_argument('--hidden2', type=int, default=16, help='Number of units in hidden layer 2.')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--dropout', type=float, default=0., help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset-str', type=str, default='cora', help='type of dataset.')
gae_args = parser.parse_args()

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda', help='torch device.')
parser.add_argument('--epochs', type=int, default=500, help='Number of epochs to train.')
parser.add_argument('--inputdim', type=int, default=10, help='Input dimension')
parser.add_argument('--hidden1', type=int, default=32, help='Number of units in hidden layer 1.')
parser.add_argument('--hidden2', type=int, default=16, help='Number of units in hidden layer 2.')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--cf-optimizer', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset-str', type=str, default='cora', help='type of dataset.')
parser.add_argument('--beta', type=float, default=0.5, help='beta variable')
parser.add_argument('--include_ae', type=bool, default=False, help='Including AutoEncoder reconstruction loss')
explainer_args = parser.parse_args()


def main(gae_args, explainer_args):
    inputdim = 10
    train_ratio = 0.8
    hidden = 20
    seed = 42
    dropout = 0.5
    n_layers = 3
    epoch = 600
    device = 'cuda'
    beta = 0.5
    learning_rate = 0.001
    n_momentum = 0.0
    cf_optimizer = 'SGD'
    # for Pertinent Negative -> edge_additions = True / Pertinent Positive -> edge_additions = False
    edge_additions = True

    # data loadig
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Planetoid')
    transform = T.Compose([
        T.NormalizeFeatures(),
        T.ToDevice(explainer_args.device),
        T.RandomLinkSplit(num_val=0.1, num_test=0.2, is_undirected=True,
                          split_labels=True, add_negative_train_samples=False),
    ])
    dataset = Planetoid(path, explainer_args.dataset_str, transform=transform)
    train_data, val_data, test_data = dataset[0]


    train_adj = to_dense_adj(train_data.pos_edge_label_index, max_num_nodes=train_data.num_nodes).squeeze(dim=0)
    train_adj_norm = preprocess_graph(train_adj.cpu(), explainer_args.device)
    val_adj = to_dense_adj(val_data.pos_edge_label_index, max_num_nodes=val_data.num_nodes).squeeze(dim=0)
    test_adj = to_dense_adj(test_data.pos_edge_label_index, max_num_nodes=test_data.num_nodes).squeeze(dim=0)

    print("Using {} dataset".format(explainer_args.dataset_str))

    n_nodes, feat_dim = train_data.num_nodes, train_data.num_features
    # Store original adjacency matrix (without diagonal entries) for later
    adj_orig = train_adj.cpu().detach().numpy()
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    pos_weight = torch.Tensor(
        [float(train_adj.shape[0] * train_adj.shape[0] - train_adj.sum()) / train_adj.sum()],

    )
    norm = train_adj.shape[0] * train_adj.shape[0] / float((train_adj.shape[0] * train_adj.shape[0] - train_adj.sum()) * 2)


    # with open('data/syn1.pickle', "rb") as f:
    #     data = pickle.load(f)
    #
    #
    # adj = torch.Tensor(data["adj"]).squeeze()  # Does not include self loops
    # features = torch.Tensor(data["feat"]).squeeze()
    # labels = torch.tensor(data["labels"]).squeeze()
    # idx_train = torch.tensor(data["train_idx"])
    # idx_test = torch.tensor(data["test_idx"])
    # edge_index = dense_to_sparse(adj)
    # norm_adj = normalize_adj(adj)
    model = GCN(
        nfeat=train_data.num_features,
        nhid=hidden,
        nout=hidden,
        nclasses=len(train_data.y.unique()),
        dropout=dropout
    )
    # model_ae = linear_autoencoder(in_channels=inputdim, out_channels=inputdim)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)

    if explainer_args.device=='cuda':
        model = model.cuda()


    train(
        model=model,
        train_features=train_data.x,
        train_adj=train_adj_norm,
        train_labels=train_data.y,
        train_mask=train_data.train_mask,
        optimizer=optimizer,
        epoch=epoch,
        val_features=val_data.x,
        val_adj=val_adj,
        val_label=val_data.y,
        val_mask=val_data.val_mask
    )

    model.eval()
    output = model(test_data.x, test_adj)
    y_pred_orig = torch.argmax(output, dim=1)
    print("test set y_true counts: {}".format(np.unique(test_data.y.cpu().detach().numpy(), return_counts=True)))
    print("test set y_pred_orig counts: {}".format(np.unique(y_pred_orig.cpu().detach().numpy(), return_counts=True)))
    idx_test = np.arange(0,test_data.num_nodes)[test_data.test_mask.cpu()]
    test_cf_examples = []
    for i in idx_test[:20]:
        sub_adj, sub_feat, sub_labels, node_dict = get_neighbourhood(
            int(i), test_data.pos_edge_label_index, n_layers + 1, test_data.x, test_data.y)
        new_idx = node_dict[int(i)]

        # Check that original model gives same prediction on full graph and subgraph
        with torch.no_grad():
            print("Output original model, full adj: {}".format(output[i]))
            print("Output original model, sub adj: {}".format(model(sub_feat, normalize_adj(sub_adj).cuda())[new_idx]))

        # Need to instantitate new cf model every time because size of P changes based on size of sub_adj

        if device == 'cuda':
            model.cuda()
            adj = adj.cuda()
            norm_adj = norm_adj.cuda()
            features = features.cuda()
            labels = labels.cuda()
            idx_train = idx_train.cuda()
            idx_test = idx_test.cuda()

        explainer = CFExplainer(
            model=model,
            sub_adj=sub_adj,
            sub_feat=sub_feat,
            n_hid=hidden,
            dropout=dropout,
            sub_labels=sub_labels,
            y_pred_orig=y_pred_orig[i],
            num_classes=len(labels.unique()),
            beta=beta,
            device=device
        )
        explainer.cf_model.cuda()
        cf_example = explainer.explain(
            node_idx=i,
            cf_optimizer=cf_optimizer,
            new_idx=new_idx,
            lr=learning_rate,
            n_momentum=n_momentum,
            num_epochs=epoch,
        )  # Need node dict for accuracy calculation
        test_cf_examples.append(cf_example)


if __name__ == '__main__':
    main(gae_args, explainer_args)
