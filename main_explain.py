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
from gae.GAE import gae
torch.manual_seed(0)
np.random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda', help='torch device.')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--hidden1', type=int, default=32, help='Number of units in hidden layer 1.')
parser.add_argument('--hidden2', type=int, default=16, help='Number of units in hidden layer 2.')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset-str', type=str, default='cora', help='type of dataset.')
gae_args = parser.parse_args()

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda', help='torch device.')
parser.add_argument('--bb-epochs', type=int, default=500, help='Number of epochs to train the ')
parser.add_argument('--cf-epochs', type=int, default=200, help='Number of epochs to train the ')
parser.add_argument('--inputdim', type=int, default=10, help='Input dimension')
parser.add_argument('--hidden', type=int, default=20, help='Number of units in hidden layer 1.')
parser.add_argument('--n-layers', type=int, default=3, help='Number of units in hidden layer 1.')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--cf-optimizer', type=str, default='SGD', help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset-str', type=str, default='cora', help='type of dataset.')
parser.add_argument('--beta', type=float, default=0.5, help='beta variable')
parser.add_argument('--include_ae', type=bool, default=False, help='Including AutoEncoder reconstruction loss')
parser.add_argument('--n-momentum', type=float, default=0.0, help='Nesterov momentum')
explainer_args = parser.parse_args()


def main(gae_args, explainer_args):
    inputdim = 10
    hidden = 20
    dropout = 0.5
    n_layers = 3

    # data loadig
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
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
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
    output = model(features, adj_norm)
    y_pred_orig = torch.argmax(output[test_mask], dim=1)
    print("test set y_true counts: {}".format(np.unique(labels[test_mask].cpu().detach().numpy(), return_counts=True)))
    print("test set y_pred_orig counts: {}".format(np.unique(y_pred_orig.cpu().detach().numpy(), return_counts=True)))
    print("Training GNN is finished.")
    print("Training AE.")
    graph_ae = gae(gae_args)
    print("Explanation step:")
    idx_test = np.arange(0, n_nodes)[test_mask.cpu()]
    test_cf_examples = []
    for i in idx_test[:20]:
        sub_adj, sub_feat, sub_labels, node_dict = get_neighbourhood(
            int(i), dataset.edge_index, explainer_args.n_layers + 1, features, labels)
        new_idx = node_dict[int(i)]
        # Check that original model gives same prediction on full graph and subgraph
        with torch.no_grad():
            print("Output original model, full adj: {}".format(output[i]))
            print(
                "Output original model, sub adj: {}".format(
                    model(sub_feat, normalize_adj(sub_adj, explainer_args.device).cuda())[new_idx]
                )
            )

        # Need to instantitate new cf model every time because size of P changes based on size of sub_adj
        explainer = CFExplainer(
            model=model,
            graph_ae=graph_ae,
            sub_adj=sub_adj,
            sub_feat=sub_feat,
            n_hid=explainer_args.hidden,
            dropout=explainer_args.dropout,
            sub_labels=sub_labels,
            y_pred_orig=y_pred_orig[i],
            num_classes=num_classes,
            beta=explainer_args.beta,
            device=explainer_args.device
        )
        explainer.cf_model.cuda()
        cf_example = explainer.explain(
            node_idx=i,
            cf_optimizer=explainer_args.cf_optimizer,
            new_idx=new_idx,
            lr=explainer_args.lr,
            n_momentum=explainer_args.n_momentum,
            num_epochs=explainer_args.cf_epochs,
        )  # Need node dict for accuracy calculation
        test_cf_examples.append(cf_example)


if __name__ == '__main__':
    main(gae_args, explainer_args)
