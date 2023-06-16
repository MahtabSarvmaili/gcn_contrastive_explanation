from __future__ import division
import argparse
import numpy as np
import torch
from clustering.main_dmon import DMon
from visualization import plotClusters
from model import GCN_dep, train
from data.gengraph import gen_syn1
from data.data_loader import load_synthetic, load_data

np.random.seed(0)
torch.manual_seed(0)

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda', help='torch device.')
parser.add_argument('--bb-epochs', type=int, default=400, help='Number of epochs to train bb')
parser.add_argument('--dmon-epochs', type=int, default=370, help='Number of epochs to train bb')
parser.add_argument('--hidden', type=int, default=20, help='Number of units in hidden layer 1.')
parser.add_argument('--n-layers', type=int, default=3, help='Number of units in hidden layer 1.')
parser.add_argument('--num-clusters', type=int, default=16, help='Number of units in hidden layer 1.')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--dmon-lr', type=float, default=0.0001, help='Initial learning rate.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset-str', type=str, default='cora', help='type of dataset.')
parser.add_argument('--beta', type=float, default=0.5, help='beta variable')
parser.add_argument('--n-momentum', type=float, default=0.0, help='Nesterov momentum')
explainer_args = parser.parse_args()


def main(explainer_args):

    dataset_name = gen_syn1.__name__
    data = load_synthetic(gen_syn1, 'cuda')
    # data = load_data(explainer_args)
    # dataset_name = 'cora'
    model = GCN_dep(
        nfeat=data['feat_dim'],
        nhid=explainer_args.hidden,
        nout=explainer_args.hidden,
        nclasses=data['num_classes'],
        dropout=explainer_args.dropout
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=explainer_args.lr, weight_decay=5e-4)

    if explainer_args.device=='cuda':
        model = model.cuda()

    train(
        model=model,
        features=data['features'],
        train_adj=data['adj_norm'],
        labels=data['labels'],
        train_mask=data['train_mask'],
        optimizer=optimizer,
        epoch=explainer_args.bb_epochs,
        val_mask=data['val_mask'],
        dataset_name=dataset_name
    )
    model.eval()
    data['cluster_features'] = model.encode(data['features'], data['adj_norm']).detach()
    output = model(data['features'], data['adj_norm'])
    y_pred_orig = torch.argmax(output[data['test_mask']], dim=1)
    print("test set y_true counts: {}".format(np.unique(data['labels'][data['test_mask']].cpu().detach().numpy(), return_counts=True)))
    print("test set y_pred_orig counts: {}".format(np.unique(y_pred_orig.cpu().detach().numpy(), return_counts=True)))
    print("Training GNN is finished.")

    print("Training clustering model:")
    dmon = DMon(data, model, explainer_args.num_clusters, explainer_args.dmon_epochs, explainer_args.dmon_lr, dataset_name)
    cluster_features, assignments = dmon(data['cluster_features'])
    plotClusters(
        data['cluster_features'].cpu(),
        data['labels'].cpu(),
        cluster_features.cpu().detach(),
        f'org_dt_{dataset_name}',
        explainer_args.num_clusters
    )
    plotClusters(
        data['cluster_features'].cpu(),
        assignments.argmax(dim=1).cpu(),
        cluster_features.cpu().detach(),
        f'clus_dt_{dataset_name}',
        data['num_classes'],
    )

if __name__ == '__main__':
    main(explainer_args)