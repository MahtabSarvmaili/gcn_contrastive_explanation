from __future__ import division

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from utils import load_data, get_neighbourhood
from model import GCN, train, test
from cf_explainer import CFExplainer


def main():

    n_layers = 3
    hidden = 20
    dropout = 0.5
    device = 'gpu'
    beta = 0.5
    epoch = 100

    adj, features, labels, edge_index, idx_train, idx_val, idx_test = load_data()
    model = GCN(
        nfeat=features.shape[1],
        nhid=hidden,
        nout=hidden,
        nclasses=len(labels.unique()),
        dropout=True
    )
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    if torch.cuda.is_available():
        model = model.cuda()
        adj = adj.cuda()
        features = features.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()

    train(model, features, adj, labels, idx_train, idx_val, optimizer, epoch)
    test(model, features, adj, labels, idx_test)

    # Get CF examples in test set
    test_cf_examples = []
    for i in idx_test[:]:
        sub_adj, sub_feat, sub_labels, node_dict = get_neighbourhood(int(i), edge_index, n_layers + 1, features,
                                                                     labels)
        new_idx = node_dict[int(i)]
        with torch.no_grad():
            print("Output original model, full adj: {}".format(model(features, adj)[i]))
            print("Output original model, sub adj: {}".format(model(sub_feat, adj(sub_adj))[new_idx]))
        explainer = CFExplainer(
            model=model,
            sub_adj=sub_adj,
            sub_feat=sub_feat,
            n_hid=hidden,
            dropout=dropout,
            sub_labels=sub_labels,
            y_pred_orig=labels[i],
            num_classes=len(labels.unique()),
            beta=beta,
            device=device
        )

if __name__ == '__main__':
    main()