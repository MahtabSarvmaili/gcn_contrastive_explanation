import sys
sys.path.append('../..')
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
import pickle
import argparse
import networkx as nx
from collections import Counter
from torch_geometric.utils import dense_to_sparse, degree
import matplotlib.pyplot as plt
from utils import gen_syn1, preprocess_input_graph, normalize_adj, gen_syn2, get_neighbourhood
from feature_generator import ConstFeatureGen
from train_utils import build_optimizer
from graph_conv_net import GCNSynthetic


def main():
    inputdim = 10
    train_ratio = 0.8
    hidden = 20
    seed = 42
    dropout = 0.0

    # G, labels, name = gen_syn1(
    #     feature_generator=ConstFeatureGen(np.ones(inputdim, dtype=float))
    # )
    G, labels, name = gen_syn2()
    input_dim = len(G.nodes[0]["feat"])
    num_classes = max(labels) + 1
    # train/test split only for nodes
    num_nodes = G.number_of_nodes()
    num_train = int(num_nodes * train_ratio)
    idx = [i for i in range(num_nodes)]

    np.random.shuffle(idx)
    train_idx = idx[:num_train]
    test_idx = idx[num_train:]

    data = preprocess_input_graph(G, labels)
    labels_train = torch.tensor(data["labels"][:, train_idx], dtype=torch.long)
    adj = torch.Tensor(data["adj"]).squeeze()  # Does not include self loops
    features = torch.Tensor(data["feat"]).squeeze()
    labels = torch.tensor(data["labels"]).squeeze()
    edge_index = dense_to_sparse(adj)
    norm_adj = normalize_adj(adj)
    model = GCNSynthetic(nfeat=features.shape[1], nhid=hidden, nout=hidden,
                         nclass=len(labels.unique()), dropout=dropout)
    model.cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, weight_decay=5e-4)
    for epoch in range(5):
        optimizer.zero_grad()
        preds = model(features.cuda(), adj.cuda())
        loss = model.loss(preds, labels.long().cuda())
        print(loss)
        loss.backward()
        optimizer.step()

    model.eval()
    output = model(features.cuda(), norm_adj.cuda())
    y_pred_orig = torch.argmax(output, dim=1)
    print("test set y_true counts: {}".format(np.unique(labels.numpy(), return_counts=True)))
    print("test set y_pred_orig counts: {}".format(np.unique(y_pred_orig.cpu().detach().numpy(), return_counts=True)))
    print("Whole graph counts: {}".format(np.unique(labels.numpy(), return_counts=True)))

    with open('../data/syn1.pickle', "rb") as f:
        data = pickle.load(f)

    adj = torch.Tensor(data["adj"]).squeeze()  # Does not include self loops
    features = torch.Tensor(data["feat"]).squeeze()
    labels = torch.tensor(data["labels"]).squeeze()
    idx_train = torch.tensor(data["train_idx"])
    idx_test = torch.tensor(data["test_idx"])
    edge_index = dense_to_sparse(adj)

    norm_adj = normalize_adj(adj)
    model = GCNSynthetic(nfeat=features.shape[1], nhid=hidden, nout=hidden,
                         nclass=len(labels.unique()), dropout=dropout)


if __name__ == '__main__':
    main()