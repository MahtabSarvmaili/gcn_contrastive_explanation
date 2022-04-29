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
from gae.loss import loss_function
from gae.utils import load_data, mask_test_edges, preprocess_graph, get_roc_score
from data.data_loader import load_data_ae
torch.manual_seed(0)
np.random.seed(0)



def gae(args):
    data = load_data_ae(args)
    print("Using {} dataset".format(args.dataset_str))
    model = GCNModelVAE(data['feat_dim'], args.hidden1, args.hidden2, args.dropout)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if args.device == 'cuda':
        model = model.cuda()

    hidden_emb = None
    loss_trace = []
    for epoch in range(args.epochs):
        t = time.time()
        model.train()
        optimizer.zero_grad()
        recovered, mu, logvar = model(data['features'], data['train_adj_norm'])
        loss = loss_function(preds=recovered, pos_labels=data['train_adj'],
                             mu=mu, logvar=logvar, n_nodes=data['n_nodes'],
                             norm=data['norm'], pos_weight=data['pos_weight'], neg_labels=data['train_neg_adj_mask'])
        loss.backward()
        cur_loss = loss.item()
        loss_trace.append(cur_loss)
        optimizer.step()
        hidden_emb = mu.data.cpu().detach().numpy()
        if (epoch+1)%10 == 0:
            model.eval()
            recovered, mu, logvar = model(data['features'], data['val_adj_norm'])
            val_loss = loss_function(preds=recovered, pos_labels=data['val_adj'],
                                 mu=mu, logvar=logvar, n_nodes=data['n_nodes'],
                                 norm=data['norm'], pos_weight=data['pos_weight'])

            roc_curr, ap_curr = get_roc_score(
                hidden_emb, data['adj_orig'], data['val_pos_edge_index'].t(), data['val_neg_edge_index'].t()
            )
            print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(cur_loss),
                  "val_loss=", "{:.5f}".format(val_loss),
                  "val_ap=", "{:.5f}".format(ap_curr),
                  "time=", "{:.5f}".format(time.time() - t)
                  )

    print("Optimization Finished!")
    roc_score, ap_score = get_roc_score(
        hidden_emb, data['adj_orig'], data['test_pos_edge_index'].t(), data['test_neg_edge_index'].t()
    )
    print('Test ROC score: ' + str(roc_score))
    print('Test AP score: ' + str(ap_score))
    return model