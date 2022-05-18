from __future__ import division
from __future__ import print_function
import argparse
import time
import numpy as np
import scipy.sparse as sp
import torch
from torch import optim
from gae.model import GCNModelVAE
from gae.loss import loss_function
from gae.utils import get_roc_score
torch.manual_seed(0)
np.random.seed(0)


def gae(args, data):

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
        recovered, mu, logvar = model(data['features'], data['adj_norm'])
        loss = loss_function(preds=recovered, labels=data['labels'],
                             mu=mu, logvar=logvar, n_nodes=data['n_nodes'],
                             norm=data['norm'], pos_weight=data['pos_weight'])
        loss.backward()
        cur_loss = loss.item()
        loss_trace.append(cur_loss)
        optimizer.step()
        hidden_emb = mu.data.cpu().numpy()
        roc_curr, ap_curr = get_roc_score(hidden_emb, data['adj_orig'], data['val_edges'], data['val_neg_edge'])


        if epoch%10==0 and epoch!=0:
            print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(cur_loss),
                  "val_ap=", "{:.5f}".format(ap_curr),
                  "time=", "{:.5f}".format(time.time() - t)
                  )
            roc_score, ap_score = get_roc_score(hidden_emb, data['adj_orig'], data['test_edge'], data['test_neg_edge'])
            print('Test ROC score: ' + str(roc_score))
            print('Test AP score: ' + str(ap_score))

    print("Optimization Finished!")

    roc_score, ap_score = get_roc_score(hidden_emb, data['adj_orig'], data['test_edge'], data['test_neg_edge'])
    print('Test ROC score: ' + str(roc_score))
    print('Test AP score: ' + str(ap_score))
    return model