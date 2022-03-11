import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from layers import GraphConvolution
from utils import accuracy


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nout, nclasses, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.gc3 = GraphConvolution(nhid, nout)
        self.lin = nn.Linear(nhid+nhid+nout, nclasses)
        self.dropout = dropout

    def forward(self, x, adj):
        x1 = F.relu(self.gc1(x, adj))
        x1 = F.dropout(x1, self.dropout, training=False)
        x2 = F.relu(self.gc2(x1, adj))
        x2 = F.dropout(x2, self.dropout, training=self.training)
        x3 = self.gc3(x2, adj)
        x = self.lin(torch.cat((x1, x2, x3), dim=1))
        return F.log_softmax(x, dim=1)


def train(
        model:nn.Module,
        features,
        adj,
        labels,
        idx_train,
        idx_val,
        optimizer:optim,
        epoch
):
    model.train()
    for i in range(epoch):
        idx_train = idx_train[torch.randperm(idx_train.size()[0])]
        idx_val = idx_val[torch.randperm(idx_val.size()[0])]

        optimizer.zero_grad()
        preds = model(features, adj)
        loss_train = F.nll_loss(preds[idx_train], labels[idx_train])
        acc_train = accuracy(preds[idx_train], labels[idx_train])
        loss_train.backward()
        optimizer.step()

        if i%50==0 and i!=0:
            model.eval()
            preds = model(features, adj)
            loss_val = F.nll_loss(preds[idx_val], labels[idx_val])
            acc_val = accuracy(preds[idx_val], labels[idx_val])
            print('Epoch: {:04d}'.format(i),
                  'loss_train: {:.4f}'.format(loss_train.item()),
                  'acc_train: {:.4f}'.format(acc_train.item()),
                  'loss_val: {:.4f}'.format(loss_val.item()),
                  'acc_val: {:.4f}'.format(acc_val.item()))


def test(
        model: nn.Module,
        features,
        adj,
        labels,
        idx_test
):
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))
    y_pred_orig = torch.argmax(output, dim=1)
    print("y_true counts: {}".format(np.unique(labels.cpu().detach().numpy(), return_counts=True)))
    print("y_pred_orig counts: {}".format(np.unique(y_pred_orig.cpu().detach().numpy(), return_counts=True)))