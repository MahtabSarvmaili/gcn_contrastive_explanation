# Based on https://github.com/RexYing/gnn-model-explainer/blob/master/explainer/explain.py
import os
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.nn.utils import clip_grad_norm_
from utils import get_degree_matrix
from gae.utils import preprocess_graph
from gcn_perturb import GCNSyntheticPerturb

torch.manual_seed(0)
os.environ['KMP_DUPLICATE_LIB_OK']='True'


class CFExplainer:
    """
    CF Explainer class, returns counterfactual subgraph
    """

    def __init__(self, model, graph_ae, sub_adj, sub_feat, n_hid, dropout,
                 sub_labels, y_pred_orig, num_classes, beta, device, edge_additions=True, kappa=10):
        super(CFExplainer, self).__init__()
        self.model = model
        self.model.eval()
        self.graph_AE = graph_ae
        self.graph_AE.eval()
        self.sub_adj = sub_adj
        self.sub_feat = sub_feat
        self.n_hid = n_hid
        self.dropout = dropout
        self.sub_labels = sub_labels
        self.y_pred_orig = y_pred_orig
        self.beta = beta
        self.num_classes = num_classes
        self.device = device
        self.kappa = kappa
        self.edge_additions = edge_additions
        # Instantiate CF model class, load weights from original model
        self.cf_model = GCNSyntheticPerturb(self.sub_feat.shape[1], n_hid, n_hid,
                                            self.num_classes, self.sub_adj, dropout, beta, edge_additions=True)

        self.cf_model.load_state_dict(self.model.state_dict(), strict=False)

        # Freeze weights from original model in cf_model
        for name, param in self.cf_model.named_parameters():
            if name.endswith("weight") or name.endswith("bias"):
                param.requires_grad = False
        for name, param in self.model.named_parameters():
            print("orig model requires_grad: ", name, param.requires_grad)
        for name, param in self.cf_model.named_parameters():
            print("cf model required_grad: ", name, param.requires_grad)

    def train_cf_model(self):
        self.cf_model.train()
        self.cf_optimizer.zero_grad()
        output = self.cf_model.forward(self.x, self.A_x, logits=False)
        output_actual = self.cf_model.forward_prediction(self.x, logits=False)
        y_pred_new = torch.argmax(output[self.new_idx])
        y_pred_new_actual = torch.argmax(output_actual[self.new_idx])
        return output, output_actual, y_pred_new, y_pred_new_actual

    def train(self, epoch):

        output, output_actual, y_pred_new, y_pred_new_actual = self.train_cf_model()
        loss_total, loss_pred, loss_graph_dist, cf_adj = self.cf_model.loss(
            output[self.new_idx], self.y_pred_orig,y_pred_new_actual
        )
        loss_total.backward()
        clip_grad_norm_(self.cf_model.parameters(), 2.0)
        self.cf_optimizer.step()
        print('Node idx: {}'.format(self.node_idx),
              'New idx: {}'.format(self.new_idx),
              'Epoch: {:04d}'.format(epoch + 1),
              'loss: {:.4f}'.format(loss_total.item()),
              'pred loss: {:.4f}'.format(loss_pred.item()),
              'graph loss: {:.4f}'.format(loss_graph_dist.item()))
        print('Output: {}\n'.format(output[self.new_idx].data),
              'Output nondiff: {}\n'.format(output_actual[self.new_idx].data),
              'orig pred: {}, new pred: {}, new pred nondiff: {}'.format(self.y_pred_orig, y_pred_new,
                                                                         y_pred_new_actual))
        print(" ")
        cf_stats = []
        if y_pred_new_actual != self.y_pred_orig:
            cf_stats = [self.node_idx.item(), self.new_idx.item(),
                        cf_adj.cpu().detach().numpy(), self.sub_adj.cpu().detach().numpy(),
                        self.y_pred_orig.item(), y_pred_new.item(),
                        y_pred_new_actual.item(), self.sub_labels[self.new_idx].cpu().numpy(),
                        self.sub_adj.shape[0], loss_total.item(),
                        loss_pred.item(), loss_graph_dist.item()]

        return (cf_stats, loss_total.item())

    def train_cf_model_pn(self, epoch, encode_sub_features):
        output, output_actual, y_pred_new, y_pred_new_actual = self.train_cf_model()
        # loss_total, loss_perturb, loss_graph_dist, cf_adj = self.cf_model.loss_PN_L1_L2(
        #     output[self.new_idx], self.y_pred_orig, self.sub_adj
        # )
        loss_total, loss_perturb, loss_graph_dist, cf_adj = self.cf_model.loss_PN_AE_L1_L2(
            self.graph_AE, self.sub_feat, output[self.new_idx], self.y_pred_orig, self.sub_adj
        )

        loss_total.backward()
        clip_grad_norm_(self.cf_model.parameters(), 2.0)
        self.cf_optimizer.step()
        if epoch%10 == 0 and epoch != 0:
            print('Node idx: {}'.format(self.node_idx),
                  'New idx: {}'.format(self.new_idx),
                  'Epoch: {:04d}'.format(epoch + 1),
                  'loss: {:.4f}'.format(loss_total.item()),
                  'pred loss: {:.4f}'.format(loss_perturb.item()),
                  'graph loss: {:.4f}'.format(loss_graph_dist.item()))
            print(" ")
        cf_stats = []

        if y_pred_new_actual != self.y_pred_orig:
            cf_stats = [self.node_idx.item(), self.new_idx.item(),
                        cf_adj.cpu().detach().numpy(), self.sub_adj.cpu().detach().numpy(),
                        self.y_pred_orig.item(), y_pred_new.item(),
                        y_pred_new_actual.item(), self.sub_labels[self.new_idx].cpu().detach().numpy(),
                        self.sub_adj.shape[0], loss_total.item(), loss_perturb.item(), loss_graph_dist.item()]
        return cf_stats, loss_total

    def explain(self, cf_optimizer, node_idx, new_idx, lr, n_momentum, num_epochs, encode_sub_features, PN_training=True):
        self.node_idx = node_idx
        self.new_idx = new_idx

        self.x = self.sub_feat
        self.A_x = self.sub_adj
        self.D_x = get_degree_matrix(self.A_x)

        if cf_optimizer == "SGD" and n_momentum == 0.0:
            self.cf_optimizer = optim.SGD(self.cf_model.parameters(), lr=lr)
        elif cf_optimizer == "SGD" and n_momentum != 0.0:
            self.cf_optimizer = optim.SGD(self.cf_model.parameters(), lr=lr, nesterov=True, momentum=n_momentum)
        elif cf_optimizer == "Adadelta":
            self.cf_optimizer = optim.Adadelta(self.cf_model.parameters(), lr=lr)

        perturb_loss_array = []
        total_loss_ = []
        best_cf_example = []
        best_loss = np.inf
        num_cf_examples = 0
        for epoch in range(num_epochs):
            if PN_training:
                new_example, loss_total = self.train_cf_model_pn(epoch, encode_sub_features)
                if new_example != [] and loss_total < best_loss:
                    best_cf_example.append(new_example)
                    best_loss = loss_total
                    num_cf_examples += 1
            else:
                new_example, loss_total = self.train(epoch)
                if new_example != [] and loss_total < best_loss:
                    best_cf_example.append(new_example)
                    best_loss = loss_total
                    num_cf_examples += 1
                total_loss_.append(loss_total)
                loss_ = loss_total

        # print("{} CF examples for node_idx = {}".format(num_cf_examples, self.node_idx))
        # print(" ")
        return (best_cf_example)

