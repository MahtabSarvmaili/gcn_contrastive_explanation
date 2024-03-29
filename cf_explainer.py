import copy
import os
import torch
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from utils import get_degree_matrix
from gcn_perturb import GCNSyntheticPerturb
from visualization import plot_errors

# torch.manual_seed(0)
# np.random.seed(0)
os.environ['KMP_DUPLICATE_LIB_OK']='True'


class CFExplainer:
    """
    CF Explainer class, returns counterfactual subgraph
    """

    def __init__(
            self, model, graph_ae, sub_adj, sub_feat, n_hid, dropout, lr, n_momentum, cf_optimizer,
            sub_labels, y_pred_orig, num_classes, beta, device, cf_expl=True, algorithm='cfgnn', kappa=10
    ):

        super(CFExplainer, self).__init__()
        self.model = model
        self.model.eval()
        self.graph_AE = graph_ae
        self.graph_AE.eval()
        self.sub_adj = copy.deepcopy(sub_adj)
        self.sub_feat = sub_feat
        self.n_hid = n_hid
        self.dropout = dropout
        self.sub_labels = sub_labels
        self.num_classes = num_classes
        self.y_pred_orig = y_pred_orig
        self.y_orig_onehot = F.one_hot(y_pred_orig, num_classes=num_classes)
        self.beta = beta
        self.device = device
        self.kappa = kappa
        self.algorithm = algorithm
        self.cf_expl = cf_expl
        self.losses = {
            'loss_total': [], 'loss_perturb': [], 'loss_graph_dist': [], 'L1': [], 'L2': [], 'l2_AE': []
        }
        self.params = {
            'loss_PN': (0,0,0,0),
            'loss_PN_dist': (0,0,0,1),
            'loss_PN_AE': (0,0,1,0),
            'loss_PN_L1_L2': (1,1,0,0),
            'loss_PN_AE_L1_L2': (1,1,1,0),
            'loss_PN_AE_L1_L2_dist': (1,1,1,1)
        }
        # Instantiate CF model class, load weights from original model
        self.cf_model = GCNSyntheticPerturb(
            self.sub_feat.shape[1], n_hid, n_hid,
            self.num_classes, self.sub_adj, dropout,
            beta, self.cf_expl, 0.1, self.kappa, 0.09
        )

        self.cf_model.load_state_dict(self.model.state_dict(), strict=False)

        # Freeze weights from original model in cf_model
        for name, param in self.cf_model.named_parameters():
            if name.endswith("weight") or name.endswith("bias"):
                param.requires_grad = False
        for name, param in self.model.named_parameters():
            print("orig model requires_grad: ", name, param.requires_grad)
        for name, param in self.cf_model.named_parameters():
            print("cf model required_grad: ", name, param.requires_grad)

        if cf_optimizer == "SGD" and n_momentum == 0.0:
            self.cf_optimizer = optim.SGD(self.cf_model.parameters(), lr=lr)
        elif cf_optimizer == "SGD" and n_momentum != 0.0:
            self.cf_optimizer = optim.SGD(self.cf_model.parameters(), lr=lr, nesterov=True, momentum=n_momentum)
        elif cf_optimizer == "Adadelta":
            self.cf_optimizer = optim.Adadelta(self.cf_model.parameters(), lr=lr)
        elif cf_optimizer == "Adam":
            self.cf_optimizer = optim.Adam(self.cf_model.parameters(), lr=lr)

    def predict_cf_model(self):
        self.cf_model.train()
        self.cf_optimizer.zero_grad()
        output = self.cf_model.forward(self.x, self.A_x, logits=False)
        output_actual = self.cf_model.forward_prediction(self.x, logits=False)
        y_pred_new = torch.argmax(output[self.new_idx])
        y_pred_new_actual = torch.argmax(output_actual[self.new_idx])
        return output, output_actual, y_pred_new, y_pred_new_actual

    def train_cf_model_pn(self, epoch):
        l2_AE = None
        self.cf_optimizer.zero_grad()
        output, output_actual, y_pred_new, y_pred_new_actual = self.predict_cf_model()
        if self.algorithm.__contains__('cfgnn'):
            loss_total, loss_perturb, loss_graph_dist, L1, L2, l2_AE, cf_adj, PLoss = self.cf_model.loss(
                output[self.new_idx], self.y_pred_orig, y_pred_new_actual
            )
        elif self.algorithm.__contains__('nll'):
            l1, l2, ae, dist = self.params[self.algorithm.replace('nll', '')]
            loss_total, loss_perturb, loss_graph_dist, L1, L2, l2_AE, cf_adj, PLoss = self.cf_model.loss__nll(
                self.graph_AE, self.sub_feat, output[self.new_idx], self.y_pred_orig, y_pred_new_actual,
                l1=l1, l2=l2, ae=ae, dist=dist
            )
        else:
            l1, l2, ae, dist = self.params[self.algorithm]
            loss_total, loss_perturb, loss_graph_dist, L1, L2, l2_AE, cf_adj, PLoss = self.cf_model.loss__(
                self.graph_AE, self.sub_feat, output[self.new_idx], self.y_orig_onehot, l1=l1, l2=l2, ae=ae, dist=dist
            )

        self.losses['loss_total'].append(loss_total.item())
        self.losses['loss_graph_dist'].append(loss_graph_dist.item())
        self.losses['loss_perturb'].append(loss_perturb.item())
        self.losses['L1'].append(L1)
        self.losses['L2'].append(L2)
        self.losses['l2_AE'].append(l2_AE)

        loss_total.backward()
        clip_grad_norm_(self.cf_model.parameters(), 2.0)
        self.cf_optimizer.step()

        if epoch % 10 == 0 and epoch != 0:
            print(
                'Node idx: {}'.format(self.node_idx),
                'New idx: {}'.format(self.new_idx),
                'Epoch: {:04d}'.format(epoch + 1),
                'loss: {:.4f}'.format(loss_total.item()),
                'pred loss: {:.4f}'.format(loss_perturb.item()),
                'graph loss: {:.4f}'.format(loss_graph_dist.item()),
            )
            print(" ")
        cf_stats = []
        if self.cf_expl:
            if y_pred_new_actual != self.y_pred_orig:
                cf_stats = [
                    self.node_idx.item(), self.new_idx.item(),
                    cf_adj.cpu().detach().numpy(), epoch,
                    self.y_pred_orig.item(), y_pred_new.item(),
                    y_pred_new_actual.item(), self.sub_labels[self.new_idx].cpu().detach().numpy(),
                    output_actual.argmax(dim=1).cpu(), self.sub_adj.shape[0],
                    loss_total.item(), loss_perturb.item(),
                    loss_graph_dist.item(), L1,
                    L2, l2_AE,
                    PLoss
                ]
        else:
            if y_pred_new_actual == self.y_pred_orig:
                cf_stats = [
                    self.node_idx.item(), self.new_idx.item(),
                    cf_adj.cpu().detach().numpy(), epoch,
                    self.y_pred_orig.item(), y_pred_new.item(),
                    y_pred_new_actual.item(), self.sub_labels[self.new_idx].cpu().detach().numpy(),
                    output_actual.argmax(dim=1).cpu(), self.sub_adj.shape[0],
                    loss_total.item(), loss_perturb.item(),
                    loss_graph_dist.item(), L1,
                    L2, l2_AE,
                    PLoss
                ]
        return cf_stats, loss_total.item()

    def explain(
            self,
            node_idx,
            new_idx,
            num_epochs,
            path=''
    ):
        self.node_idx = node_idx
        self.new_idx = new_idx

        self.x = self.sub_feat
        self.A_x = self.sub_adj
        self.D_x = get_degree_matrix(self.A_x)

        best_cf_example = []
        best_loss = np.inf
        num_cf_examples = 0
        for epoch in range(num_epochs):
            new_example, loss_total = self.train_cf_model_pn(epoch)
            if new_example != [] and loss_total <= best_loss and new_example[2].sum() < self.sub_adj.sum():
                best_cf_example.append(new_example)
                best_loss = loss_total
                num_cf_examples += 1
                print(f'Epoch {epoch}, Num_cf_examples: {num_cf_examples}, Best Loss:{best_loss}')

        plot_errors(self.losses, path)
        for x in best_cf_example:
            print(x[2].sum())
        return best_cf_example
