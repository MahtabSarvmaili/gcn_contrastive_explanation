import os
import torch
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from utils import get_degree_matrix
from gcn_perturb import GCNSyntheticPerturb
from visualization import plot_errors

torch.manual_seed(0)
np.random.seed(0)
os.environ['KMP_DUPLICATE_LIB_OK']='True'


class CFExplainer:
    """
    CF Explainer class, returns counterfactual subgraph
    """

    def __init__(
            self, model, graph_ae, sub_adj, sub_feat, n_hid, dropout, lr, n_momentum, cf_optimizer,
            sub_labels, y_pred_orig, num_classes, beta, device, AE_threshold, algorithm='cfgnn',
            edge_addition=True, kappa=10
    ):

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
        self.num_classes = num_classes
        self.y_pred_orig = y_pred_orig
        self.y_orig_onehot = F.one_hot(y_pred_orig, num_classes=num_classes)
        self.beta = beta
        self.device = device
        self.kappa = kappa
        self.edge_addition = edge_addition
        self.algorithm = algorithm
        self.losses = {'loss_total':[], 'loss_perturb':[], 'loss_graph_dist':[], 'L1':[], 'L2':[], 'l2_AE':[]}
        # Instantiate CF model class, load weights from original model
        self.cf_model = GCNSyntheticPerturb(
            self.sub_feat.shape[1], n_hid, n_hid,
            self.num_classes, self.sub_adj, dropout, beta, AE_threshold=AE_threshold, edge_addition=edge_addition
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

    def train_cf_model_pn(self, epoch, num_epochs):
        l2_AE = None
        self.cf_optimizer.zero_grad()
        output, output_actual, y_pred_new, y_pred_new_actual = self.predict_cf_model()
        if self.algorithm == 'cfgnn':
            loss_total, loss_perturb, loss_graph_dist, L1, L2, l2_AE, cf_adj = self.cf_model.loss(
                output[self.new_idx], self.y_pred_orig, y_pred_new_actual
            )
        elif self.algorithm == 'loss_PN_L1_L2':
            loss_total, loss_perturb, loss_graph_dist, L1, L2, l2_AE, cf_adj = self.cf_model.loss_PN_L1_L2(
                output[self.new_idx], self.y_orig_onehot
            )
        elif self.algorithm == 'loss_PN_AE_L1_L2':
            loss_total, loss_perturb, loss_graph_dist, L1, L2, l2_AE, cf_adj = self.cf_model.loss_PN_AE_L1_L2(
                self.graph_AE, self.sub_feat, output[self.new_idx], self.y_orig_onehot
            )
        elif self.algorithm == 'loss_PN_AE_pure':
            loss_total, loss_perturb, loss_graph_dist, L1, L2, l2_AE, cf_adj = self.cf_model.loss_PN_AE_pure(
                self.graph_AE, self.sub_feat, output[self.new_idx], self.y_orig_onehot
            )
        else:
            loss_total, loss_perturb, loss_graph_dist, L1, L2, l2_AE, cf_adj = self.cf_model.loss_PN_AE_(
                self.graph_AE, self.sub_feat, output[self.new_idx], self.y_orig_onehot
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
        if y_pred_new_actual != self.y_pred_orig:
            cf_stats = [
                self.node_idx.item(), self.new_idx.item(),
                cf_adj.cpu().detach().numpy(), self.sub_adj.cpu().detach().numpy(),
                self.y_pred_orig.item(), y_pred_new.item(),
                y_pred_new_actual.item(), self.sub_labels[self.new_idx].cpu().detach().numpy(),
                output_actual.argmax(dim=1).cpu(), self.sub_adj.shape[0],
                loss_total.item(), loss_perturb.item(),
                loss_graph_dist.item(), L1,
                L2, l2_AE,
            ]
        return cf_stats, loss_perturb

    def explain(
            self,
            node_idx,
            new_idx,
            num_epochs,
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
            new_example, loss_total = self.train_cf_model_pn(epoch, num_epochs)
            if self.edge_addition:
                if new_example != [] and loss_total <= best_loss:
                    best_cf_example.append(new_example)
                    best_loss = loss_total
                    num_cf_examples += 1
                    print(f'Epoch {epoch}, Num_cf_examples: {num_cf_examples}')
            else:
                if new_example != [] and loss_total <= best_loss:
                    best_cf_example.append(new_example)
                    num_cf_examples += 1
                    best_loss = loss_total
                    print(f'Epoch {epoch}, Num_cf_examples: {num_cf_examples}')
        plot_errors(self.losses)
        return best_cf_example
