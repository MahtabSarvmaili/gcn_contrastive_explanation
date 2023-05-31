import numpy as np
import pylab as p
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch_geometric.utils import dense_to_sparse
from utils import get_degree_matrix, create_symm_matrix_from_vec, create_vec_from_symm_matrix
from gae.utils_ae import preprocess_graph
# torch.manual_seed(0)
# np.random.seed(0)
from layers import GraphConvolution


def pertinent_negative_loss(output, y_orig_onehot, const, kappa):
    target_lab_score = (output * y_orig_onehot).sum(dim=0)
    max_nontarget_lab_score = (
            (1 - y_orig_onehot) * output -
            (y_orig_onehot * 10000)).max(dim=0).values
    loss_perturb = torch.max(const, -max_nontarget_lab_score + target_lab_score + kappa)
    return loss_perturb, max_nontarget_lab_score


def pertinent_positive_loss(output, y_orig_onehot, const, kappa):
    target_lab_score = (output * y_orig_onehot).sum(dim=0)
    max_nontarget_lab_score = (
            (1 - y_orig_onehot) * output -
            (y_orig_onehot * 10000)).max(dim=0).values
    loss_perturb = torch.max(const, max_nontarget_lab_score - target_lab_score + kappa)
    return loss_perturb, max_nontarget_lab_score


def cross_loss(output, y):
    cross_loss = torch.nn.CrossEntropyLoss()
    closs = cross_loss(output, y)
    return closs


class GraphConvolutionPerturb(nn.Module):

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolutionPerturb, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias is not None:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCNSyntheticPerturb(nn.Module):
    """
    3-layer GCN used in GNN Explainer synthetic tasks
    """

    def __init__(
            self, nfeat, nhid, nout, nclass, adj, dropout,
            beta, cf_expl=True, gamma=0.09, kappa=10, psi=0.01, device='cuda'
    ):
        # the best gamma and psi for prototype explanation are gamma=0.01, kappa=10, psi=0.09
        # the best gamma and psi for CF explanation are gamma=0.09, kappa=10, psi=0.01

        super(GCNSyntheticPerturb, self).__init__()
        self.adj = adj
        self.nclass = nclass
        self.beta = beta
        self.device = device
        self.num_nodes = self.adj.shape[0]
        self.kappa = torch.tensor(kappa).cuda()
        self.beta = torch.tensor(beta).cuda()
        self.const = torch.tensor(0.0, device=device)
        self.gamma = torch.tensor(gamma, device=device)
        self.psi_l1 = torch.tensor(psi, device=device)
        self.cf_expl = cf_expl
        # P_hat needs to be symmetric ==>
        # learn vector representing entries in upper/lower triangular matrix and use to populate P_hat later
        self.P_vec_size = int((self.num_nodes * self.num_nodes - self.num_nodes) / 2) + self.num_nodes

        self.P_vec = Parameter(torch.FloatTensor(torch.ones(self.P_vec_size)))

        self.reset_parameters()

        self.gc1 = GraphConvolutionPerturb(nfeat, nhid)
        self.gc2 = GraphConvolutionPerturb(nhid, nhid)
        self.gc3 = GraphConvolution(nhid, nout)
        self.lin = nn.Linear(nhid + nhid + nout, nclass)
        self.dropout = dropout

    def __L1__(self):
        return torch.linalg.norm(self.P_hat_symm, ord=1)

    def __L2__(self):
        return torch.linalg.norm(self.P_hat_symm, ord=2)

    def __AE_recons__(self, graph_AE, x, cf_adj):
        cf_adj_sparse = dense_to_sparse(cf_adj)[0]
        # the sigmoid is already applied in the reconstruction
        reconst_P = (graph_AE.forward(x, cf_adj_sparse) > 0.5).float()
        l2_AE = torch.dist(cf_adj, reconst_P, p=2)
        return l2_AE

    def __loss_graph_dist__(self, cf_adj):
        return torch.dist(cf_adj , self.adj.cuda(), p=1) / 2

    def reset_parameters(self, eps=10 ** -4):
        # Think more about how to initialize this
        torch.sub(self.P_vec, eps)

    def forward(self, x, sub_adj, logits=True):
        self.sub_adj = sub_adj
        # Same as normalize_adj in utils.py except includes P_hat in A_tilde
        self.P_hat_symm = create_symm_matrix_from_vec(self.P_vec, self.num_nodes)  # Ensure symmetry

        A_tilde = torch.FloatTensor(self.num_nodes, self.num_nodes)
        A_tilde.requires_grad = True

        # Learn P_hat that gets multiplied element-wise with adj -- only edge deletions
        # Use sigmoid to bound P_hat in [0,1]
        # no need for self loop since we normalized the adjacency map
        A_tilde = torch.sigmoid(self.P_hat_symm) * self.sub_adj.cuda() #+ torch.eye(self.num_nodes).cuda()

        D_tilde = get_degree_matrix(A_tilde).detach()  # Don't need gradient of this
        # Raise to power -1/2, set all infs to 0s
        D_tilde_exp = D_tilde ** (-1 / 2)
        D_tilde_exp[torch.isinf(D_tilde_exp)] = 0

        # Create norm_adj = (D + I)^(-1/2) * (A + I) * (D + I) ^(-1/2)
        norm_adj = torch.mm(torch.mm(D_tilde_exp, A_tilde), D_tilde_exp)

        x1 = F.relu(self.gc1(x, norm_adj))
        x1 = F.dropout(x1, self.dropout, training=self.training)
        x2 = F.relu(self.gc2(x1, norm_adj))
        x2 = F.dropout(x2, self.dropout, training=self.training)
        x3 = self.gc3(x2, norm_adj)
        x = self.lin(torch.cat((x1, x2, x3), dim=1))
        if logits:
            return F.log_softmax(x, dim=1)
        else:
            return F.softmax(x, dim=1)

    def forward_prediction(self, x, logits=True):
        # Same as forward but uses P instead of P_hat ==> non-differentiable
        # but needed for actual predictions
        self.P = (torch.sigmoid(self.P_hat_symm) >= 0.5).float()
            # threshold P_hat

        # no need for self loop since we normalized the adjacency map
        A_tilde = self.P * self.adj #+ torch.eye(self.num_nodes).cuda()

        D_tilde = get_degree_matrix(A_tilde)
        # Raise to power -1/2, set all infs to 0s
        D_tilde_exp = D_tilde ** (-1 / 2)
        D_tilde_exp[torch.isinf(D_tilde_exp)] = 0

        # Create norm_adj = (D + I)^(-1/2) * (A + I) * (D + I) ^(-1/2)
        norm_adj = torch.mm(torch.mm(D_tilde_exp, A_tilde), D_tilde_exp)
        x1 = F.relu(self.gc1(x, norm_adj))
        x1 = F.dropout(x1, self.dropout, training=self.training)
        x2 = F.relu(self.gc2(x1, norm_adj))
        x2 = F.dropout(x2, self.dropout, training=self.training)
        x3 = self.gc3(x2, norm_adj)
        x = self.lin(torch.cat((x1, x2, x3), dim=1))
        if logits:
            return F.log_softmax(x, dim=1)
        else:
            return F.softmax(x, dim=1)

    def encode(self, x, sub_adj):
        self.sub_adj = sub_adj
        # Same as normalize_adj in utils.py except includes P_hat in A_tilde
        self.P_hat_symm = create_symm_matrix_from_vec(self.P_vec, self.num_nodes)  # Ensure symmetry

        A_tilde = torch.FloatTensor(self.num_nodes, self.num_nodes)
        A_tilde.requires_grad = True
        # Learn P_hat that gets multiplied element-wise with adj -- only edge deletions
        A_tilde = torch.sigmoid(self.P_hat_symm) * self.sub_adj.cuda()
        # + torch.eye(self.num_nodes).cuda()  # Use sigmoid to bound P_hat in [0,1]

        D_tilde = get_degree_matrix(A_tilde).detach()  # Don't need gradient of this
        # Raise to power -1/2, set all infs to 0s
        D_tilde_exp = D_tilde ** (-1 / 2)
        D_tilde_exp[torch.isinf(D_tilde_exp)] = 0

        # Create norm_adj = (D + I)^(-1/2) * (A + I) * (D + I) ^(-1/2)
        norm_adj = torch.mm(torch.mm(D_tilde_exp, A_tilde), D_tilde_exp)

        x1 = F.relu(self.gc1(x, norm_adj))
        x1 = F.dropout(x1, self.dropout, training=self.training)
        x2 = F.relu(self.gc2(x1, norm_adj))
        x2 = F.dropout(x2, self.dropout, training=self.training)
        x3 = self.gc3(x2, norm_adj)
        x = torch.cat((x1, x2, x3), dim=1)
        return x

    def encode_prediction(self, x):
        # Same as forward but uses P instead of P_hat ==> non-differentiable
        # but needed for actual predictions
        self.P = (torch.sigmoid(self.P_hat_symm) >= 0.5).float()  # threshold P_hat

        A_tilde = self.P * self.adj #+ torch.eye(self.num_nodes).cuda()

        D_tilde = get_degree_matrix(A_tilde)
        # Raise to power -1/2, set all infs to 0s
        D_tilde_exp = D_tilde ** (-1 / 2)
        D_tilde_exp[torch.isinf(D_tilde_exp)] = 0

        # Create norm_adj = (D + I)^(-1/2) * (A + I) * (D + I) ^(-1/2)
        norm_adj = torch.mm(torch.mm(D_tilde_exp, A_tilde), D_tilde_exp)
        x1 = F.relu(self.gc1(x, norm_adj))
        x1 = F.dropout(x1, self.dropout, training=self.training)
        x2 = F.relu(self.gc2(x1, norm_adj))
        x2 = F.dropout(x2, self.dropout, training=self.training)
        x3 = self.gc3(x2, norm_adj)
        x = torch.cat((x1, x2, x3), dim=1)
        return x

    def loss(self, output, y_pred_orig, y_pred_new_actual):
        PLoss = 0
        pred_same = (y_pred_new_actual == y_pred_orig).float()

        # Need dim >=2 for F.nll_loss to work
        output = output.unsqueeze(0)
        y_pred_orig = y_pred_orig.unsqueeze(0)
        cf_adj = self.P * self.adj
        cf_adj.requires_grad = True  # Need to change this otherwise loss_graph_dist has no gradient

        # Want negative in front to maximize loss instead of minimizing it to find CFs
        loss_pred = - F.nll_loss(output, y_pred_orig)
        loss_graph_dist = sum(sum(abs(cf_adj - self.adj.cuda()))) / 2  # Number of edges changed (symmetrical)

        # Zero-out loss_pred with pred_same if prediction flips
        loss_total = pred_same * loss_pred + self.beta * loss_graph_dist
        return loss_total, loss_pred, loss_graph_dist, torch.inf, torch.inf, torch.inf, cf_adj, PLoss

    def loss__(self, graph_AE, x, output, y_orig_onehot, l1=1, l2=1, ae=1, dist=1):
        closs = 0
        if self.cf_expl is False:
            loss_perturb, PLoss = pertinent_positive_loss(output, y_orig_onehot, self.const, self.kappa)
        else:
            loss_perturb, PLoss = pertinent_negative_loss(output, y_orig_onehot, self.const, self.kappa)
        cf_adj = self.P * self.adj
        cf_adj.requires_grad = True  # Need to change this otherwise loss_graph_dist has no gradient
        loss_graph_dist = torch.dist(cf_adj , self.adj.cuda(), p=1) / 2
        l2_AE = self.__AE_recons__(graph_AE, x, cf_adj)
        L1 = self.__L1__()
        L2 = self.__L2__()
        # if self.cf_expl is False:
        #     closs = cross_loss(output.unsqueeze(dim=0), y_orig_onehot.argmax(keepdims=True))
        loss_total = loss_perturb + dist * self.beta * loss_graph_dist + l1 * self.psi_l1 * L1 + l2 * L2 + ae * self.gamma * l2_AE + closs
        return loss_total, loss_perturb, loss_graph_dist, L1.item(), L2.item(), l2_AE.item(), cf_adj, PLoss.item()

    def loss__nll(self, graph_AE, x, output, y_pred_orig, y_pred_new_actual, l1=1, l2=1, ae=1, dist=1):
        PLoss = 0

        output = output.unsqueeze(0)
        y_pred_orig = y_pred_orig.unsqueeze(0)
        cf_adj = self.P * self.adj
        cf_adj.requires_grad = True  # Need to change this otherwise loss_graph_dist has no gradient
        loss_graph_dist = torch.dist(cf_adj , self.adj.cuda(), p=1) / 2
        l2_AE = self.__AE_recons__(graph_AE, x, cf_adj)
        L1 = self.__L1__()
        L2 = self.__L2__()

        if self.cf_expl is False:
            pred_same = (y_pred_new_actual != y_pred_orig).float()
            loss_pred = F.nll_loss(output, y_pred_orig)

        else:
            pred_same = (y_pred_new_actual == y_pred_orig).float()
            loss_pred = - F.nll_loss(output, y_pred_orig)
        loss_perturb = pred_same * loss_pred

        loss_total = loss_perturb + dist * self.beta * loss_graph_dist + l1 * self.psi_l1 * L1 + l2 * L2 + ae * self.gamma * l2_AE
        return loss_total, loss_perturb, loss_graph_dist, L1.item(), L2.item(), l2_AE.item(), cf_adj, PLoss.item()