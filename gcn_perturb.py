import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch_geometric.utils import dense_to_sparse
from utils import get_degree_matrix, create_symm_matrix_from_vec, create_vec_from_symm_matrix
from gae.utils import preprocess_graph
from clustering.dmon import DMoN
torch.manual_seed(0)
np.random.seed(0)
from layers import GraphConvolution


def pertinent_negative_loss(output, y_orig_onehot, const, kappa):
    target_lab_score = (output * y_orig_onehot).sum(dim=0)
    max_nontarget_lab_score = (
            (1 - y_orig_onehot) * output -
            (y_orig_onehot * 10000)).max(dim=0).values
    loss_perturb = torch.max(const, -max_nontarget_lab_score + target_lab_score + kappa)
    return loss_perturb


class GraphConvolutionPerturb(nn.Module):
    """
    Similar to GraphConvolution except includes P_hat
    """

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
            beta, gamma=0.1, kappa=10, psi=0.1, AE_threshold=0.5, edge_addition=False, device='cuda'
    ):
        super(GCNSyntheticPerturb, self).__init__()
        self.adj = adj
        self.nclass = nclass
        self.beta = beta
        self.device = device
        self.num_nodes = self.adj.shape[0]
        self.AE_threshold = AE_threshold
        self.edge_addition = edge_addition  # are edge additions included in perturbed matrix
        self.kappa = torch.tensor(kappa).cuda()
        self.beta = torch.tensor(beta).cuda()
        self.const = torch.tensor(0.0, device=device)
        self.gamma = torch.tensor(gamma, device=device)
        self.psi = torch.tensor(psi, device=device)
        # P_hat needs to be symmetric ==>
        # learn vector representing entries in upper/lower triangular matrix and use to populate P_hat later
        self.P_vec_size = int((self.num_nodes * self.num_nodes - self.num_nodes) / 2) + self.num_nodes

        if self.edge_addition:
            self.P_vec = Parameter(torch.FloatTensor(torch.zeros(self.P_vec_size)))
        else:
            self.P_vec = Parameter(torch.FloatTensor(torch.ones(self.P_vec_size)))

        self.reset_parameters()

        self.gc1 = GraphConvolutionPerturb(nfeat, nhid)
        self.gc2 = GraphConvolutionPerturb(nhid, nhid)
        self.gc3 = GraphConvolution(nhid, nout)
        self.lin = nn.Linear(nhid + nhid + nout, nclass)
        self.dropout = dropout

    def reset_parameters(self, eps=10 ** -4):
        # Think more about how to initialize this
        with torch.no_grad():
            if self.edge_addition:
                adj_vec = create_vec_from_symm_matrix(self.adj, self.P_vec_size, device=self.device).cpu().numpy()
                for i in range(len(adj_vec)):
                    if i < 1:
                        adj_vec[i] = adj_vec[i] - eps
                    else:
                        adj_vec[i] = adj_vec[i] + eps
                torch.add(self.P_vec, torch.FloatTensor(adj_vec))  # self.P_vec is all 0s
            else:
                torch.sub(self.P_vec, eps)

    def forward(self, x, sub_adj, logits=True):
        self.sub_adj = sub_adj
        # Same as normalize_adj in utils.py except includes P_hat in A_tilde
        self.P_hat_symm = create_symm_matrix_from_vec(self.P_vec, self.num_nodes)  # Ensure symmetry

        A_tilde = torch.FloatTensor(self.num_nodes, self.num_nodes)
        A_tilde.requires_grad = True

        if self.edge_addition:  # Learn new adj matrix directly
            A_tilde = torch.sigmoid(self.P_hat_symm) + torch.eye(self.num_nodes).cuda()
            # Use sigmoid to bound P_hat in [0,1]
        else:  # Learn P_hat that gets multiplied element-wise with adj -- only edge deletions
            A_tilde = torch.sigmoid(self.P_hat_symm) * self.sub_adj.cuda() + torch.eye(
                self.num_nodes).cuda()  # Use sigmoid to bound P_hat in [0,1]

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

        self.P = (torch.sigmoid(self.P_hat_symm) >= 0.5).float()  # threshold P_hat

        if self.edge_addition:
            A_tilde = self.P + torch.eye(self.num_nodes).cuda()
        else:
            A_tilde = self.P * self.adj + torch.eye(self.num_nodes).cuda()

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

        if self.edge_addition:  # Learn new adj matrix directly
            A_tilde = torch.sigmoid(self.P_hat_symm) + torch.eye(self.num_nodes).cuda()
            # Use sigmoid to bound P_hat in [0,1]
        else:  # Learn P_hat that gets multiplied element-wise with adj -- only edge deletions
            A_tilde = torch.sigmoid(self.P_hat_symm) * self.sub_adj.cuda() + torch.eye(
                self.num_nodes).cuda()  # Use sigmoid to bound P_hat in [0,1]

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
        if self.edge_addition:
            A_tilde = self.P + torch.eye(self.num_nodes).cuda()
        else:
            A_tilde = self.P * self.adj + torch.eye(self.num_nodes).cuda()

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
        pred_same = (y_pred_new_actual == y_pred_orig).float()

        # Need dim >=2 for F.nll_loss to work
        output = output.unsqueeze(0)
        y_pred_orig = y_pred_orig.unsqueeze(0)

        if self.edge_addition:
            cf_adj = self.P
        else:
            cf_adj = self.P * self.adj
        cf_adj.requires_grad = True  # Need to change this otherwise loss_graph_dist has no gradient

        # Want negative in front to maximize loss instead of minimizing it to find CFs
        loss_pred = - F.nll_loss(output, y_pred_orig)
        loss_graph_dist = sum(sum(abs(cf_adj - self.adj.cuda()))) / 2  # Number of edges changed (symmetrical)

        # Zero-out loss_pred with pred_same if prediction flips
        loss_total = pred_same * loss_pred + self.beta * loss_graph_dist
        return loss_total, loss_pred, loss_graph_dist, cf_adj

    def loss_PN_L1_L2(self, output, y_orig_onehot):

        loss_perturb = pertinent_negative_loss(output, y_orig_onehot, self.const, self.kappa)
        if self.edge_addition:
            cf_adj = self.P
        else:
            cf_adj = self.P * self.adj
        cf_adj.requires_grad = True  # Need to change this otherwise loss_graph_dist has no gradient
        loss_graph_dist = torch.dist(cf_adj , self.adj.cuda(), p=1) / 2
        L1 = torch.linalg.norm(self.P_hat_symm, ord=1)
        L2 = torch.linalg.norm(self.P_hat_symm, ord=2)
        loss_total = loss_perturb + self.beta * loss_graph_dist + self.psi*L1 + L2
        return loss_total, loss_perturb, loss_graph_dist, cf_adj

    def loss_PN_AE_L1_L2(self, graph_AE, x, output, y_orig_onehot):

        loss_perturb = pertinent_negative_loss(output, y_orig_onehot, self.const, self.kappa)
        if self.edge_addition:
            cf_adj = self.P
        else:
            cf_adj = self.P * self.adj
        cf_adj.requires_grad = True  # Need to change this otherwise loss_graph_dist has no gradient
        cf_adj_sparse = dense_to_sparse(cf_adj)[0]
        reconst_P = (torch.sigmoid(graph_AE.forward(x, cf_adj_sparse)) >= self.AE_threshold).float()
        l2_AE = torch.dist(cf_adj, reconst_P, p=2)
        loss_graph_dist = torch.dist(cf_adj, self.adj.cuda(), p=1) / 2
        L1 = torch.linalg.norm(self.P_hat_symm, ord=1)
        L2 = torch.linalg.norm(self.P_hat_symm, ord=2)
        loss_total = loss_perturb + self.beta * loss_graph_dist + self.psi*L1 + L2 + self.gamma*l2_AE
        return loss_total, loss_perturb, loss_graph_dist, l2_AE, cf_adj

    def loss_PN_AE_(self, graph_AE, x, output, y_orig_onehot):

        loss_perturb = pertinent_negative_loss(output, y_orig_onehot, self.const, self.kappa)
        if self.edge_addition:
            cf_adj = self.P
        else:
            cf_adj = self.P * self.adj
        cf_adj.requires_grad = True  # Need to change this otherwise loss_graph_dist has no gradient
        cf_adj_sparse = dense_to_sparse(cf_adj)[0]
        reconst_P = (torch.sigmoid(graph_AE.forward(x, cf_adj_sparse)) >= self.AE_threshold).float()
        l2_AE = torch.dist(cf_adj, reconst_P, p=2)
        loss_graph_dist = torch.dist(cf_adj, self.adj.cuda(), p=1) / 2
        loss_total = loss_perturb + self.beta * loss_graph_dist
        return loss_total, loss_perturb, loss_graph_dist, l2_AE, cf_adj

    def loss_PN_AE_pure(self, graph_AE, x, output, y_orig_onehot):
        loss_perturb = pertinent_negative_loss(output, y_orig_onehot, self.const, self.kappa)
        if self.edge_addition:
            cf_adj = self.P
        else:
            cf_adj = self.P * self.adj
        cf_adj.requires_grad = True  # Need to change this otherwise loss_graph_dist has no gradient
        cf_adj_sparse = dense_to_sparse(cf_adj)[0]
        reconst_P = (torch.sigmoid(graph_AE.forward(x, cf_adj_sparse)) >= self.AE_threshold).float()
        l2_AE = torch.dist(reconst_P, cf_adj)/2
        loss_graph_dist = torch.dist(cf_adj, self.adj.cuda(), p=1) / 2
        loss_total = loss_perturb + self.gamma*l2_AE #+ self.beta * loss_graph_dist+ self.gamma*l2_AE
        return loss_total, loss_perturb, loss_graph_dist, l2_AE, cf_adj