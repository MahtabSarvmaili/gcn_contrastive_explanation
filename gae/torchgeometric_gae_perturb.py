import numpy as np
import pylab as p
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor, fill_diag, matmul, mul
from torch_sparse import sum as sparsesum
from torch.nn.parameter import Parameter
from torch_geometric.utils import dense_to_sparse
from torch_geometric.nn import inits
from utils import get_degree_matrix, create_symm_matrix_from_vec, create_vec_from_symm_matrix
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.utils import add_remaining_self_loops
from torch_scatter import scatter_add
from gae.utils import preprocess_graph
# torch.manual_seed(0)
# np.random.seed(0)
from layers import GraphConvolution


def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, flow="source_to_target", dtype=None):

    fill_value = 1.

    assert flow in ["source_to_target", "target_to_source"]
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                 device=edge_index.device)

    if add_self_loops:
        edge_index, tmp_edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)
        assert tmp_edge_weight is not None
        edge_weight = tmp_edge_weight
    edge_weight = edge_weight.to(edge_index.device)
    row, col = edge_index[0], edge_index[1]
    idx = col if flow == "source_to_target" else row
    deg = scatter_add(edge_weight, idx, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
    return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]



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


class GraphConvolutionPerturb(MessagePassing):

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolutionPerturb, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        # initializing parameters using normal distribution
        # initializing bias as vector of zeros
        if bias is not None:
            self.bias = Parameter(torch.FloatTensor(np.zeros(out_features)))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.weight, 0.0, 0.001)

    def forward(self, x, edge_index, edge_weight):

        support = torch.mm(x, self.weight)
        out = self.propagate(edge_index, x=support, edge_weight=edge_weight,
                             size=None)
        if self.bias is not None:
            out += self.bias
        return out

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCNSyntheticPerturb(nn.Module):
    """
    3-layer GCN used in GNN Explainer synthetic tasks
    """

    def __init__(
            self, nfeat, nhid, nout, edge_index, num_nodes, beta=0.1,
            cf_expl=True, gamma=0.09, kappa=10, psi=0.01, device='cuda'
    ):
        # the best gamma and psi for prototype explanation are gamma=0.01, kappa=10, psi=0.09
        # the best gamma and psi for CF explanation are gamma=0.09, kappa=10, psi=0.01

        super(GCNSyntheticPerturb, self).__init__()
        self.edge_index = edge_index
        self.num_nodes = num_nodes
        self.beta = beta
        self.device = device
        self.kappa = torch.tensor(kappa).cuda()
        self.beta = torch.tensor(beta).cuda()
        self.const = torch.tensor(0.0, device=device)
        self.gamma = torch.tensor(gamma, device=device)
        self.psi_l1 = torch.tensor(psi, device=device)
        self.cf_expl = cf_expl
        # P_hat needs to be symmetric ==>
        # learn vector representing entries in upper/lower triangular matrix and use to populate P_hat later
        self.P_vec_size = (edge_index.size(1))
        # Initializing P_vec as vector of zeros
        self.P_vec = Parameter(torch.FloatTensor(torch.zeros((self.P_vec_size,))))
        # self.reset_parameters()
        self.conv1 = GraphConvolutionPerturb(nfeat, nhid)
        self.conv2 = GraphConvolutionPerturb(nhid, nout)

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

    # def reset_parameters(self, eps=10 ** -4):
    #     # Think more about how to initialize this
    #     # nn.init.uniform_(self.P_vec, 0.0, 0.001)

    def encode(self, x):
        edge_index, edge_weight = gcn_norm(self.edge_index, self.P_vec.sigmoid(), self.num_nodes)
        # edge_index, edge_weight = gcn_norm(self.edge_index, self.P_vec, self.num_nodes)
        x = self.conv1(x, edge_index, edge_weight)
        x = x.relu()
        x = self.conv2(x, edge_index, edge_weight)
        return x

    def encode_prediction(self, x):
        # Same as forward but uses P instead of P_hat ==> non-differentiable
        # but needed for actual predictions
        self.P = (self.P_vec.sigmoid() >= 0.5).float()

        edge_index, edge_weight = gcn_norm(self.edge_index, self.P, self.num_nodes)
        # edge_index, edge_weight = gcn_norm(self.edge_index, self.P_vec, self.num_nodes)
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_weight)
        return x

    def decode(self, z, test_edge_index): # only pos and neg edges
        logits = (z[test_edge_index[0]] * z[test_edge_index[1]]).sum(dim=-1)  # dot product
        return logits

    def decode_all(self, z):
        prob_adj = z @ z.t() # get adj NxN
        return (prob_adj > 0).nonzero(as_tuple=False).t() # get predicted edge_list


    def loss(self, x, test_edge_index, link_label):

        z = self.encode(x)  # encode
        link_logits = self.decode(z, test_edge_index)  # decode
        loss = F.binary_cross_entropy_with_logits(link_logits, link_label)
        # PLoss = 0
        # pred_same = (y_pred_new_actual == y_pred_orig).float()
        #
        # # Need dim >=2 for F.nll_loss to work
        # output = output.unsqueeze(0)
        # y_pred_orig = y_pred_orig.unsqueeze(0)
        # cf_adj = self.P * self.adj
        # cf_adj.requires_grad = True  # Need to change this otherwise loss_graph_dist has no gradient
        #
        # # Want negative in front to maximize loss instead of minimizing it to find CFs
        # loss_pred = - F.nll_loss(output, y_pred_orig)
        # loss_graph_dist = sum(sum(abs(cf_adj - self.adj.cuda()))) / 2  # Number of edges changed (symmetrical)
        #
        # # Zero-out loss_pred with pred_same if prediction flips
        # loss_total = pred_same * loss_pred + self.beta * loss_graph_dist
        return loss

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