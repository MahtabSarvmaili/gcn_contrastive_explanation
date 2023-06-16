import numpy as np
import pylab as p
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.typing import Adj, OptTensor, PairTensor
from torch.nn.parameter import Parameter
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import dense_to_sparse
from torch_geometric.nn import inits
from utils import get_degree_matrix, create_symm_matrix_from_vec, create_vec_from_symm_matrix
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.utils import add_remaining_self_loops, add_self_loops, degree
from torch_scatter import scatter_add, scatter_sum
# torch.manual_seed(0)
# np.random.seed(0)
from layers import GraphConvolution


def gcn_norm(edge_index, P_vec=None, num_nodes=None, flow="source_to_target"):

    fill_value = 1.

    assert flow in ["source_to_target", "target_to_source"]
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    edge_index, tmp_edge_weight = add_remaining_self_loops(
        edge_index, P_vec, fill_value, num_nodes)

    edge_weight = tmp_edge_weight.to(edge_index.device)
    row, col = edge_index[0], edge_index[1]
    idx = col if flow == "source_to_target" else row
    deg = scatter_add(edge_weight, idx, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
    return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]


def pertinent_negative_positive_loss(output, y_orig_onehot, const, kappa, positive):
    target_lab_score = (output * y_orig_onehot).sum(dim=0)
    max_nontarget_lab_score = (
            (1 - y_orig_onehot) * output -
            (y_orig_onehot * 10000)).max(dim=0).values
    if positive:
        loss_perturb = torch.max(const, max_nontarget_lab_score - target_lab_score + kappa)
    else:
        loss_perturb = torch.max(const, -max_nontarget_lab_score + target_lab_score + kappa)
    return loss_perturb, max_nontarget_lab_score


def cross_loss(output, y):
    cross_loss = torch.nn.CrossEntropyLoss()
    closs = cross_loss(output, y)
    return closs


class GraphConvolutionPerturb(MessagePassing):
    def __init__(self, in_channels, out_channels, edge_index_size, bias: bool = True, **kwargs):

        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        super(GraphConvolutionPerturb, self).__init__(**kwargs)
        self.in_features = in_channels
        self.out_features = out_channels
        self.lin = torch.nn.Linear(in_channels, out_channels, bias=False)
        self.num_nodes = edge_index_size
        # Initializing P_vec as vector of zeros

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        # self.reset_parameters()

    def forward(self, x: Tensor, edge_index: Adj, P_vec: Tensor = None) -> Tensor:
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        # Step 1: Add self-loops to the adjacency matrix.
        # _, edge_weight = gcn_norm(edge_index, self.P_vec, self.num_nodes)

        # Step 2: Linearly transform node feature matrix.
        edge_index, edge_weight = gcn_norm(edge_index, P_vec, x.size(self.node_dim))
        x = self.lin(x)
        # Step 3-5: Start propagating messages.
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=None)
        if self.bias is not None:
            out += self.bias
        return out

    def message(self, x_j, edge_weight):
        # x_j has shape [E, out_channels]
        out = edge_weight.view(-1, 1) * x_j
        return out


class GCNPerturb(nn.Module):
    """
    3-layer GCN used in GNN Explainer synthetic tasks
    """

    def __init__(
            self, n_features, n_hidden, n_classes, edge_index, beta=0.1,
            cf_expl=True, gamma=0.09, kappa=10, psi_l1=0.01, psi_l2=0.1, device='cuda'
    ):
        # the best gamma and psi for prototype explanation are gamma=0.01, kappa=10, psi=0.09
        # the best gamma and psi for CF explanation are gamma=0.09, kappa=10, psi=0.01

        super(GCNPerturb, self).__init__()
        self.edge_index = edge_index
        self.beta = beta
        self.device = device
        self.kappa = torch.tensor(kappa).cuda()
        self.beta = torch.tensor(beta).cuda()
        self.const = torch.tensor(0.0, device=device)
        self.gamma = torch.tensor(gamma, device=device)
        self.psi_l1 = torch.tensor(psi_l1, device=device)
        self.psi_l2 = torch.tensor(psi_l2, device=device)
        self.cf_expl = cf_expl

        self.conv1 = GraphConvolutionPerturb(
            n_features, n_hidden, edge_index_size=edge_index.size(1)
        )
        self.conv2 = GraphConvolutionPerturb(
            n_hidden, n_hidden, edge_index_size=edge_index.size(1)
        )
        self.conv3 = GraphConvolutionPerturb(
            n_hidden, n_hidden, edge_index_size=edge_index.size(1)
        )
        self.lin = nn.Linear(n_hidden, n_classes)
        self.P_vec = Parameter(torch.FloatTensor(torch.ones((edge_index.size(1),))))
        self.loss_func = torch.nn.CrossEntropyLoss()

    def __L1__(self):
        return torch.linalg.norm(self.P_vec, ord=1)

    def __L2__(self):
        return torch.linalg.norm(self.P_vec, ord=2)

    def __loss_graph_dist__(self, cf_adj):
        return torch.dist(cf_adj , self.adj.cuda(), p=1) / 2

    def forward(self, x, edge_index, batch):

        x = self.conv1(x, edge_index, self.P_vec.sigmoid())
        x = x.relu()
        x = self.conv2(x, edge_index, self.P_vec.sigmoid())
        x = x.relu()
        x = self.conv3(x, edge_index, self.P_vec.sigmoid())

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        return x

    def forward_prediction(self, x, edge_index, batch):
        # Same as forward but uses P instead of P_hat ==> non-differentiable
        # but needed for actual predictions
        P_vec = (self.P_vec.sigmoid() > 0.5).float()
        x = self.conv1(x, edge_index, P_vec)
        x = x.relu()
        x = self.conv2(x, edge_index, P_vec)
        x = x.relu()
        x = self.conv3(x, edge_index, P_vec)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        return x

    def get_explanation(self, x, edge_index, batch):
        P_vec = (self.P_vec.sigmoid() > 0.5)
        expl = edge_index[:, P_vec]
        preds = self.forward_prediction(x, edge_index, batch).argmax(dim=1)
        return expl, preds

    def loss(self, x, edge_index, batch, y, l1=1, l2=1, ae=1, dist=1):
        out = self.forward(x, edge_index, batch)
        loss = self.loss_func(out, y)  # Compute the loss.
        L1 = self.__L1__()
        L2 = self.__L2__()
        loss_total = loss + self.psi_l2* l2 * L2 + l1 * self.psi_l1 * L1
        pred_labels = self.forward_prediction(x, edge_index, batch).argmax(dim=1)
        return loss_total, pred_labels

    def single_test(self, x, test_pos_edge_index):

        with torch.no_grad():
            z = self.encode(x)
            pred = self.decode(z, test_pos_edge_index)
            pred = torch.sigmoid(pred)
        return pred