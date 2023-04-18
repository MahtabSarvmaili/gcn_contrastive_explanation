import torch
import sys
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import GCNConv
from torch_geometric.utils import negative_sampling, remove_self_loops, add_self_loops
torch.manual_seed(0)
sys.path.append('/')

def get_link_labels(pos_edge_index, neg_edge_index, device):
    # returns a tensor:
    # [1,1,1,1,...,0,0,0,0,0,..] with the number of ones is equel to the lenght of pos_edge_index
    # and the number of zeros is equal to the length of neg_edge_index
    E = pos_edge_index.size(1) + neg_edge_index.size(1)
    link_labels = torch.zeros(E, dtype=torch.float, device=device)
    link_labels[:pos_edge_index.size(1)] = 1.
    return link_labels


class GAE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, device='cuda'):
        super(GAE, self).__init__()
        self.gc1 = GCNConv(in_channels, hidden_channels)
        self.gc2 = GCNConv(hidden_channels, out_channels)
        self.device = device

    def encode(self, x, edge_index):
        x = self.gc1(x, edge_index)
        x = F.relu(x)
        x = self.gc2(x, edge_index)
        return x

    def decode(self, z, edge_index, sigmoid=False):
        value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
        return torch.sigmoid(value) if sigmoid else value

    def decode_all(self, z, sigmoid=False):
        adj = torch.matmul(z, z.t())
        return torch.sigmoid(adj) if sigmoid else adj

    def __test__(self, z, test_pos_edge_index, test_neg_edge_index):
        from sklearn.metrics import average_precision_score, roc_auc_score

        pos_y = torch.ones(test_pos_edge_index.size(1))
        neg_y = torch.zeros(test_neg_edge_index.size(1))
        y = torch.cat([pos_y, neg_y], dim=0)

        test_edge_index = torch.cat([test_pos_edge_index, test_neg_edge_index], dim=-1)
        pred = self.decode(z, test_edge_index, sigmoid=True)

        y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()

        return roc_auc_score(y, pred), average_precision_score(y, pred)

    def loss(self, x, pos_edge_index):

        # Do not include self-loops in negative samples
        neg_edge_index = negative_sampling(pos_edge_index, x.size(0), pos_edge_index.size(1))
        z = self.encode(x, pos_edge_index)
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1).to('cuda')
        logits = self.decode(z, edge_index)
        y = get_link_labels(pos_edge_index, neg_edge_index, self.device)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        return loss

    def single_test(self, x, train_pos_edge_index, test_pos_edge_index):
        with torch.no_grad():
            z = self.encode(x, train_pos_edge_index)
        test_neg_edge_index = negative_sampling(test_pos_edge_index, x.size(0), test_pos_edge_index.size(1))
        roc_auc_score, average_precision_score = self.__test__(z, test_pos_edge_index, test_neg_edge_index)
        return roc_auc_score, average_precision_score