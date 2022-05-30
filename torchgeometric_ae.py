import torch
import sys
import torch.nn as nn
import torch.nn.functional as F
sys.path.append('../..')

from torch_geometric.nn.models import InnerProductDecoder, VGAE
from torch_geometric.nn.conv import GCNConv
from torch_geometric.utils import negative_sampling, remove_self_loops, add_self_loops
from data.data_loader import load_synthetic_AE
from data.gengraph import gen_syn4
from torch_geometric.utils import dense_to_sparse
class GCNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCNEncoder, self).__init__()
        self.gcn_shared = GCNConv(in_channels, hidden_channels)
        self.gcn_mu = GCNConv(hidden_channels, out_channels)
        self.gcn_logvar = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.gcn_shared(x, edge_index))
        mu = self.gcn_mu(x, edge_index)
        logvar = self.gcn_logvar(x, edge_index)
        return mu, logvar


class DeepVGAE(VGAE):
    def __init__(self, enc_in_channels, enc_hidden_channels, enc_out_channels):
        super(DeepVGAE, self).__init__(encoder=GCNEncoder(enc_in_channels,
                                                          enc_hidden_channels,
                                                          enc_out_channels),
                                       decoder=InnerProductDecoder())

    def forward(self, x, edge_index):
        z = self.encode(x, edge_index)
        adj_pred = self.decoder.forward_all(z)
        return adj_pred

    def loss(self, x, pos_edge_index, all_edge_index):
        z = self.encode(x, pos_edge_index)

        pos_loss = -torch.log(
            self.decoder(z, pos_edge_index, sigmoid=True) + 1e-15).mean()

        # Do not include self-loops in negative samples
        all_edge_index_tmp, _ = remove_self_loops(all_edge_index)
        all_edge_index_tmp, _ = add_self_loops(all_edge_index_tmp)

        neg_edge_index = negative_sampling(all_edge_index_tmp, z.size(0), pos_edge_index.size(1))
        neg_loss = -torch.log(1 - self.decoder(z, neg_edge_index, sigmoid=True) + 1e-15).mean()

        kl_loss = 1 / x.size(0) * self.kl_loss()

        return pos_loss + neg_loss + kl_loss

    def single_test(self, x, train_pos_edge_index, test_pos_edge_index, test_neg_edge_index):
        with torch.no_grad():
            z = self.encode(x, train_pos_edge_index)
        roc_auc_score, average_precision_score = self.test(z, test_pos_edge_index, test_neg_edge_index)
        return roc_auc_score, average_precision_score


import os

import torch
import torch.nn as nn
from torch.optim import Adam

from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.utils import train_test_split_edges

torch.manual_seed(12345)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

enc_in_channels = 871
enc_hidden_channels = 1
enc_out_channels = 16

lr = 0.01
epoch = 400

model = DeepVGAE(enc_in_channels, enc_hidden_channels, enc_out_channels).to(device)
optimizer = Adam(model.parameters(), lr=lr)
data = load_synthetic_AE(gen_syn4, device=device)
pos_edge = dense_to_sparse(data['train_adj'])
for epoch in range(epoch):
    model.train()
    optimizer.zero_grad()

    loss = model.loss(data['features'], data['edge_index_'], torch.FloatTensor(data['edge_index'].transpose()).cuda())
    loss.backward()
    optimizer.step()
    if epoch % 2 == 0:
        model.eval()
        roc_auc, ap = model.single_test(data.x,
                                        data.train_pos_edge_index,
                                        data.test_pos_edge_index,
                                        data.test_neg_edge_index)
        print("Epoch {} - Loss: {} ROC_AUC: {} Precision: {}".format(epoch, loss.cpu().item(), roc_auc, ap))

