import torch
import sys
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.models import InnerProductDecoder, VGAE
from torch_geometric.nn.conv import GCNConv
from torch_geometric.utils import negative_sampling, remove_self_loops, add_self_loops
torch.manual_seed(0)
sys.path.append('/')


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
        self.loss_func = torch.nn.BCEWithLogitsLoss()

    def forward(self, x, edge_index):
        z = self.encode(x, edge_index)
        adj_pred = self.decoder.forward_all(z)
        return adj_pred

    def loss(self, x, edge_index, edge_id, labels):
        z = self.encode(x, edge_index)
        preds = self.decoder(z, edge_index[:, edge_id], sigmoid=False)
        preds = preds.reshape(-1, 1)
        # pos_loss = -torch.log(
        #     self.decoder(z, edge_index, sigmoid=True) + 1e-15).mean()

        # Do not include self-loops in negative samples
        # all_edge_index_tmp, _ = remove_self_loops(all_edge_index)
        # all_edge_index_tmp, _ = add_self_loops(all_edge_index_tmp)

        loss = self.loss_func(preds, labels)
        # neg_edge_index = negative_sampling(all_edge_index_tmp, z.size(0), edge_index.size(1))
        # neg_loss = -torch.log(1 - self.decoder(z, neg_edge_index, sigmoid=True) + 1e-15).mean()

        kl_loss = 1 / x.size(0) * self.kl_loss()

        return loss + kl_loss

    def single_test(self, x, edge_index, edge_id, labels):
        with torch.no_grad():
            z = self.encode(x, edge_index)
            from sklearn.metrics import average_precision_score, roc_auc_score
            preds = self.decoder(z, edge_id, sigmoid=True)
            preds = preds.reshape(-1,1)
            a, b= roc_auc_score(labels.cpu().numpy(), preds.cpu().numpy()), \
            average_precision_score(labels.cpu().numpy(), preds.cpu().numpy())

        return a, b

    def validation_loss(self, x, edge_index, edge_id, labels):
        with torch.no_grad():
            z = self.encode(x, edge_index)
            preds = self.decoder(z, edge_id)
            loss = self.loss_func(preds, labels)
            # pos_loss = -torch.log(
            #     self.decoder(z, val_pos_edge_index, sigmoid=True) + 1e-15).mean()
            # neg_loss = -torch.log(1 - self.decoder(z, val_neg_edge_index, sigmoid=True) + 1e-15).mean()
            kl_loss = 1 / x.size(0) * self.kl_loss()
            return loss + kl_loss