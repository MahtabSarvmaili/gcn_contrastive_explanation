import sys
import traceback
from datetime import datetime
sys.path.insert(0,"../utils.py")
from torch_geometric.datasets import Planetoid, BAShapes
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from torch_geometric.nn import GCNConv, GNNExplainer, Explainer, Linear
from evaluation.evaluation_metrics import gen_graph
from networkx import double_edge_swap, adjacency_matrix
from data.data_loader import load_data, load_synthetic, load_synthetic_AE, load_data_AE
from data.gengraph import gen_syn1, gen_syn2, gen_syn3, gen_syn4
from explainers.PGExplainer import PGExplainer
from baseline_utils.graph_utils import normalize_adj, get_neighbourhood
from visualization import plot_graph
import torch_geometric.transforms as T
import torch
import torch.nn.functional as F
import os
import sys
import pandas as pd
from tqdm import trange

import numpy as np
torch.manual_seed(0)
np.random.seed(0)
sys.path.append('../../')


class Net(torch.nn.Module):
    def __init__(self, num_features, dim=20, num_classes=1):
        super(Net, self).__init__()
        self.embedding_size = 20

        self.conv1 = GCNConv(num_features, dim)
        self.conv2 = GCNConv(dim, dim)
        self.conv3 = GCNConv(dim, dim)
        self.lin = Linear(3 * dim, num_classes)

    def forward(self, x, edge_index, data=None):
        x1 = F.relu(self.conv1(x, edge_index))
        x1 = F.dropout(x1, training=self.training)
        x2 = F.relu(self.conv2(x1, edge_index))
        x2 = F.dropout(x2, training=self.training)
        x3 = self.conv3(x2, edge_index)
        x = self.lin(torch.cat((x1, x2, x3), dim=1))

        return F.log_softmax(x, dim=1)

    def embedding(self, x, edge_index):
        stack = []
        x1 = F.relu(self.conv1(x, edge_index))
        x1 = F.dropout(x1, training=self.training)
        stack.append(x1)
        x2 = F.relu(self.conv2(x1, edge_index))
        x2 = F.dropout(x2, training=self.training)
        stack.append(x2)

        x3 = self.conv3(x2, edge_index)
        return x3


def test(model, data):
    model.eval()
    logits, accs = model(data.x, data.edge_index, data), []
    for _, mask in data('train_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    print(f"Train accuracy {accs[0]}, Test Accuracy {accs[1]}")
    return accs


def train_model(model, epochs, data):
    loss = 999.0
    train_acc = 0.0
    test_acc = 0.0
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
    device = torch.device('cpu')
    t = trange(epochs, desc="Stats: ", position=0)
    model = model.to(device)
    for epoch in t:
        model.train()

        loss = 0

        data = data.to(device)
        optimizer.zero_grad()
        log_logits = model(data.x, data.edge_index, data)

        # Since the data is a single huge graph, training on the training set is done by masking the nodes that are not in the training set.
        loss = F.nll_loss(log_logits[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        # validate
        train_acc, test_acc = test(model, data)
        train_loss = loss

        t.set_description(
            '[Train_loss:{:.6f} Train_acc: {:.4f}, Test_acc: {:.4f}]'.format(loss, train_acc, test_acc))
    return model


def main():
    s = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    device = 'cpu'
    dataset = 'pubmed'
    path = os.path.join(os.getcwd(), 'data', 'Planetoid')
    transformer = T.Compose([
        T.NormalizeFeatures(),
        T.ToDevice(device),
        T.RandomNodeSplit(num_val=0.1, num_test=0.2),
    ])
    train_dataset = Planetoid(path, dataset, transform=transformer)[0]
    # train_dataset = load_synthetic(gen_syn1, device)['dataset']
    adj = to_dense_adj(train_dataset.edge_index).squeeze(dim=0)
    # train_dataset.edge_index = dense_to_sparse(adj_norm)
    idx_train = np.arange(0, train_dataset.num_nodes)[train_dataset.train_mask.cpu()]
    idx_train = np.random.choice(idx_train,40)
    idx_test = np.arange(0, train_dataset.num_nodes)[train_dataset.test_mask.cpu()]

    idx_test = [int(x) for x in idx_test]
    idx_test = idx_test[::-1]
    model = Net(num_features=train_dataset.num_features, num_classes=train_dataset.y.unique().__len__())
    train_model(model, epochs=100, data=train_dataset)
    output = model(train_dataset.x, train_dataset.edge_index)
    gnnexplainer = GNNExplainer(model, num_hops=4, epochs=100)
    ppf = 0
    for node_idx in idx_test[-5:]:
        try:
            sub_adj, sub_feat, sub_labels, node_dict, sub_edge_index = get_neighbourhood(
                int(node_idx), train_dataset.edge_index, 4, train_dataset.x, output.argmax(dim=1))
            new_idx = node_dict[int(node_idx)]
            pgexplainer = PGExplainer(model, train_dataset.edge_index, train_dataset.x, 'node')
            pgexplainer.prepare(idx_test)
            graph, expl, out, filter_edges, filter_nodes, filter_labels = pgexplainer.explain(node_idx)

            # pgexplainer = PGExplainer(model, sub_edge_index, sub_feat, 'node')
            # pgexplainer.prepare(None)
            # graph, expl, out, filter_edges, filter_nodes, filter_labels = pgexplainer.explain(new_idx)

            filter_edges = torch.tensor(filter_edges)
            pg_expl = to_dense_adj(filter_edges.T, max_num_nodes=adj.shape[0]).squeeze(dim=0)
            # pg_expl = to_dense_adj(filter_edges.T, max_num_nodes=sub_adj.shape[0]).squeeze(dim=0)
            sb_lb = train_dataset.y[filter_nodes]
            # sb_lb = sub_labels[filter_nodes]

            pg_ppf = (filter_labels == sb_lb).sum() / sb_lb.__len__()
            pg_ppf11 = pg_ppf.item()
            p = expl.sum().detach().cpu().numpy()
            s = (pg_expl < adj).sum() / adj.sum()
            s = (pg_expl < sub_adj).sum() / sub_adj.sum()
            stab = 0

            nodes = list(range(train_dataset.num_nodes))
            g = gen_graph(nodes, train_dataset.edge_index.cpu().t().numpy())
            fids = []
            for _ in range(5):
                g_p = double_edge_swap(g, nswap=1)
                edge_idx_p = dense_to_sparse(torch.FloatTensor(adjacency_matrix(g_p, nodelist=nodes).todense()))[0]
                pgexplainer = PGExplainer(model, edge_idx_p, train_dataset.x, 'node')
                pgexplainer.prepare(idx_test)
                graph, expl, out, filter_edges, filter_nodes, filter_labels = pgexplainer.explain(node_idx)
                sb_lb = train_dataset.y[filter_nodes]

                pg_ppf = (filter_labels == sb_lb).sum() / sb_lb.__len__()
                fids.append(pg_ppf)

            fids = np.array(fids)
            fid_diff_actual_perturb = np.abs(pg_ppf - fids)
            stab = fid_diff_actual_perturb.mean()
            print(f'PGExplainer: node {node_idx}, Fidelity: {pg_ppf11} - Sparsity: {s}- #edges: {pg_expl.sum()}, Stability: {stab}')

            node_feat_mask, gnnexpl_edge_mask, out = gnnexplainer.explain_node(node_idx, train_dataset.x, train_dataset.edge_index)
            gnnexpl_edge_mask = gnnexpl_edge_mask>0.5
            masked_edge_idx = train_dataset.edge_index[:, gnnexpl_edge_mask].T
            a = []
            for x in masked_edge_idx:
                a.append([node_dict[x[0].item()], node_dict[x[1].item()]])
            b = np.array(a).T
            b = torch.tensor(b, dtype=torch.int64)

            expl_labels = out.argmax(dim=1)[b.reshape(-1)]
            expl = to_dense_adj(masked_edge_idx.T, max_num_nodes=adj.shape[0]).squeeze(dim=0)
            # print(f'cf_adj present edges: {expl.sum()}')
            sb_lb = train_dataset.y[masked_edge_idx.reshape(-1)]

            ppf = (expl_labels == sb_lb).sum() / sb_lb.__len__()
            ppf = ppf.item()
            s = (expl < adj).sum() / adj.sum()
            s = s.cpu().numpy()
            p = expl.sum().cpu().numpy()

            # Stability
            nodes = list(range(train_dataset.num_nodes))
            g = gen_graph(nodes, train_dataset.edge_index.cpu().t().numpy())
            fids = []
            for _ in range(5):
                g_p = double_edge_swap(g, nswap=1)
                edge_idx_p = dense_to_sparse(torch.FloatTensor(adjacency_matrix(g_p, nodelist=nodes).todense()))[0]
                _, mask_p, out = gnnexplainer.explain_node(node_idx, train_dataset.x, edge_idx_p)
                edge_mask_p = mask_p > 0.5
                masked_edge_idx_p = train_dataset.edge_index[:, edge_mask_p].T

                a = []
                for x in masked_edge_idx:
                    a.append([node_dict[x[0].item()], node_dict[x[1].item()]])
                b = np.array(a).T
                b = torch.tensor(b, dtype=torch.int64)

                expl_labels_p = out.argmax(dim=1)[b.reshape(-1)]

                f = (expl_labels_p == output[b.reshape(-1)].argmax(dim=1)).sum() / expl_labels_p.__len__()
                f = f.cpu().numpy()
                fids.append(f)

            fids = np.array(fids)
            fid_diff_actual_perturb = np.abs(ppf - fids)
            stab = fid_diff_actual_perturb.mean()

            print(f'\nGNNExplainer: node {node_idx}, Fidelity: {ppf} - Sparsity: {s}- #edges: {p}, Stability: {stab}')
            # plt_graph = plot_graph(
            #     sub_adj,
            #     new_idx,
            #     f'./results/{dataset}/_{node_idx}_sub_adj_{s}.png'
            # )
            # plt_graph.plot_org_graph(
            #     expl,
            #     expl_labels.numpy(),
            #     new_idx,
            #     f'./results/{dataset}/_{node_idx}_masked_sub_adj_subgraph_fidelity_subgraph_fidelity_{expl.sum()}__{ppf}__{s}.png',
            #     plot_grey_edges=False
            # )
            print('test')
        except:
            traceback.print_exc()
            continue
    return


if __name__ == '__main__':
    # c = os.path.dirname(os.path.realpath(__file__))
    # p = os.path.dirname(c)
    # sys.path.insert(0, p)
    # from evaluation.evaluation_metrics import insertion
    t = main()