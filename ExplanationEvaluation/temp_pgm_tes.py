import sys
import traceback
from datetime import datetime
sys.path.insert(0,"../utils.py")
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_dense_adj
from torch_geometric.nn import GCNConv, Linear
from explainers.PGMExplainer import Node_Explainer
import torch_geometric.transforms as T
import torch
import torch.nn.functional as F
import os
import sys
from tqdm import trange

import numpy as np
torch.manual_seed(0)
np.random.seed(0)
sys.path.append('../')


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
    dataset = 'cora'
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
    idx_test = np.arange(0, train_dataset.num_nodes)[train_dataset.test_mask.cpu()]
    idx_test = [int(x) for x in idx_test]
    model = Net(num_features=train_dataset.num_features, num_classes=train_dataset.y.unique().__len__())
    train_model(model, epochs=100, data=train_dataset)
    output = model(train_dataset.x, train_dataset.edge_index)
    ppf = 0
    for node_idx in idx_test[0:10]:
        try:
            explainer = Node_Explainer(model, adj.cpu().numpy(), train_dataset.x.cpu().numpy(), output.argmax(dim=1).cpu().numpy(), 4)
            subnodes, data, stats = explainer.explain(node_idx, num_samples=200, top_node=5, pred_threshold=0.2)
            pgm_explanation = explainer.pgm_generate(node_idx, data, stats, subnodes)
            print("PGM Nodes: ", pgm_explanation.nodes())
            print("PGM Edges: ", pgm_explanation.edges())

            a = []
            for e in pgm_explanation.edges():
                a.append([int(e[0]), int(e[1])])
            b = np.array(a).T
            b = torch.tensor(b, dtype=torch.int64)
            expl_lb = model.forward(train_dataset.x, b)
            expl_lb = expl_lb[b.reshape(-1).unique()].argmax(dim=1)
            sb_lb = train_dataset.y[b.reshape(-1).unique()]
            ppf = (expl_lb == sb_lb).sum() / sb_lb.__len__()
            ppf = ppf.item()
            print(ppf)
            print('test')
        except:
            traceback.print_exc()
            continue
    return ppf


if __name__ == '__main__':
    # c = os.path.dirname(os.path.realpath(__file__))
    # p = os.path.dirname(c)
    # sys.path.insert(0, p)
    # from evaluation.evaluation_metrics import insertion
    t = main()
    print(t)