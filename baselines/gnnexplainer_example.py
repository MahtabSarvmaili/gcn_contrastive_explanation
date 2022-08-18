from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv, GNNExplainer, Linear
import torch_geometric.transforms as T
import torch
import torch.nn.functional as F
import os
import pandas as pd
from tqdm import trange
from torch_geometric.utils import to_dense_adj
from utils import normalize_adj, get_neighbourhood
import numpy as np


def deletion(model, x, edge_index, edge_mask, labels, node_idx, device='cuda', name='name'):
    p_c = []
    size = [0.1, 0.3, 0.5, 0.7, 0.9, 1]
    a = edge_index[:, (edge_mask >= 0.5)]
    for s in size:
        b = np.random.choice(a.size(1), size=int(s * a.size(1)), replace=False)
        labels_ = model(x, a[:, b]).argmax(dim=1)
        changed = labels_ != labels
        percent = (1*changed).sum()/len(labels)
        node_label_change = changed[node_idx]
        print(f"the percentage of changed labels:{percent}")
        print(f"Node label changed:{node_label_change}")
        p_c.append([percent, node_label_change])
    df = pd.DataFrame(p_c,
                      columns=['Percent', 'changed'])
    df.to_csv(f'{name}.csv', index=False)


class Net(torch.nn.Module):
    def __init__(self, num_features, dim=20, num_classes=1):
        super(Net, self).__init__()
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


def test(model, data):
    model.eval()
    logits, accs = model(data.x, data.edge_index, data), []
    for _, mask in data('train_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs


def train_model(model, epochs, data):
    loss = 999.0
    train_acc = 0.0
    test_acc = 0.0
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-3)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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


class gnn_example:

    def __init__(self):
        #Load the dataset
        dataset = 'cora'
        path = os.path.join(os.getcwd(), 'data', 'Planetoid')
        train_dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())

        # Since the dataset is comprised of a single huge graph, we extract that graph by indexing 0.
        self.data = train_dataset[0]

        self.data.train_mask = torch.zeros(self.data.num_nodes, dtype=torch.bool)
        self.data.train_mask[:self.data.num_nodes - 1000] = 1
        self.data.val_mask = None
        self.data.test_mask = torch.zeros(self.data.num_nodes, dtype=torch.bool)
        self.data.test_mask[self.data.num_nodes - 500:] = 1
        epochs = 200

        model = Net(num_features=train_dataset.num_features, num_classes=train_dataset.num_classes)
        self.model = train_model(model, epochs, self.data)

    def explain_node(self, node_idx):
        output = self.model(self.data.x, self.data.edge_index)
        explainer = GNNExplainer(self.model, epochs=200)
        sub_adj, sub_feat, sub_labels, node_dict, sub_edge_index = get_neighbourhood(
            node_idx, self.data.edge_index, 3 + 1, self.data.x, output.argmax(dim=1))
        new_idx = int(node_dict[node_idx])

        node_feat_mask, edge_mask = explainer.explain_node(new_idx, sub_feat, sub_edge_index)
        labels = self.model(sub_feat, sub_edge_index[:,(edge_mask>=0.5)], self.data).argmax(dim=1)
        return node_feat_mask, edge_mask, labels


gnnexplain = gnn_example()
gnnexplain.explain_node(4)