from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv, GNNExplainer, Linear
import torch_geometric.transforms as T
import torch
import torch.nn.functional as F
import os
import pandas as pd
from tqdm import trange
from torch_geometric.utils import to_dense_adj
import numpy as np


def deletion(model, x, edge_index, edge_mask, labels, node_idx, device='cuda', name='name'):
    p_c = []
    size = [0.1, 0.3, 0.5, 0.7, 0.9, 1]
    a = edge_index[:, (edge_mask >= 0.5)]
    for s in size:
        b = np.random.choice(a.size(1), size=int(s * a.size(1)), replace=False)
        labels_ = model(x, a[:, b], data).argmax(dim=1)
        changed = labels_ != labels
        percent = (1*changed).sum()/len(labels)
        node_label_change = changed[node_idx]
        print(f"the percentage of changed labels:{percent}")
        print(f"Node label changed:{node_label_change}")
        p_c.append([percent, node_label_change])
    df = pd.DataFrame(p_c,
                      columns=['Percent', 'changed'])
    df.to_csv(f'{name}.csv', index=False)


#Load the dataset
dataset = 'cora'
path = os.path.join(os.getcwd(), 'data', 'Planetoid')
train_dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())

# Since the dataset is comprised of a single huge graph, we extract that graph by indexing 0.
data = train_dataset[0]

data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
data.train_mask[:data.num_nodes - 1000] = 1
data.val_mask = None
data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
data.test_mask[data.num_nodes - 500:] = 1


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


epochs = 200
dim = 16
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net(num_features=train_dataset.num_features, dim=dim, num_classes=train_dataset.num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-3)


def test(model, data):
    model.eval()
    logits, accs = model(data.x, data.edge_index, data), []
    for _, mask in data('train_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs


loss = 999.0
train_acc = 0.0
test_acc = 0.0

t = trange(epochs, desc="Stats: ", position=0)

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

    t.set_description('[Train_loss:{:.6f} Train_acc: {:.4f}, Test_acc: {:.4f}]'.format(loss, train_acc, test_acc))

idx_test = np.arange(0, data.num_nodes)[data.test_mask.cpu()]
x, edge_index = data.x, data.edge_index
adj = to_dense_adj(edge_index).squeeze(0)
explainer = GNNExplainer(model, epochs=200)
for node_idx in idx_test:
    node_idx = int(node_idx)
    node_feat_mask, edge_mask = explainer.explain_node(node_idx, x, edge_index)
    edge_mask_ = to_dense_adj(edge_index[:,(edge_mask>=0.5)], max_num_nodes=x.size(0)).squeeze(0)
    # cf_adj = edge_mask_ * adj
    labels = model(x, edge_index[:,(edge_mask>=0.5)], data).argmax(dim=1)

    deletion(
        model,
        x,
        edge_index,
        edge_mask,
        labels,
        node_idx,
        name=f'cora__insertion__',
    )
