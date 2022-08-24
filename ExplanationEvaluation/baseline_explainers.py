import sys
sys.path.append('../utils.py')
from torch_geometric.nn import GCNConv, GNNExplainer, Linear
from explainers.PGExplainer import PGExplainer
import torch
from utils.graph_utils import normalize_adj, get_neighbourhood
import numpy as np
torch.manual_seed(0)
np.random.seed(0)


class gnn_explainer:

    def __init__(self, model, train_dataset, epochs = 200):
        self.data = train_dataset
        self.model = model
        self.explainer = GNNExplainer(self.model, epochs=epochs)

    def explain_node(self, node_idx):
        output = self.model(self.data.x, self.data.edge_index)

        sub_adj, sub_feat, sub_labels, node_dict, sub_edge_index = get_neighbourhood(
            node_idx, self.data.edge_index, 3 + 1, self.data.x, output.argmax(dim=1))
        new_idx = int(node_dict[node_idx])

        node_feat_mask, edge_mask = self.explainer.explain_node(new_idx, sub_feat, sub_edge_index)

        labels = self.model(sub_feat, sub_edge_index[:, (edge_mask >= 0.5)], self.data).argmax(dim=1)
        return node_feat_mask, edge_mask, labels


class pg_explainer:

    def __init__(self, model, train_dataset, epochs=30):

        self.model = model
        self.data = train_dataset

    def explain_node(self, node_idx, node_indices):

        output = self.model(self.data.x, self.data.edge_index)

        sub_adj, sub_feat, sub_labels, node_dict, sub_edge_index = get_neighbourhood(
            node_idx, self.data.edge_index, 3 + 1, self.data.x, output.argmax(dim=1))
        new_idx = int(node_dict[node_idx])
        self.pgexplainer = PGExplainer(self.model, sub_edge_index, sub_feat, 'node')
        node_feat_mask, edge_mask = self.pgexplainer.explain(new_idx)

        labels = self.model(sub_feat, sub_edge_index[:, (edge_mask >= 0.5)], self.data).argmax(dim=1)
        return node_feat_mask, edge_mask, labels

