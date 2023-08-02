import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from utils import transform_address
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
np.random.seed(0)
random.seed(0)

class PlotGraphExplanation:
    def __init__(self, edge_index, labels, num_nodes, list_classes, exp_type, dataset_str):
        np.random.seed(0)
        self.list_classes = list_classes
        self.num_nodes = num_nodes
        self.expl_type = exp_type
        self.dataset = dataset_str
        self.title = {
            'CF': "Counterfactual Explanation",
            'PT': 'Prototype Explanation',
            'EXE': 'Exemplar Explanation'
        }
        self.edge_list = [(i[0], i[1]) for i in edge_index.t().cpu().numpy()]

        self.G = nx.Graph()
        self.G.add_nodes_from(list(range(self.num_nodes)))
        self.G.add_edges_from(self.edge_list)
        self.pos = nx.spring_layout(self.G)

        self.label2nodes = {}
        for i in self.list_classes:
            self.label2nodes[i]=[]
        for i in range(self.num_nodes):
            self.label2nodes[labels[i]].append(i)

        self.colors = ["#" + ''.join([random.choice('0123456789ABCDEF') for i in range(6)])
                       for j in range(len(self.list_classes))]

    def plot_del_edges(self, explanations, res_dir, dt_id, plt_title=None):

        for j, exp_edge_index in enumerate(explanations):
            pos_edges = [(u, v) for (u, v) in exp_edge_index.t().cpu().numpy()]
            removed_edges = [x for x in self.edge_list if x not in pos_edges]
            plt.close()
            for hh in self.list_classes:
                node_filter = []
                for zz in self.label2nodes[hh]:
                    if zz in self.G.nodes():
                        node_filter.append(zz)
                nx.draw_networkx_nodes(self.G, self.pos,
                                       nodelist=node_filter,
                                       node_color=self.colors[hh % len(self.colors)],
                                       node_size=50, label=str(hh))

            nx.draw_networkx_edges(self.G, self.pos, edgelist=pos_edges,
                                   width=2, alpha=1, edge_color='black')

            nx.draw_networkx_edges(self.G, self.pos, edgelist=removed_edges,
                                   width=2, alpha=1, edge_color='red')
            plt.legend()
            if plt_title is not None:
                plt.title(plt_title)
            else:
                plt.title(self.title[self.expl_type])
            plt.savefig(
                transform_address(
                    res_dir +
                    f'\\{self.dataset}_{self.expl_type}_{dt_id}_{j}_{exp_edge_index.cpu().numpy().shape}.png'
                )
            )
            plt.close()

    def plot_pr_edges(self, explanations, res_dir, dt_id, f_name=None, plt_title=None):

        for j, exp_edge_index in enumerate(explanations):
            pos_edges = [(u, v) for (u, v) in exp_edge_index.t().cpu().numpy()]
            plt.close()
            for hh in self.list_classes:
                node_filter = []
                for zz in self.label2nodes[hh]:
                    if zz in self.G.nodes():
                        node_filter.append(zz)
                nx.draw_networkx_nodes(self.G, self.pos,
                                       nodelist=node_filter,
                                       node_color=self.colors[hh % len(self.colors)],
                                       node_size=50, label=str(hh))

            nx.draw_networkx_edges(self.G, self.pos,
                                   width=2, alpha=1, edge_color='grey', style=':')
            nx.draw_networkx_edges(self.G, self.pos, edgelist=pos_edges,
                                   width=2, alpha=1, edge_color='green')
            plt.legend()
            if plt_title is not None:
                plt.title(plt_title)
            else:
                plt.title(self.title[self.expl_type])

            if f_name is None:
                plt.savefig(
                    transform_address(
                        res_dir +
                        f'\\{self.dataset}_{self.expl_type}_{dt_id}_{j}_{exp_edge_index.cpu().numpy().shape}.png'
                    )
                )
            else:
                plt.savefig(
                    transform_address(
                        res_dir +
                        f'\\{self.dataset}_{self.expl_type}_{dt_id}_{f_name}.png')
                )
            plt.close()
