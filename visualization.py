from __future__ import division
from __future__ import print_function
import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
from torch_geometric.utils import dense_to_sparse
import numpy as np
from sklearn.manifold import TSNE
import matplotlib
import matplotlib.pyplot as plt
import torch
import networkx as nx

np.random.seed(0)
torch.manual_seed(0)

matplotlib.use('Agg')

colors = ['orange', 'green', 'blue', 'maroon', 'brown', 'darkslategray', 'paleturquoise', 'darksalmon',
          'slategray', 'mediumseagreen', 'mediumblue', 'orchid', ]
colors = np.random.permutation(colors)


def plot_explanation_subgraph(adj, labels, node_idx, name, plot_grey_edges=True):

    edge_index = dense_to_sparse(torch.tensor(adj))[0].t().cpu().numpy()
    if len(edge_index) == 0:
        print(f'{name} No edge exist -> The adjacency matrix is not valid')
        return
    edge_list = [(i[0], i[1]) for i in edge_index]
    a = np.array(edge_list)
    a = a[a[:, 0] == node_idx]

    # only plotting the edges and neighboring nodes of the node_idx
    pos_edges = [(u, v) for (u, v) in a if (u, v)]
    max_label = labels.max() + 1
    nmb_nodes = adj.shape[0]
    label2nodes = []
    for i in range(max_label):
        label2nodes.append([])
    for i in range(nmb_nodes):
        label2nodes[labels[i]].append(i)
    G = nx.Graph()
    G.add_nodes_from(np.unique(edge_index.reshape(-1)))
    G.add_edges_from(edge_list)
    pos = nx.spring_layout(G)

    plt.close()
    for i in range(max_label):
        node_filter = []
        for j in range(len(label2nodes[i])):
            if label2nodes[i][j] in G.nodes():
                node_filter.append(label2nodes[i][j])
        nx.draw_networkx_nodes(G, pos,
                               nodelist=node_filter,
                               node_color=colors[i % len(colors)],
                               node_size=20, label=str(i))

    nx.draw_networkx_nodes(G, pos,
                           nodelist=[node_idx],
                           node_color='yellow',
                           node_size=50, node_shape='s', label=str(labels[node_idx]))
    nx.draw_networkx_edges(G, pos,
                           width=1, alpha=1, edge_color='grey', style=':')
    nx.draw_networkx_edges(G, pos,
                           edgelist=pos_edges,
                           width=1, alpha=1, edge_color='black')

    ax = plt.gca()
    ax.margins(0.11)
    plt.tight_layout()
    plt.legend()
    plt.title(name)
    plt.axis("off")
    plt.savefig(name)
    plt.close()




class plot_graph:
    def __init__(self, adj, node_idx, name):

        self.n_nodes = adj.shape[0]
        self.node_idx = node_idx
        edge_index = dense_to_sparse(torch.tensor(adj))[0].t().cpu().numpy()
        if len(edge_index) == 0:
            print(f'{name} No edge exist -> The adjacency matrix is not valid')
            return
        edge_list = [(i[0], i[1]) for i in edge_index]

        self.G = nx.Graph()
        self.G.add_nodes_from(range(self.n_nodes))
        self.G.add_edges_from(edge_list)
        self.pos = nx.spring_layout(self.G)
        # explicitly set positions
        for cc in nx.connected_components(self.G):
            if self.node_idx in cc:
                self.G = self.G.subgraph(cc).copy()
                break
        plt.close()

    def __plot_graph__(
            self, max_label, label2nodes, node_idx, labels,
            pos_edges, edge_list=None, removed_edges_list=None, removed_edge_index=None, plot_all_edges=True, name=''
    ):

        plt.close()
        for i in range(max_label):
            node_filter = []
            for j in range(len(label2nodes[i])):
                if label2nodes[i][j] in self.G.nodes():
                    node_filter.append(label2nodes[i][j])
            nx.draw_networkx_nodes(self.G, self.pos,
                                   nodelist=node_filter,
                                   node_color=colors[i % len(colors)],
                                   node_size=20, label=str(i))

        nx.draw_networkx_nodes(self.G, self.pos,
                               nodelist=[node_idx],
                               node_color='yellow',
                               node_size=50, node_shape='s', label=str(labels[node_idx]))
        if plot_all_edges:
            nx.draw_networkx_edges(self.G, self.pos, width=1, alpha=1, edge_color='grey', style=':')
        else:
            nx.draw_networkx_edges(self.G, self.pos, edgelist=edge_list,
                                   width=1, alpha=1, edge_color='grey', style=':')

        nx.draw_networkx_edges(self.G, self.pos,
                               edgelist=pos_edges,
                               width=2, alpha=1, edge_color='black')

        if removed_edges_list is not None:
            removed_nodes = set(removed_edge_index.reshape(-1))
            nx.draw_networkx_nodes(self.G, self.pos,
                                   nodelist=removed_nodes,
                                   node_color='red',
                                   node_size=20)

            nx.draw_networkx_edges(self.G, self.pos,
                                   edgelist=removed_edges_list,
                                   width=1, alpha=0.8, edge_color='red', style='-')
        ax = plt.gca()
        ax.margins(0.11)
        plt.tight_layout()
        plt.legend()
        plt.title(name)
        plt.axis("off")
        plt.savefig(name)
        plt.close()

    def plot_org_graph(self, adj, labels, node_idx, name, plot_grey_edges=True):
        edge_index = dense_to_sparse(torch.tensor(adj))[0].t().cpu().numpy()
        if len(edge_index) == 0:
            print(f'{name} No edge exist -> The adjacency matrix is not valid')
            return
        edge_list = [(i[0], i[1]) for i in edge_index]
        a = np.array(edge_list)
        a = a[a[:, 0] == node_idx]
        # only plotting the edges and neighboring nodes of the node_idx
        pos_edges = [(u, v) for (u, v) in a if (u, v)]
        max_label = labels.max() + 1
        nmb_nodes = adj.shape[0]
        label2nodes = []
        for i in range(max_label):
            label2nodes.append([])
        for i in range(nmb_nodes):
            label2nodes[labels[i]].append(i)

        self.__plot_graph__(max_label, label2nodes, node_idx, labels, pos_edges, edge_list=edge_list, plot_all_edges=plot_grey_edges, name=name)

    def plot_cf_graph(self, adj, sub_adj, labels, node_idx, name):
        edge_index = dense_to_sparse(torch.tensor(adj))[0].t().cpu().numpy()
        removed_edge_index = dense_to_sparse(torch.tensor(sub_adj))[0].t().cpu().numpy()
        nodes = np.unique(edge_index.flatten())
        if len(edge_index) == 0:
            print(f'{name} No edge exist -> The adjacency matrix is not valid')
            return
        edge_list = [(i[0], i[1]) for i in edge_index]
        max_label = labels.max() + 1
        label2nodes = []
        for i in range(max_label):
            label2nodes.append([])
        for x in nodes:
            label2nodes[labels[x]].append(x)

        a = np.array(edge_list)
        a = a[a[:, 0] == node_idx]
        removed_edges_list = [(i[0], i[1]) for i in removed_edge_index]
        # only plotting the edges and neighboring nodes of the node_idx
        pos_edges = [(u, v) for (u, v) in a if (u, v)]
        # explicitly set positions
        self.__plot_graph__(
            max_label, label2nodes, node_idx, labels, pos_edges, removed_edges_list=removed_edges_list, removed_edge_index=removed_edge_index, name=name
        )


def plot_errors(losses, path):
    x = list(range(len(losses['loss_total'])))
    fig, axs = plt.subplots(3, figsize=(14, 12))
    fig.suptitle('Vertically stacked subplots')
    axs[0].plot(x, losses['loss_total'])
    axs[0].plot(x, losses['loss_graph_dist'])
    axs[0].legend(['loss_total', 'loss_graph_dist'])
    axs[1].plot(x, losses['loss_perturb'])
    axs[2].plot(x, losses['L1'])
    axs[2].plot(x, losses['L2'])
    axs[2].plot(x, losses['l2_AE'])
    axs[2].legend(['L1', 'L2', 'l2_AE'])
    plt.savefig(path)
    plt.close()


def plot_centrality(cen: dict, cf_cen:dict, name):
    for x in cen.keys():
        fig = plt.figure()
        ax = fig.add_axes([0, 0, 1, 1])
        ax.fill_between(cf_cen[x].keys(), cf_cen[x].values(), color="lightpink", alpha=0.5, label='CF_'+x)
        ax.plot(cf_cen[x].keys(), cf_cen[x].values(), color="hotpink", alpha=0.5)
        ax.fill_between(cen[x].keys(), cen[x].values(), color="skyblue", alpha=0.5, label='Adj_'+x)
        ax.plot(cen[x].keys(), cen[x].values(), color="Slateblue", alpha=0.5)
        ax.legend()
        plt.savefig(name+x+'.png', bbox_inches='tight')
        plt.close()