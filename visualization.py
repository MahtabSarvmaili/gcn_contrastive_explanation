from __future__ import division
from __future__ import print_function
import os, sys

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
from torch_geometric.utils import dense_to_sparse
import numpy as np
from sklearn.manifold import TSNE
import matplotlib
import matplotlib.pyplot as plt
from distinctipy import distinctipy
import torch
import networkx as nx

np.random.seed(0)
matplotlib.use('Agg')

colors = ['orange', 'green', 'blue', 'maroon', 'brown', 'darkslategray', 'paleturquoise', 'darksalmon',
          'slategray', 'mediumseagreen', 'mediumblue', 'orchid', ]
colors = np.random.permutation(colors)


def plot(X, fig, col, size, true_labels, centroid=None):
    ax = fig.add_subplot(1, 1, 1)
    for i, point in enumerate(X):
        ax.scatter(point[0], point[1], s=size, color=col[true_labels[i]])
    if centroid is not None:
        for i, point in enumerate(centroid):
            ax.scatter(point[0], point[1], marker='s', s=10*size, color=col[i])


def plotClusters(hidden_emb, true_labels, centroid, name, n_class):
    colors = distinctipy.get_colors(36)
    tsne = TSNE(n_components=2)
    X_tsne = tsne.fit_transform(np.concatenate((hidden_emb, centroid)))
    X_centroid = X_tsne[-n_class:]
    X_tsne = X_tsne[:-n_class]
    # Plot figure
    fig = plt.figure()
    plot(X_tsne, fig, colors, 4, true_labels, X_centroid)
    fig.savefig(f"{name}.png")


def simple_plot(x, y=None, labels=None, name=''):
    for i in range(len(y)):
        plt.plot(x, y[i], label=labels[i])
    plt.legend(loc='best')
    plt.savefig(f'{name}.png')
    plt.close()


def plot_high_dim(hidden_emb, true_labels, name):
    colors = distinctipy.get_colors(36)
    tsne = TSNE(n_components=2)
    X_tsne = tsne.fit_transform(hidden_emb)
    X_tsne = X_tsne
    # Plot figure
    fig = plt.figure()
    plot(X_tsne, fig, colors, 4, true_labels)
    fig.savefig(f"{name}.png")


def plot_graph(adj, labels, node_idx, name, org_edge_idx=None, plot_grey_edges=True):
    n_nodes = adj.shape[0]
    edge_index = dense_to_sparse(torch.tensor(adj))[0].t().cpu().numpy()
    if len(edge_index) == 0:
        print(f'{name} No edge exist -> The adjacency matrix is not valid')
        return
    edge_list = [(i[0], i[1]) for i in edge_index]

    plt.close()
    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))
    G.add_edges_from(edge_list)
    pos = nx.spring_layout(G)
    # explicitly set positions
    for cc in nx.connected_components(G):
        if node_idx in cc:
            G = G.subgraph(cc).copy()
            break
    a = np.array(edge_list)
    a = a[a[:, 0] == node_idx]
    # only plotting the edges and neighboring nodes of the node_idx
    pos_edges = [(u, v) for (u, v) in a if (u, v)]
    res = (org_edge_idx[:, None] == edge_index).all(-1).any(-1)
    removed_edges = org_edge_idx[res == False]
    removed_edges = removed_edges[removed_edges[:, 0] == node_idx]
    removed_edges_list = [(i[0], i[1]) for i in removed_edges]

    # plotting the whole adjacency matrix
    # pos_edges = edge_list
    max_label = labels.max() + 1
    nmb_nodes = adj.shape[0]
    label2nodes = []
    for i in range(max_label):
        label2nodes.append([])
    for i in range(nmb_nodes):
        label2nodes[labels[i]].append(i)

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

    if plot_grey_edges:
        nx.draw_networkx_edges(G, pos, width=1, alpha=1, edge_color='grey', style=':')

    nx.draw_networkx_edges(G, pos,
                           edgelist=pos_edges,
                           width=1, alpha=1)

    # plotting removed nodes and edges in red
    if len(removed_edges) is not 0:
        removed_nodes = set(removed_edges.reshape(-1))
        removed_nodes.remove(node_idx)
        nx.draw_networkx_nodes(G, pos,
                               nodelist=removed_nodes,
                               node_color='red',
                               node_size=20)
        nx.draw_networkx_edges(G, pos,
                               edgelist=removed_edges_list,
                               width=1, alpha=0.8, edge_color='red', style='-')

    ax = plt.gca()
    ax.margins(0.11)
    plt.tight_layout()
    plt.legend()
    plt.title(name)
    plt.axis("off")
    plt.savefig(name)


def plot_cf_graph(adj, labels, node_idx, name):

    edge_index = dense_to_sparse(torch.tensor(adj))[0].t().cpu().numpy()
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


    plt.close()
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edge_list)
    pos = nx.spring_layout(G)

    a = np.array(edge_list)
    a = a[a[:, 0] == node_idx]
    # only plotting the edges and neighboring nodes of the node_idx
    pos_edges = [(u, v) for (u, v) in a if (u, v)]
    # explicitly set positions
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

    nx.draw_networkx_edges(G, pos, width=1, alpha=1, edge_color='grey', style=':')
    nx.draw_networkx_edges(G, pos,
                           edgelist=pos_edges,
                           width=1, alpha=1)

    # plotting removed nodes and edges in red

    ax = plt.gca()
    ax.margins(0.11)
    plt.tight_layout()
    plt.legend()
    plt.title(name)
    plt.axis("off")
    plt.savefig(name)
