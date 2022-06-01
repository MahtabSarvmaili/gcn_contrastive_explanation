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


def plot_graph(adj, labels, node_idx, name, org_edge_idx=None):
    n_nodes = adj.shape[0]
    edge_index = dense_to_sparse(torch.tensor(adj))[0].t().cpu().numpy()
    if len(edge_index) == 0:
        print(f'{name} No edge exist -> The adjacency matrix is not valid')
        return
    edge_list = []
    for i in edge_index:
        edge_list.append((i[0], i[1]))

    # colors = np.random.permutation(colors)
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
    pos_edges = [(u, v) for (u, v) in a]
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
                           node_size=100, node_shape='s', label=str(labels[node_idx]))

    nx.draw_networkx_edges(G, pos, width=1, alpha=0.5, edge_color='grey')

    nx.draw_networkx_edges(G, pos,
                           edgelist=pos_edges,
                           width=1, alpha=0.5)

    if org_edge_idx is not None:
        actual_nodes = org_edge_idx[org_edge_idx[:, 0] == node_idx][:,1]
        nx.draw_networkx_nodes(G, pos,
                               nodelist=actual_nodes,
                               node_color='red',
                               node_size=10, node_shape='v', label='neighbors')

    ax = plt.gca()
    ax.margins(0.11)
    plt.tight_layout()
    plt.legend()
    plt.title(name)
    plt.axis("off")
    plt.savefig(name)