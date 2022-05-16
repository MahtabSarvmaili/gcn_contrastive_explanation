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


def plot_graph(adj, labels):
    n_nodes = adj.shape[0]
    colors = ['orange', 'red', 'green', 'blue', 'maroon', 'brown', 'darkslategray', 'paleturquoise', 'darksalmon',
              'slategray', 'mediumseagreen', 'mediumblue', 'orchid', ]
    colors = np.random.permutation(colors)
    a = dense_to_sparse(torch.tensor(adj))[0].t().cpu().numpy()
    edge_list = []
    for i in a:
        edge_list.append((i[0], i[1]))

    colors = np.random.permutation(colors)
    plt.close()
    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))
    G.add_edges_from(edge_list)
    # explicitly set positions
    pos = nx.spring_layout(G, seed=63)
    # Plot nodes with different properties for the "wall" and "roof" nodes
    for i in labels.unique().cpu():
        nx.draw_networkx_nodes(
            G, pos, node_size=20, nodelist=np.arange(n_nodes)[labels.cpu()==i], node_color=colors[i]
        )

    nx.draw_networkx_edges(G, pos, alpha=1, width=5)
    ax = plt.gca()
    ax.margins(0.11)
    plt.tight_layout()
    plt.axis("off")
    plt.savefig('testtttt')