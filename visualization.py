from __future__ import division
from __future__ import print_function
import os, sys

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
import numpy as np
from sklearn.manifold import TSNE
import matplotlib
import matplotlib.pyplot as plt
from distinctipy import distinctipy

np.random.seed(0)
matplotlib.use('Agg')


def plot(X, fig, col, size, true_labels, centroid=None):
    ax = fig.add_subplot(1, 1, 1)
    for i, point in enumerate(X):
        ax.scatter(point[0], point[1], s=size, color=col[true_labels[i]])
    if centroid is not None:
        for i, point in enumerate(centroid):
            ax.scatter(point[0], point[1], marker='s', s=10*size, color=col[i])


def plotClusters(hidden_emb, true_labels, centroid, name):
    colors = distinctipy.get_colors(36)
    tsne = TSNE(n_components=2)
    X_tsne = tsne.fit_transform(np.concatenate((hidden_emb, centroid)))
    X_centroid = X_tsne[-16:]
    X_tsne = X_tsne[:-16]
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
