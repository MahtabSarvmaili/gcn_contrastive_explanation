from datetime import datetime
import networkx as nx
import torch
from torch_geometric.utils import dense_to_sparse
import pandas as pd
import numpy as np
from utils import normalize_adj
import networkx as nx

torch.manual_seed(0)
np.random.seed(0)

def gen_graph(nodes, edge_list):
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edge_list)
    return G


def centrality(graph:nx.Graph):
    centrality_metrics = {}
    centrality_metrics['brandes'] = nx.betweenness_centrality(graph)
    centrality_metrics['closeness'] = nx.closeness_centrality(graph)
    centrality_metrics['betweenness'] = nx.betweenness_centrality(graph)

    return centrality_metrics


def clustering(graph:nx):
    return nx.clustering(graph)


def graph_evaluation_metrics(model, sub_feat, sub_adj, sub_labels, cf_examples, cen_dist, name='', device='cuda'):

    b = model.forward(sub_feat, normalize_adj(sub_adj, device), logit=False)
    res = []
    for i in range(len(cf_examples)):
        cf_adj = torch.from_numpy(cf_examples[i][2]).cuda()
        a = cf_examples[i][8]
        f = (a == b.argmax(dim=1).cpu()).sum() / a.__len__()
        f = f.cpu().numpy()
        s = (cf_adj <sub_adj).sum()/ sub_adj.sum()
        s = s.cpu().numpy()
        r = (cf_adj <sub_adj).sum().cpu().numpy()
        l = torch.linalg.norm(cf_adj, ord=1)
        l = l.cpu().numpy()
        p = cf_adj.sum().cpu().numpy()
        ###
        idxs = dense_to_sparse(torch.FloatTensor(cf_examples[i][2]))[0][1].cpu().numpy()
        ppf = (cf_examples[i][8][idxs].cpu().numpy() == sub_labels.cpu().numpy()[idxs]).sum() / idxs.__len__()

        lgd = cf_examples[i][12]
        l1 = cf_examples[i][13]
        l2 = cf_examples[i][14]
        ae = cf_examples[i][15]
        lpur = cf_examples[i][16]
        bt = cen_dist['betweenness'][i]
        cl = cen_dist['closeness'][i]
        res.append([f, s, r, l, lpur, lgd, l1, l2, ae, ppf, cl, bt, p])

    df = pd.DataFrame(
        res,
        columns=[
            'fidelity', 'sparsity',
            'removed_edges', 'l1_norm',
            'loss_perturb', 'loss_dist',
            'l1_p_hat', 'l2_p_hat',
            'ae_dist', 'pp_fidelity',
            'closeness', 'betweenness',
            'present_edges'
        ]
    )

    df.to_csv(f'{name}.csv', index=False)
