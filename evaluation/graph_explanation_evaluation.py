import gc
import pandas as pd
from scipy.spatial.distance import cosine
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from evaluation.evaluation_metrics import gen_graph, centrality
import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
from torch_geometric.utils import dense_to_sparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import networkx as nx


def graph_evaluation_metrics(
        dt,
        explanation,
        args,
        res_dir
):
    g1 = gen_graph(list(range(dt.x.shape[0])), dt.edge_index.cpu().t().numpy())
    g1_cent = centrality(g1)
    res = []
    adj = to_dense_adj(edge_index=dt.edge_index, max_num_nodes=dt.x.shape[0]).squeeze(dim=0)

    for j, expl in enumerate(explanation):
        print(f'processing {j}th explanation with number of connections as {expl.shape}')
        if not expl.shape.__contains__(0):
            expl_adj = to_dense_adj(edge_index=expl, max_num_nodes=dt.x.shape[0]).squeeze(dim=0)
            g2 = gen_graph(list(range(dt.x.shape[0])), expl.cpu().t().numpy())
            g2_cent = centrality(g2)
            s = ((expl_adj < adj).sum())/(adj.shape[0]*adj.shape[0])
            s = s.item()
            r = (expl_adj <adj).sum().item()
            l = torch.linalg.norm(adj, ord=1)
            l = l.item()
            p = expl_adj.sum().item()
            be = cosine(
                    np.array(list(g1_cent['betweenness'].values())),
                    np.array(list(g2_cent['betweenness'].values()))
                )
            cl = cosine(
                    np.array(list(g1_cent['closeness'].values())),
                    np.array(list(g2_cent['closeness'].values()))
                )

            res.append([s, r, l, cl, be, p])
            print(f'Sparsity: {s}, Removed Connections: {r}, Betweenness: {be}, Closeness: {cl}')
            gc.collect()

    df = pd.DataFrame(
        res,
        columns=[
            'sparsity',
            'removed_edges',
            'loss_perturb',
            'closeness',
            'betweenness',
            'present_edges'
        ]
    )
    df.to_csv(
        res_dir + f'\\{args.dataset_str}_{args.expl_type}_{len(explanation)}.csv'
    )


# def plot_explanation_subgraph(
#         edge_index, exp_edge_index,
#         labels,
#         num_nodes, num_classes,
#         name='tst.png',
#
# ):
#     colors = ['orange', 'green', 'blue', 'maroon', 'brown', 'darkslategray', 'paleturquoise', 'darksalmon',
#               'slategray', 'mediumseagreen', 'mediumblue', 'orchid', ]
#     colors = np.random.permutation(colors)
#
#     edge_list = [(i[0], i[1]) for i in edge_index.t().cpu().numpy()]
#     # only plotting the edges and neighboring nodes of the node_idx
#     pos_edges = [(u, v) for (u, v) in exp_edge_index.t().cpu().numpy()]
#     removed_edges = [x for x in edge_list if x not in pos_edges]
#
#     label2nodes = []
#     for i in range(num_classes):
#         label2nodes.append([])
#     for i in range(num_nodes):
#         label2nodes[labels[i]].append(i)
#
#     G = nx.Graph()
#     G.add_nodes_from(list(range(num_nodes)))
#     G.add_edges_from(edge_list)
#     pos = nx.spring_layout(G)
#
#     plt.close()
#     for i in range(num_classes):
#         node_filter = []
#         for j in range(len(label2nodes[i])):
#             if label2nodes[i][j] in G.nodes():
#                 node_filter.append(label2nodes[i][j])
#         nx.draw_networkx_nodes(G, pos,
#                                nodelist=node_filter,
#                                node_color=colors[i % len(colors)],
#                                node_size=18, label=str(i))
#
#     nx.draw_networkx_edges(G, pos, edgelist=pos_edges,
#                            width=1, alpha=1, edge_color='black')
#
#     nx.draw_networkx_edges(G, pos, edgelist=removed_edges,
#                            width=1, alpha=1, edge_color='red')
#     plt.savefig(name)
#     plt.close()
