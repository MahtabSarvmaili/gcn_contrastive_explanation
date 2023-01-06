from datetime import datetime
import torch
import pandas as pd
import numpy as np
from torch_geometric.utils import dense_to_sparse
from scipy.spatial.distance import cosine
from utils import normalize_adj
from visualization import plot_graph, plot_centrality
from evaluation.evaluation_metrics import gen_graph, graph_evaluation_metrics, centrality
from networkx import double_edge_swap, adjacency_matrix
from cf_explainer import CFExplainer
# torch.manual_seed(0)
# np.random.seed(0)


def evaluate_cf_PN(
        explainer_args, model, sub_feat, sub_adj, sub_labels, sub_edge_index, new_idx, i, cf_example, pcf_example=None
):
    s = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    plotting_graph = plot_graph(
        sub_adj.cpu().numpy(),
        new_idx,
        f'{explainer_args.graph_result_dir}/'
        f'{explainer_args.dataset_str}/'
        f'cf_expl_{explainer_args.cf_expl}/'
        f'pn_pp_{explainer_args.PN_PP}/'
        f'{explainer_args.algorithm}/'
        f'_{i}_sub_adj_{explainer_args.graph_result_name}_'
        f'{s}.png'
    )
    plotting_graph.plot_org_graph(
        sub_adj.cpu().numpy(),
        sub_labels.cpu().numpy(),
        new_idx,
        name=f'{explainer_args.graph_result_dir}/'
        f'{explainer_args.dataset_str}/'
        f'cf_expl_{explainer_args.cf_expl}/'
        f'pn_pp_{explainer_args.PN_PP}/'
        f'{explainer_args.algorithm}/'
        f'_{i}_sub_adj_{explainer_args.graph_result_name}.png',
        plot_grey_edges=True
    )

    nodes = list(range(sub_adj.shape[0]))
    g = gen_graph(nodes, sub_edge_index.cpu().t().numpy())
    cen = centrality(g)
    cen_dist = {'betweenness':[], 'closeness':[], 'brandes':[]}
    temp = []
    for j, x in enumerate(cf_example):

        cf_sub_adj = x[2]
        if cf_sub_adj.sum() < sub_adj.sum():
            del_edge_adj = 1 * (cf_sub_adj < sub_adj.cpu().numpy())
            plotting_graph.plot_cf_graph(
                cf_sub_adj,
                del_edge_adj,
                x[8].numpy(),
                new_idx,
                f'{explainer_args.graph_result_dir}/'
                f'{explainer_args.dataset_str}/'
                f'cf_expl_{explainer_args.cf_expl}/'
                f'pn_pp_{explainer_args.PN_PP}/'
                f'{explainer_args.algorithm}/'
                f'_{i}_counter_factual_{j}_'
                f'_epoch_{x[3]}_'
                f'{explainer_args.graph_result_name}__removed_edges___'
                f'{s}.png'
            )
            plotting_graph.plot_org_graph(
                del_edge_adj,
                x[8].numpy(),
                new_idx,
                name=f'{explainer_args.graph_result_dir}/'
                     f'{explainer_args.dataset_str}/'
                     f'cf_expl_{explainer_args.cf_expl}/'
                     f'pn_pp_{explainer_args.PN_PP}/'
                     f'{explainer_args.algorithm}/'
                     f'_{i}_counter_cf_adj_{explainer_args.graph_result_name}_'
                     f'{s}.png',
                plot_grey_edges=True
            )
            cf_edge_index = dense_to_sparse(torch.tensor(cf_sub_adj))[0].t().cpu().numpy()
            cf_nodes = list(range(cf_sub_adj.shape[0]))
            cf_g = gen_graph(cf_nodes, cf_edge_index)
            cf_cen = centrality(cf_g)
            cen_dist['betweenness'].append(
                cosine(
                    np.array(list(cf_cen['betweenness'].values())),
                    np.array(list(cen['betweenness'].values()))
                )
            )
            cen_dist['closeness'].append(
                cosine(
                    np.array(list(cf_cen['closeness'].values())),
                    np.array(list(cen['closeness'].values()))
                )
            )
            cen_dist['brandes'].append(
                cosine(
                    np.array(list(cf_cen['brandes'].values())),
                    np.array(list(cen['brandes'].values()))
                )
            )
            temp.append(x)
    graph_evaluation_metrics(
        model,
        sub_feat,
        sub_adj,
        sub_labels,
        temp,
        cen_dist,
        pcf_example,
        explainer_args.PN_PP,
        f'{explainer_args.graph_result_dir}/'
        f'{explainer_args.dataset_str}/'
        f'cf_expl_{explainer_args.cf_expl}/'
        f'pn_pp_{explainer_args.PN_PP}/'
        f'{explainer_args.algorithm}/'
        f'_{i}_counter_factual_{explainer_args.graph_result_name}_sub_graph_'
        f'{s}'

    )


def evaluate_cf_PP(
        explainer_args, model, sub_feat, sub_adj, sub_labels, sub_edge_index, new_idx, i, cf_example, pcf_example=None
):
    s = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    plotting_graph = plot_graph(
        sub_adj.cpu().numpy(),
        new_idx,
        f'{explainer_args.graph_result_dir}/'
        f'{explainer_args.dataset_str}/'
        f'cf_expl_{explainer_args.cf_expl}/'
        f'pn_pp_{explainer_args.PN_PP}/'
        f'{explainer_args.algorithm}/'
        f'_{i}_sub_adj_{explainer_args.graph_result_name}.png'
    )
    plotting_graph.plot_org_graph(
        sub_adj.cpu().numpy(),
        sub_labels.cpu().numpy(),
        new_idx,
        name=f'{explainer_args.graph_result_dir}/'
        f'{explainer_args.dataset_str}/'
        f'cf_expl_{explainer_args.cf_expl}/'
        f'pn_pp_{explainer_args.PN_PP}/'
        f'{explainer_args.algorithm}/'
        f'_{i}_sub_adj_{explainer_args.graph_result_name}.png',
        plot_grey_edges=True
    )

    nodes = list(range(sub_adj.shape[0]))
    g = gen_graph(nodes, sub_edge_index.cpu().t().numpy())
    cen = centrality(g)
    cen_dist = {'betweenness':[], 'closeness':[], 'brandes':[]}
    temp = []
    for j, x in enumerate(cf_example):

        cf_sub_adj = x[2]
        if cf_sub_adj.sum() < sub_adj.sum() and cf_sub_adj.sum() != 0:
            del_edge_adj = 1 * (cf_sub_adj < sub_adj.cpu().numpy())
            plotting_graph.plot_cf_graph(
                cf_sub_adj,
                del_edge_adj,
                x[8].numpy(),
                new_idx,
                f'{explainer_args.graph_result_dir}/'
                f'{explainer_args.dataset_str}/'
                f'cf_expl_{explainer_args.cf_expl}/'
                f'pn_pp_{explainer_args.PN_PP}/'
                f'{explainer_args.algorithm}/'
                f'_{i}_counter_factual_{j}_'
                f'_epoch_{x[3]}_'
                f'{explainer_args.graph_result_name}__removed_edges__'
                f'{s}.png'
            )
            plotting_graph.plot_org_graph(
                cf_sub_adj,
                x[8].numpy(),
                new_idx,
                f'{explainer_args.graph_result_dir}/'
                f'{explainer_args.dataset_str}/'
                f'cf_expl_{explainer_args.cf_expl}/'
                f'pn_pp_{explainer_args.PN_PP}/'
                f'{explainer_args.algorithm}/'
                f'_{i}_cf_PP_adj_{j}_'
                f'_epoch_{x[3]}_'
                f'{explainer_args.graph_result_name}__'
                f'{s}.png',
                plot_grey_edges=False
            )
            cf_edge_index = dense_to_sparse(torch.tensor(cf_sub_adj))[0].t().cpu().numpy()
            cf_nodes = list(range(cf_sub_adj.shape[0]))
            cf_g = gen_graph(cf_nodes, cf_edge_index)
            cf_cen = centrality(cf_g)
            cen_dist['betweenness'].append(
                cosine(
                    np.array(list(cf_cen['betweenness'].values())),
                    np.array(list(cen['betweenness'].values()))
                )
            )
            cen_dist['closeness'].append(
                cosine(
                    np.array(list(cf_cen['closeness'].values())),
                    np.array(list(cen['closeness'].values()))
                )
            )
            cen_dist['brandes'].append(
                cosine(
                    np.array(list(cf_cen['brandes'].values())),
                    np.array(list(cen['brandes'].values()))
                )
            )
            temp.append(x)

    graph_evaluation_metrics(
        model,
        sub_feat,
        sub_adj,
        sub_labels,
        temp,
        cen_dist,
        pcf_example,
        explainer_args.PN_PP,
        f'{explainer_args.graph_result_dir}/'
        f'{explainer_args.dataset_str}/'
        f'cf_expl_{explainer_args.cf_expl}/'
        f'pn_pp_{explainer_args.PN_PP}/'
        f'{explainer_args.algorithm}/'
        f'_{i}_counter_factual_{explainer_args.graph_result_name}_sub_graph_'
        f'{s}'
    )


def swap_edges(sub_adj, sub_edge_index, num_samples):
    nodes = list(range(sub_adj.shape[0]))
    g = gen_graph(nodes, sub_edge_index.cpu().t().numpy())
    sub_adj_p = []
    for _ in range(num_samples):
        g_p = double_edge_swap(g, nswap=1)
        sub_adj_p.append(torch.FloatTensor(adjacency_matrix(g_p, nodelist=nodes).todense()).cuda())
    return sub_adj_p

