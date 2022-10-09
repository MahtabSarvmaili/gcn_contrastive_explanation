import torch
from torch_geometric.utils import dense_to_sparse
import pandas as pd
import numpy as np
from utils import normalize_adj
from visualization import plot_graph, plot_centrality
from evaluation.evaluation_metrics import gen_graph, graph_evaluation_metrics, centrality
import networkx as nx

torch.manual_seed(0)
np.random.seed(0)


def evaluate_cf_PN(
        explainer_args, model, sub_feat, sub_adj, sub_labels, sub_edge_index, new_idx, i, cf_example, pcf_example=None
):
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
                f'{explainer_args.graph_result_name}__removed_edges__.png',
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
                     f'_{i}_counter_cf_adj_{explainer_args.graph_result_name}.png',
                plot_grey_edges=True
            )
            cf_edge_index = dense_to_sparse(torch.tensor(cf_sub_adj))[0].t().cpu().numpy()
            cf_nodes = list(range(cf_sub_adj.shape[0]))
            cf_g = gen_graph(cf_nodes, cf_edge_index)
            cf_cen = centrality(cf_g)
            plot_centrality(
                cen, cf_cen,
                f'{explainer_args.graph_result_dir}/'
                f'{explainer_args.dataset_str}/'
                f'cf_expl_{explainer_args.cf_expl}/'
                f'pn_pp_{explainer_args.PN_PP}/'
                f'{explainer_args.algorithm}/'
                f'_{i}_counter_factual_{j}_'
                f'_epoch_{x[3]}_'
                f'{explainer_args.graph_result_name}__centrality__'
            )
            cen_dist['betweenness'].append(np.linalg.norm(
                    np.array(list(cf_cen['betweenness'].values())) - np.array(list(cen['betweenness'].values())))
            )
            cen_dist['closeness'].append(np.linalg.norm(
                    np.array(list(cf_cen['closeness'].values())) - np.array(list(cen['closeness'].values())))
            )
            cen_dist['brandes'].append(np.linalg.norm(
                    np.array(list(cf_cen['brandes'].values())) - np.array(list(cen['brandes'].values())))
            )
            temp.append(x)
    # for j, x in enumerate(pcf_example):
    #     cf_sub_adj = x[2]
    #     if cf_sub_adj.sum() < sub_adj.sum():
    #         del_edge_adj = 1 * (cf_sub_adj < sub_adj.cpu().numpy())
    #         plotting_graph.plot_cf_graph(
    #             cf_sub_adj,
    #             del_edge_adj,
    #             x[8].numpy(),
    #             new_idx,
    #             f'{explainer_args.graph_result_dir}/'
    #             f'{explainer_args.dataset_str}/'
    #             f'cf_expl_{explainer_args.cf_expl}/'
    #             f'pn_pp_{explainer_args.PN_PP}/'
    #             f'{explainer_args.algorithm}/'
    #             f'_{i}_perturbed_CF_{j}_'
    #             f'_epoch_{x[3]}_'
    #             f'{explainer_args.graph_result_name}__removed_edges__.png',
    #         )
    #         plotting_graph.plot_org_graph(
    #             del_edge_adj,
    #             x[8].numpy(),
    #             new_idx,
    #             name=f'{explainer_args.graph_result_dir}/'
    #                  f'{explainer_args.dataset_str}/'
    #                  f'cf_expl_{explainer_args.cf_expl}/'
    #                  f'pn_pp_{explainer_args.PN_PP}/'
    #                  f'{explainer_args.algorithm}/'
    #                  f'_{i}_perturbed_CF_cf_adj_{explainer_args.graph_result_name}.png',
    #             plot_grey_edges=True
    #         )
    #
    #         cf_edge_index = dense_to_sparse(torch.tensor(cf_sub_adj))[0].t().cpu().numpy()
    #         cf_nodes = list(range(cf_sub_adj.shape[0]))
    #         cf_g = gen_graph(cf_nodes, cf_edge_index)
    #         cf_cen = centrality(cf_g)
    #         plot_centrality(
    #             cen, cf_cen,
    #             f'{explainer_args.graph_result_dir}/'
    #             f'{explainer_args.dataset_str}/'
    #             f'cf_expl_{explainer_args.cf_expl}/'
    #             f'pn_pp_{explainer_args.PN_PP}/'
    #             f'{explainer_args.algorithm}/'
    #             f'_{i}_counter_factual_{j}_'
    #             f'_epoch_{x[3]}_'
    #             f'{explainer_args.graph_result_name}__centrality__'
    #         )
    graph_evaluation_metrics(
        model,
        sub_feat,
        sub_adj,
        temp,
        cen_dist,
        f'{explainer_args.graph_result_dir}/'
        f'{explainer_args.dataset_str}/'
        f'cf_expl_{explainer_args.cf_expl}/'
        f'pn_pp_{explainer_args.PN_PP}/'
        f'{explainer_args.algorithm}/'
        f'_{i}_counter_factual_{explainer_args.graph_result_name}_sub_graph_'
    )


def evaluate_cf_PP(
        explainer_args, model, sub_feat, sub_adj, sub_labels, sub_edge_index, new_idx, i, cf_example, pcf_example=None
):
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

    # nodes = list(range(sub_adj.shape[0]))
    # g = gen_graph(nodes, sub_edge_index.cpu().t().numpy())
    # cen = centrality(g)
    for j, x in enumerate(cf_example):

        cf_sub_adj = x[2]
        if cf_sub_adj.sum() < sub_adj.sum():
            del_edge_adj = 1 * (cf_sub_adj < sub_adj.cpu().numpy())
            # plotting_graph.plot_cf_graph(
            #     cf_sub_adj,
            #     del_edge_adj,
            #     x[8].numpy(),
            #     new_idx,
            #     f'{explainer_args.graph_result_dir}/'
            #     f'{explainer_args.dataset_str}/'
            #     f'cf_expl_{explainer_args.cf_expl}/'
            #     f'pn_pp_{explainer_args.PN_PP}/'
            #     f'{explainer_args.algorithm}/'
            #     f'_{i}_counter_factual_{j}_'
            #     f'_epoch_{x[3]}_'
            #     f'{explainer_args.graph_result_name}__removed_edges__.png',
            # )
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
                f'{explainer_args.graph_result_name}__.png',
                plot_grey_edges=False
            )
            # cf_edge_index = dense_to_sparse(torch.tensor(cf_sub_adj))[0].t().cpu().numpy()
            # cf_nodes = list(range(cf_sub_adj.shape[0]))
            # cf_g = gen_graph(cf_nodes, cf_edge_index)
            # cf_cen = centrality(cf_g)
            # plot_centrality(
            #     cen, cf_cen,
            #     f'{explainer_args.graph_result_dir}/'
            #     f'{explainer_args.dataset_str}/'
            #     f'cf_expl_{explainer_args.cf_expl}/'
            #     f'pn_pp_{explainer_args.PN_PP}/'
            #     f'{explainer_args.algorithm}/'
            #     f'_{i}_counter_factual_{j}_'
            #     f'_epoch_{x[3]}_'
            #     f'{explainer_args.graph_result_name}__centrality__'
            # )
    for j, x in enumerate(pcf_example):
        cf_sub_adj = x[2]
        if cf_sub_adj.sum() < sub_adj.sum():
            del_edge_adj = 1 * (cf_sub_adj < sub_adj.cpu().numpy())
            plotting_graph.plot_org_graph(
                cf_sub_adj,
                x[8].numpy(),
                new_idx,
                name=f'{explainer_args.graph_result_dir}/'
                     f'{explainer_args.dataset_str}/'
                     f'cf_expl_{explainer_args.cf_expl}/'
                     f'pn_pp_{explainer_args.PN_PP}/'
                     f'{explainer_args.algorithm}/'
                     f'_{i}_perturbed_CF_cf_adj_{j}_{explainer_args.graph_result_name}.png',
                plot_grey_edges=False
            )
    # graph_evaluation_metrics(
    #     model,
    #     sub_feat,
    #     sub_adj,
    #     cf_example,
    #     f'{explainer_args.graph_result_dir}/'
    #     f'{explainer_args.dataset_str}/'
    #     f'cf_expl_{explainer_args.cf_expl}/'
    #     f'pn_pp_{explainer_args.PN_PP}/'
    #     f'{explainer_args.algorithm}/'
    #     f'_{i}_counter_factual_{explainer_args.graph_result_name}_sub_graph_'
    # )
