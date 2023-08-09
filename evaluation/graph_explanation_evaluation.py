import gc
import os
import sys
import pandas as pd
import numpy as np
import torch
from scipy.spatial.distance import euclidean
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from evaluation.evaluation_metrics import gen_graph, centrality
from sklearn.metrics import roc_auc_score
from utils import transform_address
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))


def evaluation_criteria(expl, adj, dt, g1_cent):
    expl_adj = to_dense_adj(edge_index=expl, max_num_nodes=dt.x.shape[0]).squeeze(dim=0)
    g2 = gen_graph(list(range(dt.x.shape[0])), expl.cpu().t().numpy())
    g2_cent = centrality(g2)
    s = ((expl_adj < adj).sum()) / (adj.shape[0] ** 2)
    s = s.item()
    r = (expl_adj < adj).sum().item()
    l = torch.linalg.norm(expl_adj, ord=1)
    l = l.item()
    p = expl_adj.sum().item()
    be = euclidean(
        np.array(list(g1_cent['betweenness'].values())),
        np.array(list(g2_cent['betweenness'].values()))
    )
    cl = euclidean(
        np.array(list(g1_cent['closeness'].values())),
        np.array(list(g2_cent['closeness'].values()))
    )
    return s, r, l, cl, be, p


def graph_evaluation_metrics(
        dt,
        explanation,
        edge_pred,
        args,
        res_dir,
        dt_id,
        ground_dt=None,
        gnnexplainer_mask=None,
        pgexplainer_mask=None,
        expl_vis_func=None
):
    g1 = gen_graph(list(range(dt.x.shape[0])), dt.edge_index.cpu().t().numpy())
    g1_cent = centrality(g1)
    adj = to_dense_adj(edge_index=dt.edge_index, max_num_nodes=dt.x.shape[0]).squeeze(dim=0)

    res = []
    pg_acc= pg_s= pg_r= pg_l= pg_be = pg_cl = pg_p = 0
    gnn_acc = gn_s = gn_r = gn_l = gn_be = gn_cl = gn_p = 0

    if None not in [gnnexplainer_mask, pgexplainer_mask]:
        if ground_dt is not None:
            gnn_acc = roc_auc_score(ground_dt, gnnexplainer_mask)
            pg_acc = roc_auc_score(ground_dt, pgexplainer_mask)
        gnn_expl = dt.edge_index[:,(gnnexplainer_mask>0.5)]
        gn_s, gn_r, gn_l, gn_be, gn_cl, gn_p = evaluation_criteria(gnn_expl, adj, dt, g1_cent)
        pg_expl = dt.edge_index[:,(pgexplainer_mask>0.5)]
        pg_s, pg_r, pg_l, pg_be, pg_cl, pg_p = evaluation_criteria(pg_expl, adj, dt, g1_cent)

    for j, expl in enumerate(explanation):
        if not expl.shape.__contains__(0):
            expl_vis_func(expl, res_dir, dt_id.item(), j)
            s, r, l, cl, be, p = evaluation_criteria(expl, adj, dt, g1_cent)
            if ground_dt is None:
                acc=0
            else:
                acc = roc_auc_score(ground_dt, edge_pred[j])

            res.append([acc, s, r, l, cl, be, p])
            gc.collect()

    res.append([gnn_acc, gn_s, gn_r, gn_l, gn_cl, gn_be, gn_p])
    res.append([pg_acc, pg_s, pg_r, pg_l, pg_cl, pg_be, pg_p])
    df = pd.DataFrame(
        res,
        columns=[
            'accuracy',
            'sparsity',
            'removed_edges',
            'loss_perturb',
            'closeness',
            'betweenness',
            'present_edges'
        ]
    )
    df.to_csv(
        transform_address(
            res_dir + f'\\{args.dataset_str}_{args.expl_type}_{dt_id}_{len(explanation)}.csv'
        ),
        index=False
    )
    print(

        f'Quantitative evaluation has finished!\n'
        f'=> AUC: {df["accuracy"][:-2].max()}, PGExplainer: {pg_acc}, GNNExplainer: {gnn_acc}'
    )
    # if args.expl_type == 'CF':
    #     return df['sparsity'][:-2].argmin()
    # else:
    #     return df['accuracy'][:-2].argmax()

