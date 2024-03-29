import gc
import os
import sys
import pandas as pd
import numpy as np
import torch
from datetime import datetime
from scipy.spatial.distance import euclidean
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from evaluation.evaluation_metrics import gen_graph, centrality
from sklearn.metrics import roc_auc_score
from utils import transform_address, influential_func
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
        edge_preds,
        args,
        res_dir,
        dt_id,
        f_model,
        ground_dt=None,
        gnnexplainer_mask=None,
        pgexplainer_mask=None,
        expl_vis_func=None,
        l1_l2_strs=''
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
        gnn_inf, _ = influential_func(f_model, dt, gnnexplainer_mask, args)
        pg_inf, _ = influential_func(f_model, dt, pgexplainer_mask, args)

    for j, expl in enumerate(explanation):
        if not expl.shape.__contains__(0):
            # expl_vis_func(expl, res_dir, dt_id.item(), j)
            s, r, l, cl, be, p = evaluation_criteria(expl, adj, dt, g1_cent)

            if ground_dt is None:
                acc=0
            else:
                acc = roc_auc_score(ground_dt, edge_preds[j].cpu().numpy())
            inf_expl, inf_rand = influential_func(f_model, dt, edge_preds[j], args)
            res.append([j, acc, s, r, l, cl, be, p, inf_expl.item(), inf_rand.item()])
            gc.collect()

    res.append(['gnn', gnn_acc, gn_s, gn_r, gn_l, gn_cl, gn_be, gn_p, gnn_inf.item(), 0])
    res.append(['pg', pg_acc, pg_s, pg_r, pg_l, pg_cl, pg_be, pg_p, pg_inf.item(), 0])
    df = pd.DataFrame(
        res,
        columns=[
            'index',
            'accuracy',
            'sparsity',
            'removed_edges',
            'loss_perturb',
            'closeness',
            'betweenness',
            'present_edges',
            'pred_shift_expl',
            'pred_shift_rand'
        ]
    )

    now = datetime.now()
    dt_string = now.strftime("%d-%m-%H-%M-%S")
    df.to_csv(
        transform_address(
            res_dir + f'\\{args.dataset_str}_{args.expl_type}_{dt_id}_{len(explanation)}_{dt_string}_{l1_l2_strs}.csv'
        ),
        index=False, header=True
    )
    print(

        f'Quantitative evaluation has finished!\n'
        f'=> AUC: {df["accuracy"][:-2].max()}, PGExplainer: {pg_acc}, GNNExplainer: {gnn_acc}'
    )
    # if args.expl_type == 'CF':
    #     return df['sparsity'][:-2].argmin()
    # else:
    #     return df['accuracy'][:-2].argmax()

