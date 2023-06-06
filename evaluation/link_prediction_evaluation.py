import gc
import torch
import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from evaluation.evaluation_metrics import gen_graph, centrality


def graph_evaluation_metrics(
        edge_index,
        edge_labels,
        explanation,
        expls_preds,
        num_nodes,
        name='tst',
        device='cuda'
):
    g1 = gen_graph(list(range(num_nodes)), edge_index.cpu().t().numpy())
    g1_cent = centrality(g1)
    res = []
    adj = to_dense_adj(edge_index=edge_index, max_num_nodes=num_nodes).squeeze(dim=0)
    # calculating the stability of fidelity for perturbed examples
    lbs = edge_labels.detach().cpu().numpy()
    # fids = []
    # for x in expls_preds:
    #     f = (x == lbs).sum() / lbs.__len__()
    #     fids.append(f)

    for x, expl, j in zip(expls_preds, explanation, list(range(len(explanation)))):
        print(f'processing {j}th explanation')
        if not expl.shape.__contains__(0):
            expl_adj = to_dense_adj(edge_index=expl, max_num_nodes=num_nodes,).squeeze(dim=0)
            g2 = gen_graph(list(range(num_nodes)), expl.cpu().t().numpy())c
            g2_cent = centrality(g2)
            # a = cf_examples[i][8]
            f = (x == lbs).sum() / lbs.__len__()
            # f = f.cpu().numpy()
            # cf_expl_fids.append(f)
            s = ((expl_adj < adj).sum())/(adj.shape[0]*adj.shape[0])
            s = s.item()
            # cf_expl_sparsity.append(s)
            r = (expl_adj <adj).sum().item()
            l = torch.linalg.norm(adj, ord=1)
            l = l.item()
            p = expl_adj.sum().item()
            ###
            # idxs = dense_to_sparse(torch.FloatTensor(cf_examples[i][2]))[0][1].cpu().numpy()
            # ppf = (cf_examples[i][8][idxs].cpu().numpy() == sub_labels.cpu().numpy()[idxs]).sum() / idxs.__len__()
            # sub_graph_fid.append(ppf)
            #
            # lgd = cf_examples[i][12]
            # l1 = cf_examples[i][13]
            # l2 = cf_examples[i][14]
            # ae = cf_examples[i][15]
            # lpur = cf_examples[i][16]
            be = cosine(
                    np.array(list(g1_cent['betweenness'].values())),
                    np.array(list(g2_cent['betweenness'].values()))
                )
            cl = cosine(
                    np.array(list(g1_cent['closeness'].values())),
                    np.array(list(g2_cent['closeness'].values()))
                )

            res.append([f, s, r, l, cl, be, p])
            # res.append([f, s, r, l, p])
            gc.collect()

    # if cf_expl is True:
    #     cf_expl_sparsity = np.array(cf_expl_sparsity)
    #     max_selected_fids = np.array(cf_expl_fids)[cf_expl_sparsity.argmin()]
    # else:
    #     # if we are extracting prototype subgraph we need to use the PPF
    #     max_selected_fids = np.array(sub_graph_fid).max()
    #
    # fid_diff_actual_perturb = np.abs(max_selected_fids - pert_fidelity_max_list)
    # fid_diff_actual_perturb.mean()
    df = pd.DataFrame(
        res,
        columns=[
            'fidelity',
            'sparsity',
            'removed_edges',
            'loss_perturb',
            'closeness',
            'betweenness',
            'present_edges'
        ]
    )
    # df['stability'] = fid_diff_actual_perturb.mean()
    df.to_csv(f'{name}.csv', index=False)
