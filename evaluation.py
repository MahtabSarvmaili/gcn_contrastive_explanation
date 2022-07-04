import torch
from torch_geometric.utils import dense_to_sparse
from gae.utils import preprocess_graph
import pandas as pd
torch.manual_seed(0)


def prob_necessity(sub_feat, sub_adj, cf_examples, model, device='cuda'):
    elimination_idx = dense_to_sparse(cf_examples)[0].t()
    cf_adj_norm = preprocess_graph(torch.tensor(cf_examples), device='cuda')
    norm_sub_adj = preprocess_graph(sub_adj.cpu(), device=device)
    sub_node_y_pred = model.forward(sub_feat, norm_sub_adj, logit=False).argmax(dim=1)
    sub_node_cf_y_pred = model.forward(sub_feat, cf_adj_norm, logit=False).argmax(dim=1)
    labels = []

    for i in range(0, len(elimination_idx), int(len(elimination_idx) / 10) + 1):
        for x in elimination_idx[i:i + int(len(elimination_idx) / 10)]:
            a = x.cpu()
            cf_adj_norm[a[0]][a[1]] = 0
            cf_adj_norm[a[1]][a[0]] = 0
        labels.append(
            model.forward(sub_feat, torch.tensor(cf_adj_norm, device='cuda'), logit=False).argmax(dim=1))


    print(
        f'Percentage of agreement CF and Sub_adj{(sub_node_y_pred != sub_node_cf_y_pred).sum() / labels[0].__len__()}'
    )

    for j in range(len(labels)):
        print(f'Percentage of agreement Sparsed_CF and Sub_adj{(sub_node_y_pred != labels[j]).sum() / labels[0].__len__()}')


def fidelity_size_sparsity(model, sub_feat, sub_adj, cf_examples, edge_addition, name=''):

    b = model.forward(sub_feat, sub_adj, logit=False)
    res = []
    for i in range(len(cf_examples)):
        cf_adj = torch.from_numpy(cf_examples[i][2]).cuda()
        if edge_addition is not True:
            cf_adj = cf_adj.mul(sub_adj)

        a = model.forward(sub_feat, cf_adj, logit=False)
        f = (a.argmax(dim=1) == b.argmax(dim=1)).sum() / a.__len__()
        f = f.cpu().numpy()
        s = (cf_adj <sub_adj).sum()/ sub_adj.sum()
        s = s.cpu().numpy()
        l = torch.linalg.norm(cf_adj, ord=1)
        l = l.cpu().numpy()
        if len(cf_examples[i]) == 14:
            ae = cf_examples[i][-1]
            res.append([f, s, l, ae])
        else:
            res.append([f, s, l])
    if len(cf_examples[0])==14:
        df = pd.DataFrame(res, columns=['fidelity', 'sparsity', 'l1_dist', 'ae_dist'])
    else:
        df = pd.DataFrame(res, columns=['fidelity', 'sparsity', 'l1_dist'])
    df.to_csv(f'{name}.csv', index=False)


# def remove_edges(edge_list, adj, model):
#     for x in edge_list:
