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
        lpur = cf_examples[i][11]
        lgd = cf_examples[i][12]
        l1 = cf_examples[i][13]
        l2 = cf_examples[i][14]
        ae = cf_examples[i][15]
        res.append([f, s, l, lpur, lgd, l1, l2, ae])

    df = pd.DataFrame(res, columns=['fidelity', 'sparsity', 'l1_norm', 'loss_perturb', 'loss_dist', 'l1_p_hat', 'l2_p_hat', 'ae_dist'])
    df.to_csv(f'{name}.csv', index=False)


def removed_1by1_edges(edge_list, adj, features, model, name):
    adj1 = adj.clone()
    b = model.forward(features, adj, logit=False)
    res = []
    for x in edge_list:
        adj1[x[0]][x[1]] = 0
        adj1[x[1]][x[0]] = 0
        a = model(features, adj1)
        f = (a.argmax(dim=1) == b.argmax(dim=1)).sum() / a.__len__()
        f = f.cpu().numpy()
        s = (adj1 < adj).sum() / adj.sum()
        s = s.cpu().numpy()
        res.append([f, s])
    df = pd.DataFrame(res, columns=['fidelity', 'sparsity'])
    df.to_csv(f'{name}.csv', index=False)
