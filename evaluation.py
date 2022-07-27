import torch
from torch_geometric.utils import dense_to_sparse
import pandas as pd
import numpy as np
from utils import normalize_adj
torch.manual_seed(0)
np.random.seed(0)


def fidelity_size_sparsity(model, sub_feat, sub_adj, cf_examples, name='', device='cuda'):

    b = model.forward(sub_feat, normalize_adj(sub_adj, device), logit=False)
    res = []
    for i in range(len(cf_examples)):
        cf_adj = torch.from_numpy(cf_examples[i][2]).cuda()
        a = model.forward(sub_feat, normalize_adj(cf_adj, device), logit=False)
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


def insertion(model, features, cf_example, removed_edges, labels, node_idx, device='cuda'):
    size = [0.1, 0.3, 0.5, 0.7, 0.9, 1]
    a = dense_to_sparse(torch.tensor(removed_edges))[0].t().cpu().numpy()
    for s in size:
        copy_cf_example = cf_example.copy()
        b = np.random.choice(len(a), size=int(s*len(a)), replace=False)
        add_idx = a[b]
        for x in add_idx:
            copy_cf_example[x[0]][x[1]] = 1
            copy_cf_example[x[1]][x[0]] = 1

        ins_labels = model(features, normalize_adj(torch.tensor(copy_cf_example).cuda(), device)).argmax(dim=1).cpu().numpy()
        changed = ins_labels != labels
        percent = (1*changed).sum()/len(labels)
        node_label_change = changed[node_idx]
        print(f"the percentage of changed labels:{percent}")
        print(f"Node label changed:{node_label_change}")
    return percent


def deletion(model, features, sub_adj, removed_edges, labels, node_idx, device='cuda'):
    size = [0.1, 0.3, 0.5, 0.7, 0.9, 1]
    a = dense_to_sparse(torch.tensor(removed_edges))[0].t().cpu().numpy()
    for s in size:
        copy_cf_example = sub_adj.copy()
        b = np.random.choice(len(a), size=int(s*len(a)), replace=False)
        add_idx = a[b]
        for x in add_idx:
            copy_cf_example[x[0]][x[1]] = 0
            copy_cf_example[x[1]][x[0]] = 0

        del_labels = model(features, normalize_adj(torch.tensor(copy_cf_example).cuda(), device)).argmax(dim=1).cpu().numpy()
        changed = del_labels != labels
        percent = (1*changed).sum()/len(labels)
        node_label_change = changed[node_idx]
        print(f"Deletion: Percentage of changed labels:{percent}")
        print(f"Deletion: Node label changed:{node_label_change}")
    return percent


