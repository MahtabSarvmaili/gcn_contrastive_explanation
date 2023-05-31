from __future__ import division
from __future__ import print_function
import os
import numpy as np
import torch
from torch_geometric.utils import negative_sampling
from gae.torchgeometric_vgae import DeepVGAE
from data.data_loader import load_data_AE
from sklearn.metrics import precision_recall_fscore_support
torch.manual_seed(0)
np.random.seed(0)


# This code needs more debugging - the main code has changed but not properly adjusted for DeepVGAE
def gae(args):
    log_file = os.getcwd() + args.model_dir
    torch.cuda.empty_cache()
    data = load_data_AE(args)

    print("Using {} dataset".format(args.dataset_str))
    model = DeepVGAE(data['n_features'], args.hidden1, args.hidden2).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-6)
    model.to(args.device)
    p_ = r_ = f_ = 0

    for e in range(200):

        model.train()
        optimizer.zero_grad()
        node_embed = model(data['train'].x, data['train'].edge_index)
        neg_edge_index = negative_sampling(
            edge_index=data['train'].edge_index, num_nodes=data['train'].num_nodes,
            num_neg_samples=data['train'].edge_label_index.size(1), method='sparse')

        edge_label_index = torch.cat(
            [data['train'].edge_label_index, neg_edge_index],
            dim=-1,
        )
        edge_label = torch.cat([
            data['train'].edge_label,
            data['train'].edge_label.new_zeros(neg_edge_index.size(1))
        ], dim=0)

        train_loss = model.loss(node_embed, edge_label_index, edge_label)
        train_loss.backward()

        optimizer.step()

        if e % 5 == 0:
            with torch.no_grad():
                model.eval()
                # node_embed = model(data['val'].x, data['train'].adj)
                node_embed = model(data['val'].x, data['val'].edge_index)
                edges = data['val'].edge_label_index
                labels = data['val'].edge_label

                preds = model.link_pred(node_embed[edges[0]], node_embed[edges[1]]) > 0.5
                val_loss = model
                roc_auc, ap = model.single_test(
                    data['dataset'].x,
                    data['dataset'].train_pos_edge_index,
                    data['dataset'].test_pos_edge_index,
                    data['dataset'].test_neg_edge_index
                )
                p, r, f, _ = precision_recall_fscore_support(
                    labels.cpu().numpy(), preds.cpu().numpy(), average='macro'
                )
                print(f'VAL - Epoch {e},  loss {train_loss} VAL Loss {val_loss}, Precision {p}, Recall {r}, F-score {f}')
                if (p_+r_+f_)/3 < (p+r+f)/3:
                    torch.save(model.state_dict(), log_file)
                    p_, r_, f_ = p, r, f
    return model