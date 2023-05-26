import argparse
import gc
import sys
import os
import traceback
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
import numpy as np
from data.data_loader import load_data, load_synthetic, load_synthetic_AE, load_data_AE
from data.gengraph import gen_syn1, gen_syn2, gen_syn3, gen_syn4
from utils import normalize_adj, get_neighbourhood, perturb_sub_adjacency, encode_onehot
from torch_geometric.utils import negative_sampling
from torch_geometric.loader import DataLoader
from torch.nn.utils import clip_grad_norm_
from sklearn.metrics import precision_recall_fscore_support
from cf_explainer import CFExplainer
from gae.GAE import gae
from evaluation.evaluation import evaluate_cf_PN, evaluate_cf_PP, swap_edges
from evaluation.link_prediction_evaluation import graph_evaluation_metrics
from gnn_models.gcn_model import GCN, GraphSparseConv
from gnn_models.gcn_perturbation import GCNSyntheticPerturb

from utils import get_link_labels

# torch.manual_seed(0)
# np.random.seed(0)

sys.path.append('../..')


def main(explainer_args):
    log_file = os.getcwd() + explainer_args.model_dir
    torch.cuda.empty_cache()
    # data = load_data(explainer_args)
    data = load_data_AE(explainer_args)

    model = GraphSparseConv(
        num_features=data['n_features'],
        num_hidden1=explainer_args.hidden1,
        num_hidden2=explainer_args.hidden2,
        nout=1,
        device=explainer_args.device
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=explainer_args.lr, weight_decay=5e-6)
    './re'
    model.to(explainer_args.device)
    p_ = r_ = f_ = 0
    model.train()

    for e in range(200):
        for edge_id in DataLoader(
                range(data['train'].edge_label_index.size(-1)), batch_size=explainer_args.batch_size, shuffle=True
        ):
            model.train()
            optimizer.zero_grad()
            # node_embed = model(data['train'].x, data['train'].adj)
            node_embed = model(data['train'].x, data['train'].edge_index)
            edges = data['train'].edge_label_index.t()[edge_id].T
            labels = data['train'].edge_label[edge_id].reshape(-1, 1)

            train_loss = model.loss(node_embed, edges, labels)
            train_loss.backward()

            optimizer.step()

        if e % 5 == 0:
            with torch.no_grad():
                model.eval()
                # node_embed = model(data['val'].x, data['train'].adj)
                node_embed = model(data['val'].x, data['val'].edge_index)
                edges = data['val'].edge_label_index
                labels = data['val'].edge_label.reshape(-1, 1)

                preds = model.link_pred(node_embed[edges[0]], node_embed[edges[1]]) >= 0.5

                val_loss = model.loss(node_embed, edges, labels)
                p, r, f, _ = precision_recall_fscore_support(
                    labels.cpu().numpy(), preds.cpu().numpy(), average='macro'
                )
                print(f'VAL - Epoch {e},  loss {train_loss} VAL Loss {val_loss}, Precision {p}, Recall {r}, F-score {f}')
                if (p_+r_+f_)/3 < (p+r+f)/3:
                    torch.save(model.state_dict(), log_file)
                    p_, r_, f_ = p, r, f

    # explanation step
    explainer = GCNSyntheticPerturb(
        data['n_features'], explainer_args.hidden1, explainer_args.hidden2, 1, data['test'].edge_index, data['n_nodes']
    )
    explainer.to(explainer_args.device)
    # Copy GAE's parameters to the explainer
    explainer.load_state_dict(model.state_dict(), strict=False)
    for name, param in explainer.named_parameters():
        if name.endswith("weight") or name.endswith("bias"):
            param.requires_grad = False
    explainer_optimizer = torch.optim.Adam(explainer.parameters(), lr=0.01)
    test_labels = data['test'].edge_label

    if explainer_args.expl_type == 'CF':
        expl_train_labels = (~(data['test'].edge_label > 0)).float()
    else:
        expl_train_labels = data['test'].edge_label
    explanations = []
    expls_preds = []
    node_embeds = model(data['test'].x, data['test'].edge_index)
    predicted_edge_labels = model.link_pred(node_embeds[data['test'].edge_index[0]], node_embeds[data['test'].edge_index[1]]) >= 0.5

    for i, edge_id in enumerate(data['test'].edge_label_index.t()[:1]):

        for j in range(300):

            explainer_optimizer.zero_grad()
            node_embed = explainer(data['test'].x, data['test'].edge_index)
            loss_total, pred_label = explainer.loss(node_embed, edge_id, expl_train_labels[i])
            expl, preds = explainer.get_explanation(node_embed, data['test'].edge_index)

            if explainer_args.expl_type == 'PT' or explainer_args.expl_type == 'EXE':
                if pred_label == test_labels[i] and expl.shape != data['test'].edge_index:
                    explanations.append(expl)
                    expls_preds.append(preds.detach().cpu().numpy())
            if explainer_args.expl_type == 'CF':
                if pred_label != test_labels[i] and expl.shape != data['test'].edge_index.shape:
                    explanations.append(expl)
                    expls_preds.append(preds.detach().cpu().numpy())
            loss_total.backward()
            clip_grad_norm_(explainer.parameters(), 2.0)
            explainer_optimizer.step()

        graph_evaluation_metrics(data['test'].edge_index, predicted_edge_labels, explanations, expls_preds, data['n_nodes'])




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda', help='torch device.')
    parser.add_argument('--bb_epochs', type=int, default=500, help='Number of epochs to train the ')
    parser.add_argument('--explainer_epochs', type=int, default=300, help='Number of epochs to train the ')
    parser.add_argument('--hidden1', type=int, default=300, help='Number of units in hidden layer 1.')
    parser.add_argument('--hidden2', type=int, default=100, help='Number of units in hidden layer 1.')
    parser.add_argument('--lr', type=float, default=0.009, help='Initial learning rate.')
    parser.add_argument('--bb_training_mode', type=bool, default=True, help='Initial learning rate.')
    parser.add_argument('--expl_type', type=str, default='CF', help='Type of explanation.')

    parser.add_argument('--cf_lr', type=float, default=0.009, help='CF-explainer learning rate.')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--cf_optimizer', type=str, default='Adam', help='Dropout rate (1 - keep probability).')
    parser.add_argument('--dataset_str', type=str, default='citeseer', help='type of dataset.')
    parser.add_argument('--dataset_func', type=str, default='Planetoid', help='type of dataset.')
    parser.add_argument('--beta', type=float, default=0.1, help='beta variable')
    parser.add_argument('--include_ae', type=bool, default=True, help='Including AutoEncoder reconstruction loss')
    parser.add_argument('--model_dir', type=str, default='\\saved_models\\model.pt', help='Result directory')
    parser.add_argument('--algorithm', type=str, default='loss_PN_AE', help='Result directory')
    parser.add_argument('--graph_result_name', type=str, default='loss_PN_AE', help='Result name')
    parser.add_argument('--cf_train_loss', type=str, default='loss_PN_AE',
                        help='CF explainer loss function')
    parser.add_argument('--cf_expl', type=bool, default=True, help='CF explainer loss function')
    parser.add_argument('--n_momentum', type=float, default=0.5, help='Nesterov momentum')
    explainer_args = parser.parse_args()

    # algorithms = [
    #     'cfgnn', 'loss_PN_L1_L2',
    #     'loss_PN_AE_L1_L2_dist', 'loss_PN_AE_L1_L2', 'loss_PN_AE', 'loss_PN', 'loss_PN_dist'
    # ]
    # datasets = ['cora', 'citeseer', 'pubmed']

    # if os.listdir(f'{explainer_args.graph_result_dir}/'
    #               f'{explainer_args.dataset_str}/'
    #               f'cf_expl_{explainer_args.cf_expl}/'
    #               ).__contains__(explainer_args.algorithm) is False:
    #     os.mkdir(
    #         f'{explainer_args.graph_result_dir}/'
    #         f'{explainer_args.dataset_str}/'
    #         f'cf_expl_{explainer_args.cf_expl}/'
    #         f'{explainer_args.algorithm}'
    #     )
    main(explainer_args)
