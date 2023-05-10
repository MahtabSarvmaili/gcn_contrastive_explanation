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
from sklearn.metrics import precision_recall_fscore_support
from cf_explainer import CFExplainer
from gae.GAE import gae
from evaluation.evaluation import evaluate_cf_PN, evaluate_cf_PP, swap_edges
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

    for e in range(500):
        for edge_id in DataLoader(range(data['train'].edge_label_index.size(-1)), batch_size=256, shuffle=True):
            model.train()
            optimizer.zero_grad()
            # node_embed = model(data['train'].x, data['train'].adj)
            node_embed = model(data['train'].x, data['train'].edge_index)
            edges = data['train'].edge_label_index.t()[edge_id].T
            labels = data['train'].edge_label[edge_id].reshape(-1,1)

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
        data['n_features'], explainer_args.hidden1, explainer_args.hidden2, 1, data['train'].edge_index, data['n_nodes']
    )
    explainer.to(explainer_args.device)
    # Copy GAE's parameters to the explainer
    explainer.load_state_dict(model.state_dict(), strict=False)
    for name, param in explainer.named_parameters():
        if name.endswith("weight") or name.endswith("bias"):
            param.requires_grad = False
    explainer_optimizer = torch.optim.Adam(explainer.parameters(), lr=0.01)
    node_embed = model(data['test'].x, data['test'].edge_index)

    for i, edge_id in enumerate(data['test'].edge_label_index.t()):
        preds = model.link_pred(node_embed[edge_id[0]], node_embed[edge_id[1]], sigmoid=False)
        for j in range(explainer_args.explainer_epochs):
            explainer_optimizer.zero_grad()
            explainer.loss(data['test'].x)
        val_loss = model.loss(node_embed, edge_id, data['test'].edge_label)
        p, r, f, _ = precision_recall_fscore_support(
            labels.cpu().numpy(), preds.cpu().numpy(), average='macro'
        )
    # # link_logits = model.link_pred(z[])  # decode

    # link_labels = get_link_labels(data.train_pos_edge_index, neg_edge_index, device=gae_args.device)
    #
    # loss.backward()
    # optimizer.step()
    #
    # model = train(
    #     model=model,
    #     features=data['features'],
    #     train_adj=data['adj_norm'],
    #     labels=data['labels'],
    #     train_mask=data['train_mask'],
    #     optimizer=optimizer,
    #     epoch=explainer_args.bb_epochs,
    #     val_mask=data['val_mask'],
    #     dataset_name=explainer_args.dataset_str
    # )
    # model.eval()
    # output = model(data['features'], data['adj_norm'])
    # y_pred_orig = torch.argmax(output, dim=1)
    # print("test set y_true counts: {}".format(
    #     np.unique(data['labels'][data['test_mask']].cpu().detach().numpy(), return_counts=True)))
    # print(
    #     "test set y_pred_orig counts: {}".format(
    #         np.unique(y_pred_orig[data['test_mask']].cpu().detach().numpy(), return_counts=True)
    #     )
    # )
    # print("Training GNN is finished.")
    #
    # print("Training AE.")
    # graph_ae = gae(gae_args, data_AE)
    # print("Explanation step:")
    #
    # idx_test = np.arange(0, data['n_nodes'])[data['test_mask'].cpu()]
    # test_cf_examples = []
    # for i in idx_test:
    #     try:
    #         sub_adj, sub_feat, sub_labels, node_dict, sub_edge_index = get_neighbourhood(
    #             int(i), data['edge_index'],
    #             explainer_args.n_layers + 1,
    #             data['features'], output.argmax(dim=1)
    #         )
    #
    #         new_idx = node_dict[int(i)]
    #         sub_output = model(sub_feat, normalize_adj(sub_adj, explainer_args.device)).argmax(dim=1)
    #         # Check that original model gives same prediction on full graph and subgraph
    #         with torch.no_grad():
    #             print(f"Output original model, normalized - actual label: {data['labels'][i]}")
    #             print(f"Output original model, full adj: {output[i].argmax()}")
    #             print(
    #                 f"Output original model, not normalized - sub adj: {sub_output[new_idx]}"
    #             )
    #         # Need to instantitate new cf model every time because size of P changes based on size of sub_adj
    #         explainer = CFExplainer(
    #             model=model,
    #             graph_ae=graph_ae,
    #             sub_adj=sub_adj,
    #             sub_feat=sub_feat,
    #             n_hid=explainer_args.hidden,
    #             dropout=explainer_args.dropout,
    #             cf_optimizer=explainer_args.cf_optimizer,
    #             lr=explainer_args.cf_lr,
    #             n_momentum=explainer_args.n_momentum,
    #             sub_labels=sub_labels,
    #             y_pred_orig=sub_labels[new_idx],
    #             num_classes=data['num_classes'],
    #             beta=explainer_args.beta,
    #             device=explainer_args.device,
    #             cf_expl=explainer_args.cf_expl,
    #             algorithm=explainer_args.algorithm,
    #         )
    #         explainer.cf_model.cuda()
    #         cf_example = explainer.explain(
    #             node_idx=i,
    #             new_idx=new_idx,
    #             num_epochs=explainer_args.cf_epochs,
    #             path=f'{explainer_args.graph_result_dir}/'
    #             f'{explainer_args.dataset_str}/'
    #             f'cf_expl_{explainer_args.cf_expl}/'
    #             f'{explainer_args.algorithm}/'
    #             f'_{i}_loss_.png'
    #         )
    #         min_sum = 10000
    #         min_idx = 0
    #         for ii, x in enumerate(cf_example):
    #             if x[2].sum() < min_sum and x[2].sum()>0:
    #                 min_sum = x[2].sum()
    #                 min_idx = ii
    #         cf_example_p_list = []
    #         # stability evaluation
    #         sub_adj_p_list = swap_edges(sub_adj, sub_edge_index, 1)
    #         for sub_adj_p in sub_adj_p_list:
    #             explainer = CFExplainer(
    #                 model=model,
    #                 graph_ae=graph_ae,
    #                 sub_adj=sub_adj_p,
    #                 sub_feat=sub_feat,
    #                 n_hid=explainer_args.hidden,
    #                 dropout=explainer_args.dropout,
    #                 cf_optimizer=explainer_args.cf_optimizer,
    #                 lr=explainer_args.cf_lr,
    #                 n_momentum=explainer_args.n_momentum,
    #                 sub_labels=sub_labels,
    #                 y_pred_orig=sub_labels[new_idx],
    #                 num_classes=data['num_classes'],
    #                 beta=explainer_args.beta,
    #                 device=explainer_args.device,
    #                 cf_expl=explainer_args.cf_expl,
    #                 algorithm=explainer_args.algorithm,
    #             )
    #             explainer.cf_model.cuda()
    #             cf_example_p = explainer.explain(
    #                 node_idx=i,
    #                 new_idx=new_idx,
    #                 num_epochs=explainer_args.cf_epochs,
    #                 path=f'{explainer_args.graph_result_dir}/'
    #                 f'{explainer_args.dataset_str}/'
    #                 f'cf_expl_{explainer_args.cf_expl}/'
    #                 f'{explainer_args.algorithm}/'
    #                 f'_{i}_loss_.png'
    #             )
    #             cf_example_p_list.append(cf_example_p)
    #
    #         test_cf_examples.append(cf_example)
    #         if explainer_args.cf_expl is True:
    #             evaluate_cf_PN(explainer_args, model, sub_feat, sub_adj, sub_labels, sub_edge_index, new_idx, i, cf_example, cf_example_p_list)
    #         else:
    #             evaluate_cf_PP(explainer_args, model, sub_feat, sub_adj, sub_labels, sub_edge_index, new_idx, i, cf_example, cf_example_p_list)
    #         print('yes!')
    #
    #         torch.cuda.empty_cache()
    #         gc.collect()
    #     except:
    #         traceback.print_exc()
    #         pass


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda', help='torch device.')
    parser.add_argument('--bb_epochs', type=int, default=500, help='Number of epochs to train the ')
    parser.add_argument('--explainer_epochs', type=int, default=300, help='Number of epochs to train the ')
    parser.add_argument('--hidden1', type=int, default=100, help='Number of units in hidden layer 1.')
    parser.add_argument('--hidden2', type=int, default=50, help='Number of units in hidden layer 1.')
    parser.add_argument('--lr', type=float, default=0.009, help='Initial learning rate.')
    parser.add_argument('--bb_training_mode', type=bool, default=True, help='Initial learning rate.')

    parser.add_argument('--cf_lr', type=float, default=0.009, help='CF-explainer learning rate.')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--cf_optimizer', type=str, default='Adam', help='Dropout rate (1 - keep probability).')
    parser.add_argument('--dataset_str', type=str, default='cora', help='type of dataset.')
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
