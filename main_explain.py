import argparse
import gc
import sys
import os
import traceback
import torch
import numpy as np
from data.data_loader import load_data, load_graph_data_, load_data_AE
from utils import normalize_adj, get_neighbourhood, perturb_sub_adjacency
from model import GCN, train, test
from torch.nn.utils import clip_grad_norm_

from gnn_models.gcn_explainer import GCNPerturb
from evaluation.graph_explanation_evaluation import graph_evaluation_metrics, plot_explanation_subgraph
from cf_explainer import CFExplainer
from gae.GAE import gae
from evaluation.evaluation import evaluate_cf_PN, evaluate_cf_PP, swap_edges


# torch.manual_seed(0)
# np.random.seed(0)

sys.path.append('../..')


def main(args):
    torch.cuda.empty_cache()
    data = load_graph_data_(args)
    result_dir = os.getcwd()+f'{args.graph_result_dir}'+f'\\{args.dataset_str}'
    # data_AE = load_data_AE(explainer_args)
    # data =load_synthetic(gen_syn1, device=explainer_args.device)
    # data_AE = load_synthetic_AE(gen_syn1, device=explainer_args.device)
    model = GCN(data['n_features'], args.hidden, data['n_classes'])
    if args.device== 'cuda':
        model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(1, 171):
        train(model, criterion, optimizer, data['train'])
        train_acc = test(model, data['test'])
        test_acc = test(model, data['test'])
        print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')

    for dt in data['test'].dataset[:1]:
        expl_preds = []
        explanations = []
        print(f'Explanation for {dt} has started!')
        explainer = GCNPerturb(data['n_features'], args.hidden, data['n_classes'], dt.edge_index, dt.x.shape[0])
        explainer_optimizer = torch.optim.Adam(explainer.parameters(), lr=args.cf_lr)
        explainer.load_state_dict(model.state_dict(), strict=False)
        explainer.to(args.device)
        for name, param in explainer.named_parameters():
            if name.endswith("weight") or name.endswith("bias"):
                param.requires_grad = False
        if args.expl_type == 'CF':
            y = (~(dt.y > 0)).to(torch.int64)
        else:
            y = dt.y

        for i in range(args.expl_epochs):
            loss, _ = explainer.loss(dt.x, dt.edge_index, dt.batch, y)
            expl, pred_y = explainer.get_explanation(dt.x, dt.edge_index, dt.batch)

            if args.expl_type == 'PT' or args.expl_type == 'EXE':
                if pred_y == dt.y and expl.shape != data['test'].edge_index.shape:
                    explanations.append(expl)
                    expl_preds.append(pred_y.detach().cpu().numpy())
            if args.expl_type == 'CF':
                if pred_y != dt.y \
                        and expl.shape != dt.edge_index.shape:
                    explanations.append(expl)
                    expl_preds.append(pred_y.detach().cpu().numpy())
            clip_grad_norm_(explainer.parameters(), 2.0)
            loss.backward()
            explainer_optimizer.step()
        print(f'Explanation has finished, number of generated explanations: {len(explanations)}')
        graph_evaluation_metrics(dt, explanations, result_dir, args)
        for c, exp in enumerate(explanations):
            if not expl.shape.__contains__(0):
                plot_explanation_subgraph(
                    dt.edge_index, exp, dt.x.argmax(dim=1), dt.x.shape[0], dt.x.shape[1],
                    result_dir + f'\\{args.dataset_str}_{c}_{exp.cpu().numpy().shape}.png',
                    result_dir + f'\\{args.dataset_str}_{c}_{exp.cpu().numpy().shape}.png'
                )
    #
    # idx_test = np.arange(0, data['n_nodes'])[data['test_mask'].cpu()]
    # test_cf_examples = []
    # for i in idx_test[:1]:
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
    parser.add_argument('--expl_epochs', type=int, default=300, help='Number of epochs to train the ')
    parser.add_argument('--expl_type', type=str, default='CF', help='Type of explanation.')
    parser.add_argument('--hidden', type=int, default=64, help='Number of units in hidden layer 1.')
    parser.add_argument('--lr', type=float, default=0.009, help='Initial learning rate.')
    parser.add_argument('--cf_lr', type=float, default=0.009, help='CF-explainer learning rate.')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--cf_optimizer', type=str, default='Adam', help='Dropout rate (1 - keep probability).')
    parser.add_argument('--dataset_str', type=str, default='MUTAG', help='type of dataset.')
    parser.add_argument('--dataset_func', type=str, default='TUDataset', help='type of dataset.')
    parser.add_argument('--beta', type=float, default=0.1, help='beta variable')
    parser.add_argument('--include_ae', type=bool, default=True, help='Including AutoEncoder reconstruction loss')
    parser.add_argument('--graph_result_dir', type=str, default='\\results', help='Result directory')
    parser.add_argument('--cf_expl', type=bool, default=True, help='CF explainer loss function')
    parser.add_argument('--n_momentum', type=float, default=0.5, help='Nesterov momentum')
    args = parser.parse_args()

    # algorithms = [
    #     'cfgnn', 'loss_PN_L1_L2',
    #     'loss_PN_AE_L1_L2_dist', 'loss_PN_AE_L1_L2', 'loss_PN_AE', 'loss_PN', 'loss_PN_dist'
    # ]
    # datasets = ['cora', 'citeseer', 'pubmed']
    #
    if os.listdir(os.getcwd()+f'{args.graph_result_dir}').__contains__(args.dataset_str) is False:
        os.mkdir(os.getcwd()+f'{args.graph_result_dir}'+f'\\{args.dataset_str}', )
    #     os.mkdir(
    #         f'{explainer_args.graph_result_dir}/'
    #         f'{explainer_args.dataset_str}/'
    #         f'cf_expl_{explainer_args.cf_expl}/'
    #         f'{explainer_args.algorithm}'
    #     )
    main(args)
