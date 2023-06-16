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
from evaluation.graph_explanation_evaluation import graph_evaluation_metrics
from evaluation.visualization import PlotGraphExplanation
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
        train_acc = test(model, data['train'])
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
                if pred_y == dt.y and expl.shape != dt.edge_index.shape:
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
        graph_evaluation_metrics(dt, explanations, args, result_dir)
        viz_exp = PlotGraphExplanation(
            dt.edge_index, dt.x.argmax(dim=1), dt.x.shape[0], dt.x.shape[1], args.expl_type, args.dataset_str
        )
        if args.expl_type == 'CF':
            viz_exp.plot_cf(explanations, result_dir)
        else:
            viz_exp.plot_pt(explanations, result_dir)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda', help='torch device.')
    parser.add_argument('--bb_epochs', type=int, default=500, help='Number of epochs to train the ')
    parser.add_argument('--expl_epochs', type=int, default=300, help='Number of epochs to train the ')
    parser.add_argument('--expl_type', type=str, default='PT', help='Type of explanation.')
    parser.add_argument('--hidden', type=int, default=64, help='Number of units in hidden layer 1.')
    parser.add_argument('--lr', type=float, default=0.009, help='Initial learning rate.')
    parser.add_argument('--cf_lr', type=float, default=0.009, help='CF-explainer learning rate.')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--cf_optimizer', type=str, default='Adam', help='Dropout rate (1 - keep probability).')
    parser.add_argument('--dataset_str', type=str, default='AIDS', help='type of dataset.')
    parser.add_argument('--dataset_func', type=str, default='TUDataset', help='type of dataset.')
    parser.add_argument('--beta', type=float, default=0.1, help='beta variable')
    parser.add_argument('--include_ae', type=bool, default=True, help='Including AutoEncoder reconstruction loss')
    parser.add_argument('--graph_result_dir', type=str, default='\\results', help='Result directory')
    parser.add_argument('--cf_expl', type=bool, default=True, help='CF explainer loss function')
    parser.add_argument('--n_momentum', type=float, default=0.5, help='Nesterov momentum')
    args = parser.parse_args()

    if os.listdir(os.getcwd()+f'{args.graph_result_dir}').__contains__(args.dataset_str) is False:
        os.mkdir(os.getcwd()+f'{args.graph_result_dir}'+f'\\{args.dataset_str}', )
    main(args)
