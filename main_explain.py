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
from data.graph_utils import get_graph_data
from baselines.graph_baseline_explainer import gnnexplainer, pgexplainer

# torch.manual_seed(0)
# np.random.seed(0)

sys.path.append('../..')


def main(args):
    torch.cuda.empty_cache()
    data = load_graph_data_(args)

    re_edge_lists, re_graph_labels, re_edge_label_lists, re_node_label_lists = get_graph_data(
        os.getcwd()+'\\data'+f'\\{args.dataset_func}'+f'\\{args.dataset_str}'+f'\\raw', args.dataset_str
    )
    result_dir = os.getcwd()+f'{args.graph_result_dir}'+f'\\{args.dataset_str}'
    # data_AE = load_data_AE(explainer_args)
    # data =load_synthetic(gen_syn1, device=explainer_args.device)
    # data_AE = load_synthetic_AE(gen_syn1, device=explainer_args.device)
    model = GCN(data['n_features'], args.hidden, data['n_classes'])
    if args.device== 'cuda':
        model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()
    test_acc_prev = 0
    # manually shuffling the data loaders since it doesn't shuffle automatically
    data['train']._DataLoader__initialized=False
    data['test']._DataLoader__initialized=False
    for epoch in range(1, args.epochs):
        train(model, criterion, optimizer, data['train'])
        train_acc = test(model, data['train'])
        test_acc = test(model, data['test'])
        data['train'].dataset = data['train'].dataset.shuffle()
        data['test'].dataset = data['test'].dataset.shuffle()
        print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
        if test_acc > test_acc_prev:
            test_acc_prev = test_acc
            torch.save(model.state_dict(), result_dir+f"\\{args.dataset_str}_{args.expl_type}_model.pt")

    for dt_id, dt in enumerate(data['expl_tst_dt'].dataset[:50]):
        expl_preds = []
        explanations = []
        edge_preds = []
        print(f'Explanation for {dt} has started!')
        explainer = GCNPerturb(data['n_features'], args.hidden, data['n_classes'], dt.edge_index, dt.x.shape[0])
        explainer_optimizer = torch.optim.Adam(explainer.parameters(), lr=args.cf_lr)

        explainer.load_state_dict(
            torch.load(result_dir+f"\\{args.dataset_str}_{args.expl_type}_model.pt"),
            strict=False
        )
        explainer.to(args.device)
        for name, param in explainer.named_parameters():
            if name.endswith("weight") or name.endswith("bias"):
                param.requires_grad = False
        if args.expl_type == 'CF':
            y = (~(dt.y > 0)).to(torch.int64).reshape(-1,)
        else:
            y = dt.y.to(torch.int64).reshape(-1)

        for i in range(args.expl_epochs):
            loss, _ = explainer.loss(dt.x.to(torch.float), dt.edge_index, dt.batch, y)
            expl, edge_pred, pred_y = explainer.get_explanation(dt.x.to(torch.float), dt.edge_index, dt.batch)

            if args.expl_type == 'PT' or args.expl_type == 'EXE':
                if pred_y == dt.y and expl.shape != dt.edge_index.shape:
                    explanations.append(expl)
                    edge_preds.append(edge_pred.detach().cpu().numpy())
                    expl_preds.append(pred_y.detach().cpu().numpy())
            if args.expl_type == 'CF':
                if pred_y != dt.y \
                        and expl.shape != dt.edge_index.shape:
                    explanations.append(expl)
                    edge_preds.append(edge_pred.detach().cpu().numpy())
                    expl_preds.append(pred_y.detach().cpu().numpy())
            clip_grad_norm_(explainer.parameters(), 2.0)
            loss.backward()
            explainer_optimizer.step()
        print(f'Explanation has finished, number of generated explanations: {len(explanations)}')
        # 3447 - 622
        pg_mask = pgexplainer(data['train'], model, dt)
        gnn_mask = gnnexplainer(dt, model, None)
        print('Quantitative evaluation of explanations has started!')
        try:
            graph_evaluation_metrics(
                dt,
                np.array(re_edge_label_lists[data['indices'][data['split']+dt_id]]),
                explanations,
                edge_preds,
                args,
                result_dir,
                data['indices'][data['split']+dt_id],
                gnn_mask,
                pg_mask
            )
            labels = dt.x.argmax(dim=1).cpu().numpy()
            list_classes = list(range(dt.x.shape[1]))
            # viz_exp = PlotGraphExplanation(
            #     dt.edge_index, labels, dt.x.shape[0], list_classes, args.expl_type, args.dataset_str
            # )
            # if args.expl_type == 'CF':
            #     viz_exp.plot_cf(explanations, result_dir, dt_id)
            # else:
            #     # viz_exp.plot_pt(explanations, result_dir, dt_id)
        except:
            print(f"Error for {data['indices'][data['split']+dt_id]} data sample")
            continue

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda', help='torch device.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train the ')
    parser.add_argument('--expl_epochs', type=int, default=200, help='Number of epochs to train the ')
    parser.add_argument('--expl_type', type=str, default='PT', help='Type of explanation.')
    parser.add_argument('--hidden', type=int, default=100, help='Number of units in hidden layer 1.')
    parser.add_argument('--lr', type=float, default=0.009, help='Initial learning rate.')
    parser.add_argument('--cf_lr', type=float, default=0.01, help='CF-explainer learning rate.')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--cf_optimizer', type=str, default='Adam', help='Dropout rate (1 - keep probability).')
    parser.add_argument('--dataset_str', type=str, default='Mutagenicity', help='type of dataset.')
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
