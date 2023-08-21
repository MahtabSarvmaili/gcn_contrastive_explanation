import argparse
import traceback
import sys
import os
import platform
import torch
import numpy as np
from data.data_loader import load_graph_data_
from gnn_models.model import GCNGraph, train_graph_classifier, test_graph_classifier
from torch.nn.utils import clip_grad_norm_


from gnn_models.gcn_explainer import GCNPerturb
from evaluation.graph_explanation_evaluation import graph_evaluation_metrics
from evaluation.visualization import PlotGraphExplanation
from data.graph_utils import get_graph_data
from baselines.graph_baseline_explainer import gnnexplainer, pgexplainer
from utils import transform_address, influential_func
torch.manual_seed(0)
np.random.seed(0)

sys.path.append('../..')


def main(args):
    torch.cuda.empty_cache()
    data = load_graph_data_(args)

    org_edge_lists, org_graph_labels, org_edge_label_lists, org_node_label_lists = get_graph_data(
        transform_address(os.getcwd()+'\\data'+f'\\{args.dataset_func}'+f'\\{args.dataset_str}'+f'\\raw'),
        args.dataset_str
    )
    result_dir = transform_address(
        os.getcwd()+f'{args.graph_result_dir}'+f'\\{args.dataset_str}'+f'\\{args.expl_type}'
    )
    model = GCNGraph(data['n_features'], args.hidden, data['n_classes'])
    if args.device== 'cuda':
        model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-5)
    criterion = torch.nn.CrossEntropyLoss(weight=data['weight'])
    test_acc_prev = 0
    # manually shuffling the data loaders since it doesn't shuffle automatically
    if args.dataset_func == 'TUDataset':
        data['train']._DataLoader__initialized=False
        data['test']._DataLoader__initialized=False

    for epoch in range(1, args.epochs):
        train_graph_classifier(model, criterion, optimizer, data['train'])
        train_acc = test_graph_classifier(model, data['train'])
        test_acc = test_graph_classifier(model, data['test'])
        if args.dataset_func == 'TUDataset':
            data['train'].dataset = data['train'].dataset.shuffle()
            data['test'].dataset = data['test'].dataset.shuffle()
        print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
        if test_acc > test_acc_prev and epoch > 5:
            test_acc_prev = test_acc
            torch.save(
                model.state_dict(),
                transform_address(result_dir+f"\\{args.dataset_str}_{args.expl_type}_model.pt")
            )

    for dt_id, dt in enumerate(data['expl_tst_dt'].dataset[:200]):
        expl_preds = []
        explanations = []
        edge_preds = []
        print(f"Explanation for {data['indices'][data['split']+dt_id]} has started!")
        explainer = GCNPerturb(data['n_features'], args.hidden, data['n_classes'], dt.edge_index, args.expl_type)
        explainer_optimizer = torch.optim.Adam(explainer.parameters(), lr=args.cf_lr)
        explainer.load_state_dict(
            torch.load(transform_address(result_dir+f"\\{args.dataset_str}_{args.expl_type}_model.pt")),
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
            if args.expl_type in ['CF', 'PT', 'EXE']:
                loss = explainer.loss(dt.x.to(torch.float), dt.edge_index, dt.batch, y)
            if args.expl_type=='CFGNN':
                loss = explainer.loss_cfgnn(dt.x.to(torch.float), dt.edge_index, dt.batch, dt.y)

            expl, edge_pred, pred_y = explainer.get_explanation(dt.x.to(torch.float), dt.edge_index, dt.batch)

            if args.expl_type in ['EXE', 'PT']:
                if pred_y == dt.y and expl.shape != dt.edge_index.shape:
                    explanations.append(expl)
                    edge_preds.append(edge_pred)
                    expl_preds.append(pred_y.detach().cpu().numpy())

            if args.expl_type in ['CF', 'CFGNN']:
                if pred_y != dt.y \
                        and expl.shape != dt.edge_index.shape:
                    explanations.append(expl)
                    edge_preds.append(edge_pred)
                    expl_preds.append(pred_y.detach().cpu().numpy())
            clip_grad_norm_(explainer.parameters(), 2.0)
            loss.backward()
            explainer_optimizer.step()
        print(f'Explanation has finished, number of generated explanations: {len(explanations)}')

        pg_mask = pgexplainer(data['train'], model, dt)
        gnn_mask = gnnexplainer(dt, model, None)
        actual_expls = None
        if org_edge_label_lists is not None:
            actual_dt = np.int32(np.array(org_edge_label_lists[data['indices'][data['split'] + dt_id]]) > 0)
            actual_idxs = np.array(org_edge_label_lists[data['indices'][data['split'] + dt_id]]) > 0
            actual_expls = dt.edge_index[:, actual_idxs]
        else:
            actual_dt = None
        print(f'Quantitative evaluation:')
        try:
            if args.dataset_func =='TUDataset':
                labels = dt.x.argmax(dim=1).cpu().numpy()
                list_classes = list(range(dt.x.shape[1]))
            if args.dataset_func =='MoleculeNet':
                labels = dt.x[:, 0].cpu().numpy()
                list_classes = dt.x[:, 0].unique().cpu().numpy()

            expl_plot = PlotGraphExplanation(
                dt.edge_index, labels, dt.x.shape[0], list_classes, args.expl_type, args.dataset_str
            )
            # plotting the ground truth explanation
            if actual_expls is not None:
                expl_plot.plot_pr_edges(
                    exp_edge_index=actual_expls,
                    res_dir=result_dir,
                    dt_id=data['indices'][data['split'] + dt_id],
                    f_name='actual_expl',
                    plt_title='Actual Explanation'
                )
            if args.expl_type == 'PT':
                graph_evaluation_metrics(
                    dt,
                    explanations,
                    edge_preds,
                    args,
                    result_dir,
                    data['indices'][data['split']+dt_id],
                    model,
                    actual_dt,
                    gnn_mask,
                    pg_mask,
                    expl_plot.plot_pr_edges
                )

            else:
                graph_evaluation_metrics(
                    dt,
                    explanations,
                    edge_preds,
                    args,
                    result_dir,
                    data['indices'][data['split'] + dt_id],
                    model,
                    actual_dt,
                    gnn_mask,
                    pg_mask,
                    expl_plot.plot_del_edges,
                )
        except:
            print(f"Error for {data['indices'][data['split']+dt_id]} data sample")
            print(traceback.format_exc())
            continue


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda', help='torch device.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train the black box model')
    parser.add_argument('--expl_epochs', type=int, default=300, help='Number of epochs to train explainer.')
    parser.add_argument('--expl_type', type=str, default='CFGNN', help='Type of explanation: PT, CF, EXE, CFGNN')
    parser.add_argument('--hidden', type=int, default=100, help='Number of units in hidden layer 1.')
    parser.add_argument('--lr', type=float, default=0.009, help='Initial learning rate.')
    parser.add_argument('--cf_lr', type=float, default=0.01, help='CF-explainer learning rate.')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--cf_optimizer', type=str, default='Adam', help='Dropout rate (1 - keep probability).')
    parser.add_argument('--dataset_str', type=str, default='bbbp', help='type of dataset.')
    parser.add_argument('--dataset_func', type=str, default='MoleculeNet', help='type of dataset.')
    parser.add_argument('--beta', type=float, default=0.1, help='beta variable')
    parser.add_argument('--include_ae', type=bool, default=True, help='Including AutoEncoder reconstruction loss')
    parser.add_argument('--graph_result_dir', type=str, default='\\results', help='Result directory')
    parser.add_argument('--cf_expl', type=bool, default=True, help='CF explainer loss function')
    parser.add_argument('--n_momentum', type=float, default=0.5, help='Nesterov momentum')
    args = parser.parse_args()

    if os.listdir(
            transform_address(os.getcwd()+f'{args.graph_result_dir}')
    ).__contains__(args.dataset_str) is False:
        os.mkdir(
            transform_address(os.getcwd()+f'{args.graph_result_dir}'+f'\\{args.dataset_str}')
        )
    if os.listdir(
            transform_address(os.getcwd()+f'{args.graph_result_dir}'f'\\{args.dataset_str}')
    ).__contains__(args.expl_type) is False:
        os.mkdir(
            transform_address(os.getcwd()+f'{args.graph_result_dir}'+f'\\{args.dataset_str}'+f'\\{args.expl_type}')
        )
    main(args)
