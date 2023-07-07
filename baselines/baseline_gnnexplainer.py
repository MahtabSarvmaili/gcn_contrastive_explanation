import argparse
import sys
import traceback
import torch
import numpy as np
from data.data_loader import load_data
from utils import get_neighbourhood
from gnn_models.model import GCN_dep, train_graph_classifier
from gnnexplainer import GNNExplainer
from visualization import plot_graph
from evaluation.evaluation import insertion, deletion
torch.manual_seed(0)
np.random.seed(0)

sys.path.append('../..')


def main(explainer_args):
    data = load_data(explainer_args)
    model = GCN_dep(
        nfeat=data['feat_dim'],
        nhid=explainer_args.hidden,
        nout=explainer_args.hidden,
        nclasses=data['num_classes'],
        dropout=explainer_args.dropout
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=explainer_args.lr, weight_decay=5e-4)
    './re'
    if explainer_args.device=='cuda':
        model = model.cuda()

    train_graph_classifier(
        model=model,
        features=data['features'],
        train_adj=data['adj_norm'],
        labels=data['labels'],
        train_mask=data['train_mask'],
        optimizer=optimizer,
        epoch=explainer_args.bb_epochs,
        val_mask=data['val_mask'],
        dataset_name=explainer_args.dataset_str
    )
    model.eval()
    explainer = GNNExplainer(model, explainer_args.cf_epochs, explainer_args.cf_lr, explainer_args.n_layers + 1)
    idx_test = np.arange(0, data['n_nodes'])[data['test_mask'].cpu()]
    for i in idx_test[:10]:
        try:
            output = model(data['features'], data['adj_norm'])
            sub_adj, sub_feat, sub_labels, node_dict, sub_edge_index = get_neighbourhood(
                int(i),
                data['edge_index'],
                explainer_args.n_layers+1,
                data['features'],
                output.argmax(dim=1),
            )
            new_idx = node_dict[int(i)]
            sub_output = model(sub_feat, sub_adj).argmax(dim=1)
            node_mask, edge_mask, labels = explainer.explain_node(
                i, data, sub_adj, sub_feat, sub_labels, node_dict, sub_edge_index
            )
            if edge_mask.sum() > 0 or node_mask.sum()>0:
                plotting_graph = plot_graph(
                    sub_adj.cpu().numpy(),
                    new_idx,
                    f'../{explainer_args.graph_result_dir}/'
                    f'{explainer_args.dataset_str}/'
                    f'edge_addition_{explainer_args.edge_addition}/'
                    f'{explainer_args.algorithm}/'
                    f'_{i}_sub_adj_{explainer_args.graph_result_name}.png'
                )
                plotting_graph.plot_org_graph(
                    sub_adj.cpu().numpy(),
                    sub_labels.cpu().numpy(),
                    new_idx,
                    f'../{explainer_args.graph_result_dir}/'
                    f'{explainer_args.dataset_str}/'
                    f'edge_addition_{explainer_args.edge_addition}/'
                    f'{explainer_args.algorithm}/'
                    f'_{i}_sub_adj_{explainer_args.graph_result_name}.png',
                    sub_edge_index.t().cpu().numpy()
                )
                del_edge_adj = 1 * (edge_mask < sub_adj).cpu().numpy()
                plotting_graph.plot_cf_graph(
                    1*edge_mask,
                    del_edge_adj,
                    labels.cpu().numpy(),
                    new_idx,
                    f'../{explainer_args.graph_result_dir}/'
                    f'{explainer_args.dataset_str}/'
                    f'edge_addition_{explainer_args.edge_addition}/'
                    f'{explainer_args.algorithm}/'
                    f'_{i}_counter_factual'
                    f'{explainer_args.graph_result_name}__removed_edges__.png',
                )
                insertion(
                    model,
                    sub_feat,
                    1*edge_mask.cpu().numpy(),
                    del_edge_adj,
                    labels.cpu().numpy(),
                    new_idx,
                    name=f'../{explainer_args.graph_result_dir}/'
                         f'{explainer_args.dataset_str}/'
                         f'edge_addition_{explainer_args.edge_addition}/'
                         f'{explainer_args.algorithm}/'
                         f'_{i}_counter_factual_'
                         f'{explainer_args.graph_result_name}__insertion__',
                )
                deletion(
                    model,
                    sub_feat,
                    sub_adj.cpu().numpy(),
                    del_edge_adj,
                    sub_output.cpu().numpy(),
                    new_idx,
                    name=f'../{explainer_args.graph_result_dir}/'
                         f'{explainer_args.dataset_str}/'
                         f'edge_addition_{explainer_args.edge_addition}/'
                         f'{explainer_args.algorithm}/'
                         f'_{i}_counter_factual_'
                         f'{explainer_args.graph_result_name}__deletion__',
                )

            # fidelity_size_sparsity(
            #     model,
            #     sub_feat,
            #     [sub_adj],
            #     1*edge_mask.cpu().numpy(),
            #     f'../{explainer_args.graph_result_dir}/'
            #     f'{explainer_args.dataset_str}/'
            #     f'edge_addition_{explainer_args.edge_addition}/'
            #     f'{explainer_args.algorithm}/'
            #     f'_{i}_counter_factual_{explainer_args.graph_result_name}_sub_graph_'
            # )
        except:
            traceback.print_exc()
            pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda', help='torch device.')
    parser.add_argument('--bb-epochs', type=int, default=500, help='Number of epochs to train the ')
    parser.add_argument('--cf-epochs', type=int, default=100, help='Number of epochs to train the ')
    parser.add_argument('--inputdim', type=int, default=10, help='Input dimension')
    parser.add_argument('--hidden', type=int, default=20, help='Number of units in hidden layer 1.')
    parser.add_argument('--n-layers', type=int, default=3, help='Number of units in hidden layer 1.')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
    parser.add_argument('--cf-lr', type=float, default=0.009, help='CF-explainer learning rate.')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--dataset-str', type=str, default='cora', help='type of dataset.')
    parser.add_argument('--dataset-func', type=str, default='Planetoid', help='type of dataset.')
    parser.add_argument('--beta', type=float, default=0.5, help='beta variable')
    parser.add_argument('--include_ae', type=bool, default=True, help='Including AutoEncoder reconstruction loss')
    parser.add_argument('--edge-addition', type=bool, default=False, help='CF edge_addition')
    parser.add_argument('--algorithm', type=str, default='gnnexplainer', help='Result directory')
    parser.add_argument('--graph-result-dir', type=str, default='./results', help='Result directory')
    parser.add_argument('--graph-result-name', type=str, default='gnnexplainer', help='Result name')
    explainer_args = parser.parse_args()
    main(explainer_args)