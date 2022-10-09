import argparse
import gc
import sys
import os
import traceback
import torch
import numpy as np
from data.data_loader import load_data, load_synthetic, load_synthetic_AE, load_data_AE
from data.gengraph import gen_syn1, gen_syn2, gen_syn3, gen_syn4
from utils import normalize_adj, get_neighbourhood, perturb_sub_adjacency
from model import GCN, train
from cf_explainer import CFExplainer
from gae.GAE import gae
from visualization import plot_graph, plot_centrality
from evaluation.evaluation import evaluate_cf_PN, evaluate_cf_PP
from torch_geometric.utils import dense_to_sparse

torch.manual_seed(0)
np.random.seed(0)

sys.path.append('../..')


def main(gae_args, explainer_args):
    torch.cuda.empty_cache()
    data = load_data(explainer_args)
    data_AE = load_data_AE(explainer_args)
    # data =load_synthetic(gen_syn2, device=explainer_args.device)
    # data_AE = load_synthetic_AE(gen_syn2, device=explainer_args.device)
    AE_threshold = {
        'gen_syn1': 0.5,
        'gen_syn2': 0.65,
        'gen_syn3': 0.6,
        'gen_syn4': 0.62,
        'cora': 0.5,
        'citeseer': 0.5,
        'pubmed': 0.5,
    }
    model = GCN(
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

    model = train(
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
    output = model(data['features'], data['adj_norm'])
    y_pred_orig = torch.argmax(output, dim=1)
    print("test set y_true counts: {}".format(
        np.unique(data['labels'][data['test_mask']].cpu().detach().numpy(), return_counts=True)))
    print(
        "test set y_pred_orig counts: {}".format(
            np.unique(y_pred_orig[data['test_mask']].cpu().detach().numpy(), return_counts=True)
        )
    )
    print("Training GNN is finished.")

    print("Training AE.")
    graph_ae = gae(gae_args, data_AE)
    print("Explanation step:")

    idx_test = np.arange(0, data['n_nodes'])[data['test_mask'].cpu()]
    test_cf_examples = []
    for i in idx_test[9:10]:
        try:
            sub_adj, sub_feat, sub_labels, node_dict, sub_edge_index = get_neighbourhood(
                int(i), data['edge_index'],
                explainer_args.n_layers + 1,
                data['features'], output.argmax(dim=1)
            )
            new_idx = node_dict[int(i)]
            sub_output = model(sub_feat, normalize_adj(sub_adj, explainer_args.device)).argmax(dim=1)
            # Check that original model gives same prediction on full graph and subgraph
            with torch.no_grad():
                print(f"Output original model, normalized - actual label: {data['labels'][i]}")
                print(f"Output original model, full adj: {output[i].argmax()}")
                print(
                    f"Output original model, not normalized - sub adj: {sub_output[new_idx]}"
                )
            # Need to instantitate new cf model every time because size of P changes based on size of sub_adj
            explainer = CFExplainer(
                model=model,
                graph_ae=graph_ae,
                sub_adj=sub_adj,
                sub_feat=sub_feat,
                n_hid=explainer_args.hidden,
                dropout=explainer_args.dropout,
                cf_optimizer=explainer_args.cf_optimizer,
                lr=explainer_args.cf_lr,
                n_momentum=explainer_args.n_momentum,
                sub_labels=sub_labels,
                y_pred_orig=sub_labels[new_idx],
                num_classes=data['num_classes'],
                beta=explainer_args.beta,
                device=explainer_args.device,
                AE_threshold=AE_threshold[explainer_args.dataset_str],
                PN_PP=explainer_args.PN_PP,
                cf_expl=explainer_args.cf_expl,
                algorithm=explainer_args.algorithm,
                edge_addition=explainer_args.edge_addition
            )
            explainer.cf_model.cuda()
            cf_example = explainer.explain(
                node_idx=i,
                new_idx=new_idx,
                num_epochs=explainer_args.cf_epochs,
                path=f'{explainer_args.graph_result_dir}/'
                f'{explainer_args.dataset_str}/'
                f'cf_expl_{explainer_args.cf_expl}/'
                f'pn_pp_{explainer_args.PN_PP}/'
                f'{explainer_args.algorithm}/'
                f'_{i}_loss_.png'
            )
            # stability evaluation + applying perturbation
            # the choice of perturbation depends on the type of features
            psub_adj_list = perturb_sub_adjacency(sub_adj.shape[0], sub_adj, [0.5, 0.4, 0.3, 0.2, 0.1])
            psub_adj = psub_adj_list[-1] * sub_adj
            explainer = CFExplainer(
                model=model,
                graph_ae=graph_ae,
                sub_adj=psub_adj,
                sub_feat=sub_feat,
                n_hid=explainer_args.hidden,
                dropout=explainer_args.dropout,
                cf_optimizer=explainer_args.cf_optimizer,
                lr=explainer_args.cf_lr,
                n_momentum=explainer_args.n_momentum,
                sub_labels=sub_labels,
                y_pred_orig=sub_labels[new_idx],
                num_classes=data['num_classes'],
                beta=explainer_args.beta,
                device=explainer_args.device,
                AE_threshold=AE_threshold[explainer_args.dataset_str],
                PN_PP=explainer_args.PN_PP,
                cf_expl=explainer_args.cf_expl,
                algorithm=explainer_args.algorithm,
                edge_addition=explainer_args.edge_addition
            )
            explainer.cf_model.cuda()
            pcf_example = explainer.explain(
                node_idx=i,
                new_idx=new_idx,
                num_epochs=explainer_args.cf_epochs,
                path=f'{explainer_args.graph_result_dir}/'
                     f'{explainer_args.dataset_str}/'
                     f'cf_expl_{explainer_args.cf_expl}/'
                     f'pn_pp_{explainer_args.PN_PP}/'
                     f'{explainer_args.algorithm}/'
                     f'_{i}_loss_.png'
            )

            test_cf_examples.append(cf_example)
            if explainer_args.PN_PP == "PN":
                evaluate_cf_PN(explainer_args, model, sub_feat, sub_adj, sub_labels, sub_edge_index, new_idx, i, cf_example, pcf_example)
            else:
                evaluate_cf_PP(explainer_args, model, sub_feat, sub_adj, sub_labels, sub_edge_index, new_idx, i, cf_example, pcf_example)
            print('yes!')

            torch.cuda.empty_cache()
            gc.collect()
        except:
            traceback.print_exc()
            pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda', help='torch device.')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
    parser.add_argument('--hidden1', type=int, default=32, help='Number of units in hidden layer 1.')
    parser.add_argument('--hidden2', type=int, default=16, help='Number of units in hidden layer 2.')
    parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--dataset_str', type=str, default='cora', help='type of dataset.')
    gae_args = parser.parse_args()

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda', help='torch device.')
    parser.add_argument('--bb_epochs', type=int, default=500, help='Number of epochs to train the ')
    parser.add_argument('--cf_epochs', type=int, default=300, help='Number of epochs to train the ')
    parser.add_argument('--inputdim', type=int, default=10, help='Input dimension')
    parser.add_argument('--hidden', type=int, default=20, help='Number of units in hidden layer 1.')
    parser.add_argument('--n_layers', type=int, default=3, help='Number of units in hidden layer 1.')
    parser.add_argument('--lr', type=float, default=0.009, help='Initial learning rate.')
    parser.add_argument('--cf_lr', type=float, default=0.009, help='CF-explainer learning rate.')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--cf_optimizer', type=str, default='Adam', help='Dropout rate (1 - keep probability).')
    parser.add_argument('--dataset_str', type=str, default='cora', help='type of dataset.')
    parser.add_argument('--dataset_func', type=str, default='Planetoid', help='type of dataset.')
    parser.add_argument('--beta', type=float, default=0.1, help='beta variable')
    parser.add_argument('--include_ae', type=bool, default=True, help='Including AutoEncoder reconstruction loss')
    parser.add_argument('--edge-addition', type=bool, default=False, help='CF edge_addition')
    parser.add_argument('--graph_result_dir', type=str, default='./results', help='Result directory')
    parser.add_argument('--algorithm', type=str, default='loss_PN_AE_L1_L2', help='Result directory')
    parser.add_argument('--graph_result_name', type=str, default='loss_PN_AE_L1_L2', help='Result name')
    parser.add_argument('--cf_train_loss', type=str, default='loss_PN_AE_L1_L2',
                        help='CF explainer loss function')
    parser.add_argument('--PN_PP', type=str, default="PN", help='CF explainer loss function')
    parser.add_argument('--cf_expl', type=bool, default=True, help='CF explainer loss function')
    parser.add_argument('--n_momentum', type=float, default=0.5, help='Nesterov momentum')
    explainer_args = parser.parse_args()

    # algorithms = [
    #     'cfgnn', 'loss_PN_L1_L2',
    #     'loss_PN_AE_L1_L2_dist', 'loss_PN_AE_L1_L2', 'loss_PN_AE', 'loss_PN', 'loss_PN_dist'
    # ]
    # datasets = ['cora', 'citeseer', 'pubmed']

    if os.listdir(f'{explainer_args.graph_result_dir}/'
                  f'{explainer_args.dataset_str}/'
                  f'cf_expl_{explainer_args.cf_expl}/'
                  f'pn_pp_{explainer_args.PN_PP}/'
                  ).__contains__(explainer_args.algorithm) is False:
        os.mkdir(
            f'{explainer_args.graph_result_dir}/'
            f'{explainer_args.dataset_str}/'
            f'cf_expl_{explainer_args.cf_expl}/'
            f'pn_pp_{explainer_args.PN_PP}/'
            f'{explainer_args.algorithm}'
        )
    main(gae_args, explainer_args)
