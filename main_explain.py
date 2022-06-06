import argparse
import sys

sys.path.append('../..')
import torch
import numpy as np
from torch_geometric.utils import  dense_to_sparse
from data.data_loader import load_data, load_synthetic, load_synthetic_AE, load_data_AE
from data.gengraph import gen_syn1, gen_syn2, gen_syn3, gen_syn4
from utils import normalize_adj, get_neighbourhood
from model import GCN, train
from cf_explainer import CFExplainer
from gae.GAE import gae
from visualization import plot_graph
from evaluation import fidelity_size_sparsity
torch.manual_seed(0)
np.random.seed(0)


parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda', help='torch device.')
parser.add_argument('--epochs', type=int, default=400, help='Number of epochs to train.')
parser.add_argument('--hidden1', type=int, default=32, help='Number of units in hidden layer 1.')
parser.add_argument('--hidden2', type=int, default=16, help='Number of units in hidden layer 2.')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset-str', type=str, default='cora', help='type of dataset.')
gae_args = parser.parse_args()

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda', help='torch device.')
parser.add_argument('--bb-epochs', type=int, default=500, help='Number of epochs to train the ')
parser.add_argument('--cf-epochs', type=int, default=300, help='Number of epochs to train the ')
parser.add_argument('--inputdim', type=int, default=10, help='Input dimension')
parser.add_argument('--hidden', type=int, default=20, help='Number of units in hidden layer 1.')
parser.add_argument('--n-layers', type=int, default=3, help='Number of units in hidden layer 1.')
parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate.')
parser.add_argument('--cf-lr', type=float, default=0.01, help='CF-explainer learning rate.')
parser.add_argument('--n-clusters', type=int, default=16, help='Maximum number of Clusters.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--cf-optimizer', type=str, default='SGD', help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset-str', type=str, default='cora', help='type of dataset.')
parser.add_argument('--dataset-func', type=str, default='__load__planetoid__', help='type of dataset.')
parser.add_argument('--beta', type=float, default=0.5, help='beta variable')
parser.add_argument('--include_ae', type=bool, default=True, help='Including AutoEncoder reconstruction loss')
parser.add_argument('--edge-addition', type=bool, default=True, help='CF edge_addition')
parser.add_argument('--algorithm', type=str, default='loss_PN_AE_', help='Result directory')
parser.add_argument('--graph-result-dir', type=str, default='./results/graphs', help='Result directory')
parser.add_argument('--graph-result-name', type=str, default='loss_PN_AE_', help='Result name')
parser.add_argument('--cf_train_loss', type=str, default='loss_PN_AE_', help='CF explainer loss function')
parser.add_argument('--n-momentum', type=float, default=0.0, help='Nesterov momentum')
explainer_args = parser.parse_args()


def main(gae_args, explainer_args):
    data = load_data(explainer_args)
    data_AE = load_data_AE(explainer_args)

    # data =load_synthetic(gen_syn3, device=explainer_args.device)
    # data_AE = load_synthetic_AE(gen_syn3, device=explainer_args.device)
    AE_threshold = {'gen_syn1': 0.5, 'gen_syn2': 0.65, 'gen_syn3':0.6, 'gen_syn4':0.62, 'cora':0.66}
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

    train(
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
    data['cluster_features'] = model.encode(data['features'], data['adj_norm']).detach()
    output = model(data['features'], data['adj_norm'])
    y_pred_orig = torch.argmax(output[data['test_mask']], dim=1)
    print("test set y_true counts: {}".format(np.unique(data['labels'][data['test_mask']].cpu().detach().numpy(), return_counts=True)))
    print("test set y_pred_orig counts: {}".format(np.unique(y_pred_orig.cpu().detach().numpy(), return_counts=True)))
    print("Training GNN is finished.")

    print("Training AE.")
    graph_ae = gae(gae_args, data_AE)
    print("Explanation step:")

    idx_test = np.arange(0, data['n_nodes'])[data['test_mask'].cpu()]
    test_cf_examples = []
    for i in idx_test[:20]:
        sub_adj, sub_feat, sub_labels, node_dict, sub_edge_index = get_neighbourhood(
            int(i), data['edge_index'], explainer_args.n_layers + 1, data['features'], data['labels'])
        new_idx = node_dict[int(i)]
        # Check that original model gives same prediction on full graph and subgraph
        with torch.no_grad():
            print("Output original model, full adj: {}".format(output[i]))
            print(
                "Output original model, sub adj: {}".format(
                    model(sub_feat, normalize_adj(sub_adj, explainer_args.device))[new_idx]
                )
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
            y_pred_orig=y_pred_orig[i],
            num_classes=data['num_classes'],
            beta=explainer_args.beta,
            device=explainer_args.device,
            AE_threshold=AE_threshold[explainer_args.dataset_str],
            algorithm=explainer_args.algorithm,
            edge_addition=explainer_args.edge_addition
        )
        explainer.cf_model.cuda()
        cf_example = explainer.explain(
            node_idx=i,
            new_idx=new_idx,
            num_epochs=explainer_args.cf_epochs
        )
        test_cf_examples.append(cf_example)
        plot_graph(
            sub_adj.cpu().numpy(),
            sub_labels.cpu().numpy(),
            new_idx,
            f'{explainer_args.graph_result_dir}/{explainer_args.dataset_str}_{new_idx}_sub_adj_{explainer_args.graph_result_name}.png',
            sub_edge_index.t().cpu().numpy()
        )
        for i, x in enumerate(cf_example):
            if ~explainer_args.edge_addition:
                cf_sub_adj = sub_adj.mul(torch.from_numpy(x[2]).cuda())
            else:
                cf_sub_adj = sub_adj


            plot_graph(
                cf_sub_adj,
                x[8].cpu().numpy(),
                new_idx,
                f'{explainer_args.graph_result_dir}/{explainer_args.dataset_str}_{new_idx}_counter_factual_{i}_{explainer_args.graph_result_name}.png',
                sub_edge_index.t().cpu().numpy()
            )
        fidelity_size_sparsity(
            model,
            sub_feat,
            sub_adj,
            cf_example,
            explainer_args.edge_addition,
            f'{explainer_args.graph_result_dir}/{explainer_args.dataset_str}_{new_idx}_counter_factual_{explainer_args.graph_result_name}'
        )
        print('yes!')


if __name__ == '__main__':
    main(gae_args, explainer_args)
