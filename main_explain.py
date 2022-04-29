import argparse
import sys

sys.path.append('../..')
import torch
import numpy as np
from data.data_loader import load_data, load_data_ae
from utils import normalize_adj, get_neighbourhood
from model import GCN, train
from cf_explainer import CFExplainer
from gae.GAE import gae
from clustering.main_dmon import DMon
torch.manual_seed(0)
np.random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda', help='torch device.')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--hidden1', type=int, default=32, help='Number of units in hidden layer 1.')
parser.add_argument('--hidden2', type=int, default=16, help='Number of units in hidden layer 2.')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset-str', type=str, default='cora', help='type of dataset.')
gae_args = parser.parse_args()

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda', help='torch device.')
parser.add_argument('--bb-epochs', type=int, default=500, help='Number of epochs to train the ')
parser.add_argument('--cf-epochs', type=int, default=200, help='Number of epochs to train the ')
parser.add_argument('--inputdim', type=int, default=10, help='Input dimension')
parser.add_argument('--hidden', type=int, default=20, help='Number of units in hidden layer 1.')
parser.add_argument('--n-layers', type=int, default=3, help='Number of units in hidden layer 1.')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--dmon-lr', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--n-clusters', type=int, default=16, help='Maximum number of Clusters.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--cf-optimizer', type=str, default='SGD', help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset-str', type=str, default='cora', help='type of dataset.')
parser.add_argument('--beta', type=float, default=0.5, help='beta variable')
parser.add_argument('--include_ae', type=bool, default=False, help='Including AutoEncoder reconstruction loss')
parser.add_argument('--n-momentum', type=float, default=0.0, help='Nesterov momentum')
explainer_args = parser.parse_args()


def main(gae_args, explainer_args):
    inputdim = 10
    hidden = 20
    dropout = 0.5
    n_layers = 3

    data =load_data(explainer_args)
    model = GCN(
        nfeat=data['feat_dim'],
        nhid=explainer_args.hidden,
        nout=explainer_args.hidden,
        nclasses=data['num_classes'],
        dropout=explainer_args.dropout
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=explainer_args.lr, weight_decay=5e-4)

    if explainer_args.device=='cuda':
        model = model.cuda()

    #     'dataset':dataset,
    #     'train_mask':train_mask,
    #     'val_mask':val_mask,
    #     'test_mask':test_mask,
    #     'idx_test':idx_test,
    #     'features':features,
    #     'labels':labels,
    #     'adj':adj,
    #     'adj_norm':adj_norm,
    #     'adj_orig':adj_orig,
    #     'pos_weight':pos_weight,
    #     'norm':norm,
    #     'n_nodes':n_nodes,
    #     'feat_dim':feat_dim,
    #     'num_classes':num_classes
    # }

    train(
        model=model,
        features=data['features'],
        train_adj=data['adj_norm'],
        labels=data['labels'],
        train_mask=data['train_mask'],
        optimizer=optimizer,
        epoch=explainer_args.bb_epochs,
        val_mask=data['val_mask']
    )

    model.eval()
    data['cluster_features'] = model.encode(data['features'], data['adj_norm']).detach()
    output = model(data['features'], data['adj_norm'])
    y_pred_orig = torch.argmax(output[data['test_mask']], dim=1)
    print("test set y_true counts: {}".format(np.unique(data['labels'][data['test_mask']].cpu().detach().numpy(), return_counts=True)))
    print("test set y_pred_orig counts: {}".format(np.unique(y_pred_orig.cpu().detach().numpy(), return_counts=True)))
    print("Training GNN is finished.")

    print("Training clustering model:")
    dmon = DMon(explainer_args, data, model)
    print("Training AE.")
    graph_ae = gae(gae_args)
    print("Explanation step:")

    idx_test = np.arange(0, data['n_nodes'])[data['test_mask'].cpu()]
    test_cf_examples = []
    for i in idx_test[:20]:
        sub_adj, sub_feat, sub_labels, node_dict, sub_edge_idx = get_neighbourhood(
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
        sub_feat_ = model.encode(sub_feat, normalize_adj(sub_adj, explainer_args.device))
        # Need to instantitate new cf model every time because size of P changes based on size of sub_adj
        explainer = CFExplainer(
            model=model,
            graph_ae=graph_ae,
            cluster=dmon,
            sub_adj=sub_adj,
            sub_feat=sub_feat,
            n_hid=explainer_args.hidden,
            dropout=explainer_args.dropout,
            sub_labels=sub_labels,
            y_pred_orig=y_pred_orig[i],
            num_classes=data['num_classes'],
            beta=explainer_args.beta,
            device=explainer_args.device
        )
        explainer.cf_model.cuda()
        cf_example = explainer.explain(
            node_idx=i,
            cf_optimizer=explainer_args.cf_optimizer,
            new_idx=new_idx,
            lr=explainer_args.lr,
            n_momentum=explainer_args.n_momentum,
            num_epochs=explainer_args.cf_epochs,
            encode_sub_features=sub_feat_
        )  # Need node dict for accuracy calculation
        test_cf_examples.append(cf_example)
        print('yes!')


if __name__ == '__main__':
    main(gae_args, explainer_args)
