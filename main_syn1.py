import sys

sys.path.append('../..')
import torch
import numpy as np
import pickle
from torch_geometric.utils import dense_to_sparse
from utils import normalize_adj, get_neighbourhood
from model import GCN, train
from cf_explainer import CFExplainer


def main():
    inputdim = 10
    train_ratio = 0.8
    hidden = 20
    seed = 42
    dropout = 0.5
    n_layers = 3
    epoch = 200
    device = 'cuda'
    beta = 0.5
    learning_rate = 0.001
    n_momentum = 0.0
    cf_optimizer = 'SGD'
    # for Pertinent Negative -> edge_additions = True / Pertinent Positive -> edge_additions = False
    edge_additions =  True


    with open('data/syn1.pickle', "rb") as f:
        data = pickle.load(f)

    adj = torch.Tensor(data["adj"]).squeeze()  # Does not include self loops
    features = torch.Tensor(data["feat"]).squeeze()
    labels = torch.tensor(data["labels"]).squeeze()
    idx_train = torch.tensor(data["train_idx"])
    idx_test = torch.tensor(data["test_idx"])
    edge_index = dense_to_sparse(adj)

    norm_adj = normalize_adj(adj)
    model = GCN(nfeat=features.shape[1], nhid=hidden, nout=hidden, nclasses=len(labels.unique()), dropout=dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)

    if torch.cuda.is_available():
        model = model.cuda()
        norm_adj = norm_adj.cuda()
        adj = adj.cuda()
        features = features.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_test = idx_test.cuda()

    train(model, features, adj, labels, idx_train, idx_test, optimizer, epoch)

    model.eval()
    output = model(features.cuda(), norm_adj.cuda())
    y_pred_orig = torch.argmax(output, dim=1)
    print("test set y_true counts: {}".format(np.unique(labels.cpu().detach().numpy(), return_counts=True)))
    print("test set y_pred_orig counts: {}".format(np.unique(y_pred_orig.cpu().detach().numpy(), return_counts=True)))

    test_cf_examples = []
    for i in idx_test[:20]:
        sub_adj, sub_feat, sub_labels, node_dict = get_neighbourhood(int(i), edge_index, n_layers + 1, features,
                                                                     labels)
        new_idx = node_dict[int(i)]

        # Check that original model gives same prediction on full graph and subgraph
        with torch.no_grad():
            print("Output original model, full adj: {}".format(output[i]))
            print("Output original model, sub adj: {}".format(model(sub_feat, normalize_adj(sub_adj).cuda())[new_idx]))

        # Need to instantitate new cf model every time because size of P changes based on size of sub_adj

        if device == 'cuda':
            model.cuda()
            adj = adj.cuda()
            norm_adj = norm_adj.cuda()
            features = features.cuda()
            labels = labels.cuda()
            idx_train = idx_train.cuda()
            idx_test = idx_test.cuda()

        explainer = CFExplainer(
            model=model,
            sub_adj=sub_adj,
            sub_feat=sub_feat,
            n_hid=hidden,
            dropout=dropout,
            sub_labels=sub_labels,
            y_pred_orig=y_pred_orig[i],
            num_classes=len(labels.unique()),
            beta=beta,
            device=device
        )
        explainer.cf_model.cuda()
        cf_example = explainer.explain(
            node_idx=i,
            cf_optimizer=cf_optimizer,
            new_idx=new_idx,
            lr=learning_rate,
            n_momentum=n_momentum,
            num_epochs=epoch,
        )  # Need node dict for accuracy calculation
        test_cf_examples.append(cf_example)


if __name__ == '__main__':
    main()
