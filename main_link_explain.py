import argparse
import gc
import sys
import torch
from data.data_loader import load_data, load_synthetic, load_synthetic_AE, load_data_AE
from torch.optim import Adam
from gae.torchgeometric_gae import GAE
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score
from gae.torchgeometric_gae_perturb import GCNSyntheticPerturb
sys.path.append('../..')


def load_state_dict(model):
    # this function specifically loads the GAE to the GAEPerturbation that is inherited from MessagePassign
    from collections import OrderedDict

    new_state_dict = OrderedDict()
    for k, v in model.state_dict().items():
        name = k.replace("lin.", "")  # remove `module.`
        new_state_dict[name] = v.t()
    return new_state_dict

def get_link_labels(pos_edge_index, neg_edge_index, device):
    # returns a tensor:
    # [1,1,1,1,...,0,0,0,0,0,..] with the number of ones is equel to the lenght of pos_edge_index
    # and the number of zeros is equal to the length of neg_edge_index
    E = pos_edge_index.size(1) + neg_edge_index.size(1)
    link_labels = torch.zeros(E, dtype=torch.float, device=device)
    link_labels[:pos_edge_index.size(1)] = 1.
    return link_labels


def train(model, data, optimizer):
    model.train()

    neg_edge_index = negative_sampling(
        edge_index=data.train_pos_edge_index,  # positive edges
        num_nodes=data.num_nodes,  # number of nodes
        num_neg_samples=data.train_pos_edge_index.size(1))  # number of neg_sample equal to number of pos_edges

    optimizer.zero_grad()

    z = model.encode(data.x, data.train_pos_edge_index)  # encode
    link_logits = model.decode(z, data.train_pos_edge_index, neg_edge_index)  # decode

    link_labels = get_link_labels(data.train_pos_edge_index, neg_edge_index, device=gae_args.device)
    loss = F.binary_cross_entropy_with_logits(link_logits, link_labels)
    loss.backward()
    optimizer.step()

    return loss


@torch.no_grad()
def test(model, data, device='cuda'):
    model.eval()
    perfs = []
    for prefix in ["val", "test"]:
        pos_edge_index = data[f'{prefix}_pos_edge_index'].to(device)
        neg_edge_index = data[f'{prefix}_neg_edge_index'].to(device)
        z = model.encode(data.x, data.train_pos_edge_index)  # encode train
        link_logits = model.decode(z, pos_edge_index, neg_edge_index)  # decode test or val
        link_probs = link_logits.sigmoid()  # apply sigmoid
        link_labels = get_link_labels(pos_edge_index, neg_edge_index, gae_args.device)  # get link
        perfs.append(roc_auc_score(link_labels.cpu(), link_probs.cpu()))  # compute roc_auc score
    return perfs


def main(gae_args):
    torch.cuda.empty_cache()
    data_AE = load_data_AE(explainer_args)
    # data =load_synthetic(gen_syn1, device=explainer_args.device)
    # data_AE = load_synthetic_AE(gen_syn1, device=explainer_args.device)

    print("Training AE.")
    print("Using {} dataset".format(gae_args.dataset_str))
    model = GAE(data_AE['feat_dim'], gae_args.hidden1, gae_args.hidden2).to(gae_args.device)
    optimizer = Adam(model.parameters(), lr=gae_args.lr)
    best_val_perf = test_perf = 0
    for epoch in range(1, 101):
        train_loss = train(model, data_AE['dataset'], optimizer)
        val_perf, tmp_test_perf = test(model, data_AE['dataset'])
        if val_perf > best_val_perf:
            best_val_perf = val_perf
            test_perf = tmp_test_perf
        log = 'Epoch: {:03d}, Loss: {:.4f}, Val: {:.4f}, Test: {:.4f}'
        if epoch % 10 == 0:
            print(log.format(epoch, train_loss, best_val_perf, test_perf))

    explainer = GCNSyntheticPerturb(
        data_AE['feat_dim'], gae_args.hidden1, gae_args.hidden2, data_AE['dataset'].train_pos_edge_index, data_AE['n_nodes']
    )
    explainer.cuda()
    state_dict = load_state_dict(model)
    explainer.load_state_dict(state_dict, strict=False)
    z = explainer.encode(data_AE['dataset'].x)
    explainer.decode(z, data_AE['dataset'].test_pos_edge_index[:,2].reshape(-1,1))
    explainer.loss(data_AE['dataset'].x, data_AE['dataset'].test_pos_edge_index[:,2].reshape(-1,1), torch.ones(size=(1,1)))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda', help='torch device.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
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
    parser.add_argument('--graph_result_dir', type=str, default='./results', help='Result directory')
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

    main(gae_args)