import argparse
import gc
import sys
import torch
from data.data_loader import load_data, load_synthetic, load_synthetic_AE, load_data_AE
from torch.optim import Adam
from gae.torchgeometric_gae import GAE
from gae.torchgeometric_vgae import DeepVGAE


# torch.manual_seed(0)
# np.random.seed(0)

sys.path.append('../..')


def main(gae_args):
    torch.cuda.empty_cache()
    data_AE = load_data_AE(explainer_args)
    # data =load_synthetic(gen_syn1, device=explainer_args.device)
    # data_AE = load_synthetic_AE(gen_syn1, device=explainer_args.device)

    print("Training AE.")
    print("Using {} dataset".format(gae_args.dataset_str))
    model = GAE(data_AE['feat_dim'], gae_args.hidden1, gae_args.hidden2).to(gae_args.device)
    optimizer = Adam(model.parameters(), lr=gae_args.lr)

    for epoch in range(gae_args.epochs):
        model.train()
        optimizer.zero_grad()

        loss = model.loss(
            data_AE['train_data'].x,
            data_AE['train_data'].edge_label_index,
        )
        loss.backward()
        optimizer.step()
        if epoch % 5 == 0:

            model.eval()
            roc_auc, ap = model.single_test(data_AE['train_data'].x,
                                            data_AE['train_data'].edge_label_index,
                                            data_AE['test_data'].edge_label_index)
            print("Epoch {} - Loss: {} ROC_AUC: {} Precision: {}".format(epoch, loss.cpu().item(), roc_auc, ap))
    print("Explanation step:")


    torch.cuda.empty_cache()
    gc.collect()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu', help='torch device.')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
    parser.add_argument('--hidden1', type=int, default=128, help='Number of units in hidden layer 1.')
    parser.add_argument('--hidden2', type=int, default=64, help='Number of units in hidden layer 2.')
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