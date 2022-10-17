from __future__ import division
from __future__ import print_function
import numpy as np
import torch
from torch.optim import Adam
from gae.torchgeometric_ae import DeepVGAE
torch.manual_seed(0)
np.random.seed(0)


def gae(args, data):

    print("Using {} dataset".format(args.dataset_str))
    model = DeepVGAE(data['feat_dim'], args.hidden1, args.hidden2).to(args.device)
    optimizer = Adam(model.parameters(), lr=args.lr)
    prev_prec = 0
    patience = 15
    trigger_times = 0
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()

        loss = model.loss(data['dataset'].x, data['dataset'].train_pos_edge_index, data['all_edge_index'])
        loss.backward()
        optimizer.step()
        if epoch % 2 == 0:

            model.eval()
            roc_auc, ap = model.single_test(data['dataset'].x,
                                            data['dataset'].train_pos_edge_index,
                                            data['dataset'].test_pos_edge_index,
                                            data['dataset'].test_neg_edge_index)
            if prev_prec > ap:
                trigger_times +=1
                print('Trigger Times:', trigger_times)

                if trigger_times >= patience:
                    print('Early stopping!\nStart to test process.')
                    print("Epoch {} - Loss: {} ROC_AUC: {} Precision: {}".format(epoch, loss.cpu().item(), roc_auc, ap))
                    return model
            else:
                print("Epoch {} - Loss: {} ROC_AUC: {} Precision: {}".format(epoch, loss.cpu().item(), roc_auc, ap))
                trigger_times = 0
            prev_prec = ap
    return model