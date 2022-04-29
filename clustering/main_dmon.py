from clustering.dmon import DMoN
import torch
import numpy as np
import sklearn
import clustering.metrics as metrics
torch.manual_seed(0)
np.random.seed(0)


def DMon(args, data, model):
    model.eval()

    dmon = DMoN(data['cluster_features'].shape[1], n_clusters=args.n_clusters)
    optimizer = torch.optim.Adam(dmon.parameters(), lr=args.dmon_lr, weight_decay=5e-4)

    for i in range(2000):
        dmon.train()
        optimizer.zero_grad()
        loss = dmon.loss(data['cluster_features'], data['adj_norm'].to_dense())
        loss.backward()
        optimizer.step()
        if i%100==0 and i!=0:
            print(f'epoch {i}, loss: {loss}')

    dmon.eval()
    features_pooled, assignments = dmon.forward(data['cluster_features'])
    assignments = assignments.cpu().detach().numpy()
    clusters = assignments.argmax(axis=1)
    # Prints some metrics used in the paper.
    print('Conductance:', metrics.conductance(data['adj_orig'], clusters))
    print('Modularity:', metrics.modularity(data['adj_orig'], clusters))
    print(
        'NMI:',
        sklearn.metrics.normalized_mutual_info_score(
            data['labels'].cpu(), clusters, average_method='arithmetic'))
    precision = metrics.pairwise_precision(data['labels'].cpu(), clusters)
    recall = metrics.pairwise_recall(data['labels'].cpu(), clusters)
    print('F1:', 2 * precision * recall / (precision + recall))
    return dmon