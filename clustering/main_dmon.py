from collections import Counter
from clustering.dmon import DMoN
import torch
import numpy as np
import sklearn
import clustering.metrics as metrics
from visualization import simple_plot
torch.manual_seed(0)
np.random.seed(0)


def DMon(data, model, num_clusters, epochs, lr,dataset_name=''):
    model.eval()

    dmon = DMoN(data['cluster_features'].shape[1], n_clusters=num_clusters)
    optimizer = torch.optim.Adam(dmon.parameters(), lr=lr, weight_decay=5e-4)
    loss_ = []
    conductance_ = []
    modularity_ = []
    f1_ = []
    epochs_=[]
    for i in range(epochs):
        dmon.train()
        optimizer.zero_grad()
        if data['adj_norm'].is_sparse:
            loss = dmon.loss(data['cluster_features'], data['adj_norm'].to_dense())
        else:
            loss = dmon.loss(data['cluster_features'], data['adj_norm'])
        loss.backward()
        optimizer.step()

        if i%10==0 and i!=0:
            epochs_.append(i)
            print(f'epoch {i}, loss: {loss}')
            print('Num of dt instances:', Counter(data['labels'].cpu().numpy()))
            print('Num of Clusters:', Counter(dmon(data['cluster_features'])[-1].argmax(dim=1).cpu().numpy()))
            loss_.append(loss.cpu().detach().numpy())
            dmon.eval()
            features_pooled, assignments = dmon.forward(data['cluster_features'])
            assignments = assignments.cpu().detach().numpy()
            clusters = assignments.argmax(axis=1)
            c = metrics.conductance(data['adj_orig'], clusters)
            conductance_.append(c)
            m = metrics.modularity(data['adj_orig'], clusters)
            modularity_.append(m)
            # Prints some metrics used in the paper.
            print('Conductance:', c)
            print('Modularity:', m)
            print(
                'NMI:',
                sklearn.metrics.normalized_mutual_info_score(
                    data['labels'].cpu(), clusters, average_method='arithmetic'))
            precision = metrics.pairwise_precision(data['labels'].cpu(), clusters)
            recall = metrics.pairwise_recall(data['labels'].cpu(), clusters)
            f = 2 * precision * recall / (precision + recall)
            f1_.append(f)
            print('F1:', f)
    simple_plot(x=epochs_, y=[loss_], labels=['loss_'], name=f'loss_{dataset_name}')
    simple_plot(
        x=epochs_, y=[modularity_, conductance_, f1_], labels=['modularity_', 'conductance_', 'f1_'], name=f'multiple_{dataset_name}'
    )
    return dmon