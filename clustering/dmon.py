import torch.nn
from torch.nn.modules import Module
import torch.nn.functional as F


class DMoN(Module):
    """Implementation of Deep Modularity Network (DMoN) layer.
    Deep Modularity Network (DMoN) layer implementation as presented in
    "Graph Clustering with Graph Neural Networks" in a form of TF 2.0 Keras layer.
    DMoN optimizes modularity clustering objective in a fully unsupervised mode,
    however, this implementation can also be used as a regularizer in a supervised
    graph neural network. Optionally, it does graph unpooling.
    Attributes:
      n_clusters: Number of clusters in the model.
      collapse_regularization: Collapse regularization weight.
      dropout_rate: Dropout rate. Note that the dropout in applied to the
        intermediate representations before the softmax.
      do_unpooling: Parameter controlling whether to perform unpooling of the
        features with respect to their soft clusters. If true, shape of the input
        is preserved.
    """

    def __init__(self,
                 input_dim,
                 n_clusters,
                 collapse_regularization=0.1,
                 dropout_rate=0,
                 do_unpooling=False,
                 device='cuda'
                 ):
        """Initializes the layer with specified parameters."""
        super(DMoN, self).__init__()
        self.n_clusters = n_clusters
        self.collapse_regularization = collapse_regularization
        self.dropout_rate = dropout_rate
        self.do_unpooling = do_unpooling
        self.input_dim = input_dim
        self.transform = torch.nn.Linear(self.input_dim, self.n_clusters, device=device, bias=True)
        self.init_weights()

    def init_weights(self):
        torch.nn.init.orthogonal_(self.transform.weight)

    def loss(self, features, adjacency):

        assignments = F.softmax(self.transform(features), dim=1)
        cluster_size = torch.sum(assignments, dim=0)
        assignments_pooling = assignments/cluster_size
        degrees = adjacency.sum(dim=0)
        degrees = torch.reshape(degrees, (-1, 1))

        number_of_nodes = adjacency.shape[1]
        number_of_edges = torch.sum(degrees)
        graph_pooled = torch.spmm(adjacency, assignments).t()
        graph_pooled = torch.mm(graph_pooled, assignments)

        # We compute the rank-1 normaizer matrix S^T*d*d^T*S efficiently
        # in three matrix multiplications by first processing the left part S^T*d
        # and then multyplying it by the right part d^T*S.
        # Left part is [k, 1] tensor.
        normalizer_left = torch.mm(assignments.t(), degrees)
        # Right part is [1, k] tensor.
        normalizer_right = torch.mm(degrees.t(), assignments)

        # Normalizer is rank-1 correction for degree distribution for degrees of the
        # nodes in the original graph, casted to the pooled graph.
        normalizer = torch.mm(normalizer_left,
                               normalizer_right) / 2 / number_of_edges

        spectral_loss = -(torch.diag(graph_pooled -
                                         normalizer).sum()) / 2 / number_of_edges
        collapse_loss = torch.norm(cluster_size) / number_of_nodes * torch.sqrt(
            torch.tensor(self.n_clusters, dtype=torch.float)) - 1
        return spectral_loss + self.collapse_regularization * collapse_loss

    def forward(self, features):
        assignments = F.softmax(self.transform(features), dim=1)
        cluster_size = torch.sum(assignments, dim=0)
        assignments_pooling = assignments/cluster_size
        features_pooled = torch.mm(assignments_pooling.t(), features)
        features_pooled = F.relu(features_pooled)
        if self.do_unpooling:
            features_pooled = torch.mm(assignments_pooling, features_pooled)
        return features_pooled, assignments
