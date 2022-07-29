import errno
import torch
import pandas as pd
from torch_geometric.utils import k_hop_subgraph, to_dense_adj, subgraph

from matplotlib import pyplot as plt

# Set matplotlib backend to file writing
plt.switch_backend("agg")
import numpy as np
import networkx as nx
import synthetic_structsim
import featgen
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'


def mkdir_p(path):
	try:
		os.makedirs(path)
	except OSError as exc:  # Python >2.5
		if exc.errno == errno.EEXIST and os.path.isdir(path):
			pass
		else:
			raise


def safe_open(path, w):
	''' Open "path" for writing, creating any parent directories as needed.'''
	mkdir_p(os.path.dirname(path))
	return open(path, w)


def accuracy(output, labels):
	preds = output.max(1)[1].type_as(labels)
	correct = preds.eq(labels).double()
	correct = correct.sum()
	return correct / len(labels)


def get_degree_matrix(adj):
	return torch.diag(sum(adj))


def normalize_adj(adj):
	# Normalize adjacancy matrix according to reparam trick in GCN paper
	A_tilde = adj + torch.eye(adj.shape[0])
	D_tilde = get_degree_matrix(A_tilde)
	# Raise to power -1/2, set all infs to 0s
	D_tilde_exp = D_tilde ** (-1 / 2)
	D_tilde_exp[torch.isinf(D_tilde_exp)] = 0

	# Create norm_adj = (D + I)^(-1/2) * (A + I) * (D + I) ^(-1/2)
	norm_adj = torch.mm(torch.mm(D_tilde_exp, A_tilde), D_tilde_exp)
	return norm_adj

def get_neighbourhood(node_idx, edge_index, n_hops, features, labels):
	edge_subset = k_hop_subgraph(node_idx, n_hops, edge_index[0])     # Get all nodes involved
	edge_subset_relabel = subgraph(edge_subset[0], edge_index[0], relabel_nodes=True)       # Get relabelled subset of edges
	sub_adj = to_dense_adj(edge_subset_relabel[0]).squeeze()
	sub_feat = features[edge_subset[0], :]
	sub_labels = labels[edge_subset[0]]
	new_index = np.array([i for i in range(len(edge_subset[0]))])
	node_dict = dict(zip(edge_subset[0].numpy(), new_index))        # Maps orig labels to new
	# print("Num nodes in subgraph: {}".format(len(edge_subset[0])))
	return sub_adj, sub_feat, sub_labels, node_dict


def create_symm_matrix_from_vec(vector, n_rows):
	matrix = torch.zeros(n_rows, n_rows)
	idx = torch.tril_indices(n_rows, n_rows)
	matrix[idx[0], idx[1]] = vector
	symm_matrix = torch.tril(matrix) + torch.tril(matrix, -1).t()
	return symm_matrix


def create_vec_from_symm_matrix(matrix, P_vec_size):
	idx = torch.tril_indices(matrix.shape[0], matrix.shape[0])
	vector = matrix[idx[0], idx[1]]
	return vector


def index_to_mask(index, size):
	mask = torch.zeros(size, dtype=torch.bool, device=index.device)
	mask[index] = 1
	return mask

def get_S_values(pickled_results, header):
	df_prep = []
	for example in pickled_results:
		if example != []:
			df_prep.append(example[0])
	return pd.DataFrame(df_prep, columns=header)


def redo_dataset_pgexplainer_format(dataset, train_idx, test_idx):

	dataset.data.train_mask = index_to_mask(train_idx, size=dataset.data.num_nodes)
	dataset.data.test_mask = index_to_mask(test_idx[len(test_idx)], size=dataset.data.num_nodes)


def perturb(graph_list, p):
	""" Perturb the list of (sparse) graphs by adding/removing edges.
	Args:
		p: proportion of added edges based on current number of edges.
	Returns:
		A list of graphs that are perturbed from the original graphs.
	"""
	perturbed_graph_list = []
	for G_original in graph_list:
		G = G_original.copy()
		edge_count = int(G.number_of_edges() * p)
		# randomly add the edges between a pair of nodes without an edge.
		for _ in range(edge_count):
			while True:
				u = np.random.randint(0, G.number_of_nodes())
				v = np.random.randint(0, G.number_of_nodes())
				if (not G.has_edge(u, v)) and (u != v):
					break
			G.add_edge(u, v)
		perturbed_graph_list.append(G)
	return perturbed_graph_list


def join_graph(G1, G2, n_pert_edges):
	""" Join two graphs along matching nodes, then perturb the resulting graph.
	Args:
		G1, G2: Networkx graphs to be joined.
		n_pert_edges: number of perturbed edges.
	Returns:
		A new graph, result of merging and perturbing G1 and G2.
	"""
	assert n_pert_edges > 0
	F = nx.compose(G1, G2)
	edge_cnt = 0
	while edge_cnt < n_pert_edges:
		node_1 = np.random.choice(G1.nodes())
		node_2 = np.random.choice(G2.nodes())
		F.add_edge(node_1, node_2)
		edge_cnt += 1
	return F


def preprocess_input_graph(G, labels, normalize_adj=False):
	""" Load an existing graph to be converted for the experiments.
	Args:
		G: Networkx graph to be loaded.
		labels: Associated node labels.
		normalize_adj: Should the method return a normalized adjacency matrix.
	Returns:
		A dictionary containing adjacency, node features and labels
	"""
	adj = np.array(nx.to_numpy_matrix(G))
	if normalize_adj:
		sqrt_deg = np.diag(1.0 / np.sqrt(np.sum(adj, axis=0, dtype=float).squeeze()))
		adj = np.matmul(np.matmul(sqrt_deg, adj), sqrt_deg)

	existing_node = list(G.nodes)[-1]
	feat_dim = G.nodes[existing_node]["feat"].shape[0]
	f = np.zeros((G.number_of_nodes(), feat_dim), dtype=float)
	for i, u in enumerate(G.nodes()):
		f[i, :] = G.nodes[u]["feat"]

	# add batch dim
	adj = np.expand_dims(adj, axis=0)
	f = np.expand_dims(f, axis=0)
	labels = np.expand_dims(labels, axis=0)
	return {"adj": adj, "feat": f, "labels": labels}


	####################################
	#
	# Generating synthetic graphs
	#
	###################################
def gen_syn1(nb_shapes=80, width_basis=300, feature_generator=None, m=5):
	""" Synthetic Graph #1:
	Start with Barabasi-Albert graph and attach house-shaped subgraphs.
	Args:
		nb_shapes         :  The number of shapes (here 'houses') that should be added to the base graph.
		width_basis       :  The width of the basis graph (here 'Barabasi-Albert' random graph).
		feature_generator :  A `FeatureGenerator` for node features. If `None`, add constant features to nodes.
		m                 :  number of edges to attach to existing node (for BA graph)
	Returns:
		G                 :  A networkx graph
		role_id           :  A list with length equal to number of nodes in the entire graph (basis
						  :  + shapes). role_id[i] is the ID of the role of node i. It is the label.
		name              :  A graph identifier
	"""
	basis_type = "ba"
	list_shapes = [["house"]] * nb_shapes

	plt.figure(figsize=(8, 6), dpi=300)

	G, role_id, _ = synthetic_structsim.build_graph(
		width_basis, basis_type, list_shapes, start=0, m=5
	)
	G = perturb([G], 0.01)[0]

	if feature_generator is None:
		feature_generator = featgen.ConstFeatureGen(1)
	feature_generator.gen_node_features(G)

	name = basis_type + "_" + str(width_basis) + "_" + str(nb_shapes)
	return G, role_id, name


def gen_syn2(nb_shapes=100, width_basis=350):
	""" Synthetic Graph #2:
	Start with Barabasi-Albert graph and add node features indicative of a community label.
	Args:
		nb_shapes         :  The number of shapes (here 'houses') that should be added to the base graph.
		width_basis       :  The width of the basis graph (here 'Barabasi-Albert' random graph).
	Returns:
		G                 :  A networkx graph
		label             :  Label of the nodes (determined by role_id and community)
		name              :  A graph identifier
	"""
	basis_type = "ba"

	random_mu = [0.0] * 8
	random_sigma = [1.0] * 8

	# Create two grids
	mu_1, sigma_1 = np.array([-1.0] * 2 + random_mu), np.array([0.5] * 2 + random_sigma)
	mu_2, sigma_2 = np.array([1.0] * 2 + random_mu), np.array([0.5] * 2 + random_sigma)
	feat_gen_G1 = featgen.GaussianFeatureGen(mu=mu_1, sigma=sigma_1)
	feat_gen_G2 = featgen.GaussianFeatureGen(mu=mu_2, sigma=sigma_2)
	G1, role_id1, name = gen_syn1(feature_generator=feat_gen_G1, m=4)
	G2, role_id2, name = gen_syn1(feature_generator=feat_gen_G2, m=4)
	G1_size = G1.number_of_nodes()
	num_roles = max(role_id1) + 1
	role_id2 = [r + num_roles for r in role_id2]
	label = role_id1 + role_id2

	# Edit node ids to avoid collisions on join
	g1_map = {n: i for i, n in enumerate(G1.nodes())}
	G1 = nx.relabel_nodes(G1, g1_map)
	g2_map = {n: i + G1_size for i, n in enumerate(G2.nodes())}
	G2 = nx.relabel_nodes(G2, g2_map)

	# Join
	n_pert_edges = width_basis
	G = join_graph(G1, G2, n_pert_edges)

	name = basis_type + "_" + str(width_basis) + "_" + str(nb_shapes) + "_2comm"

	return G, label, name


def gen_syn3(nb_shapes=80, width_basis=300, feature_generator=None, m=5):
	""" Synthetic Graph #3:
	Start with Barabasi-Albert graph and attach grid-shaped subgraphs.
	Args:
		nb_shapes         :  The number of shapes (here 'grid') that should be added to the base graph.
		width_basis       :  The width of the basis graph (here 'Barabasi-Albert' random graph).
		feature_generator :  A `FeatureGenerator` for node features. If `None`, add constant features to nodes.
		m                 :  number of edges to attach to existing node (for BA graph)
	Returns:
		G                 :  A networkx graph
		role_id           :  Role ID for each node in synthetic graph.
		name              :  A graph identifier
	"""
	basis_type = "ba"
	list_shapes = [["grid", 3]] * nb_shapes

	plt.figure(figsize=(8, 6), dpi=300)

	G, role_id, _ = synthetic_structsim.build_graph(
		width_basis, basis_type, list_shapes, start=0, m=5
	)
	G = perturb([G], 0.01)[0]

	if feature_generator is None:
		feature_generator = featgen.ConstFeatureGen(1)
	feature_generator.gen_node_features(G)

	name = basis_type + "_" + str(width_basis) + "_" + str(nb_shapes)
	return G, role_id,