import networkx as nx


def gen_graph(nodes, edge_list):
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edge_list)
    return G


def centrality(graph:nx.Graph):
    centrality_metrics = {}
    centrality_metrics['brandes'] = nx.betweenness_centrality(graph)
    centrality_metrics['closeness'] = nx.closeness_centrality(graph)
    centrality_metrics['betweenness'] = nx.betweenness_centrality(graph)

    return centrality_metrics


def clustering(graph:nx):
    return nx.clustering(graph)

