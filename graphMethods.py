import numpy as np
from collections import Counter
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


def get_degrees_of_vertices_dict(adj_matrix, vertex_list):
    return {v: adj_matrix[v].sum() for v in vertex_list}

def get_degrees_of_all_vertices(adj_matrix):
    return adj_matrix[1:].sum(axis = 1)

def get_largest_connected_component(matrix):
    graph = csr_matrix(matrix)
    n_components, labels = connected_components(csgraph=graph, directed=False, return_labels=True)
    counter = Counter(labels)
    largest_cc_id = counter.most_common()[0][0]
    agent_ids = [i for i, x in enumerate(labels) if x == largest_cc_id]
    return agent_ids