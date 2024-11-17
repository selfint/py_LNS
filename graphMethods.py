import numpy as np
from collections import Counter
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import networkx as nx
import matplotlib.pyplot as plt





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


def get_random_connected_component(matrix):
    # Turn matrix into graph ds
    graph = csr_matrix(matrix)

    # Get connected components from graph
    n_components, labels = connected_components(csgraph=graph, directed=False, return_labels=True)

    # Choose random connected component
    random_cc_id = np.random.randint(n_components)

    # Retrieve agents in connected component
    agent_ids = [i for i, x in enumerate(labels) if x == random_cc_id]
    return agent_ids


def random_walk_until_neighborhood_is_full(matrix, connected_component, subset_size):
    # Turn matrix into networkx graph ds
    graph = nx.from_numpy_array(np.array(matrix))

    # Produce the subgraph induced by the connected component
    graph = nx.subgraph(graph, connected_component)

    # Pick random agent as start of walk
    agent_id = int(np.random.choice(connected_component, 1))

    # Initialize subset as node of connected component
    subset = {agent_id}

    current_agent = agent_id
    while len(subset) < subset_size:
        next_agent = get_random_neighbor(graph, current_agent)
        subset.add(next_agent)
        current_agent = next_agent

    return subset



def get_random_neighbor(graph, node):
    neighbors = np.array(list(nx.all_neighbors(graph, node)))
    return int(np.random.choice(neighbors, 1))


