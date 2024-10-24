import numpy as np
import instance
import PathTable
from graphMethods import get_largest_connected_component, get_degrees_of_vertices


class DestroyHeuristic:
    def __init__(self, instance: instance.instance, path_table:PathTable.PathTable, subset_size):
        self.instance = instance
        self.path_table = path_table
        self.subset_size = subset_size

    def generate_subset(self):
        raise NotImplementedError()


class RandomDestroyHeuristic(DestroyHeuristic):
    def generate_subset(self):
        return np.random.choice(range(1, self.instance.num_agents+1), self.subset_size, replace=False)

class ConnectedComponentDestroyHeuristic(DestroyHeuristic):
    def generate_subset(self):
        adj_matrix = self.path_table.get_collisions_matrix(self.instance.num_agents)
        return get_largest_connected_component(adj_matrix)

class PriorityDestroyHeuristic(DestroyHeuristic):
    def generate_subset(self):
        argmax_size = min(3, self.subset_size)
        random_size = self.subset_size - argmax_size
        cols = []
        for agent in self.instance.agents.values():
            path = agent.paths[agent.path_id]
            cols += [self.path_table.count_collisions_along_existing_path(path)]
        cols = np.array(cols)
        argmax_ids = []
        if argmax_size > 0:
            argmax_incides = np.argpartition(cols, -argmax_size)[-argmax_size:]
            argmax_ids = (argmax_incides+1).tolist()
        random_ids = [agent_id for agent_id in range(1, self.instance.num_agents+1) if agent_id not in argmax_ids]
        subset = argmax_ids + np.random.choice(random_ids, random_size, replace=False).tolist()
        return subset


