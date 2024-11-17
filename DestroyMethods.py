import numpy as np
import instance
import PathTable
from graphMethods import *


class DestroyHeuristic:
    def __init__(self, instance: instance.instance, path_table:PathTable.PathTable, subset_size):
        self.instance = instance
        self.path_table = path_table
        self.subset_size = subset_size

    def generate_subset(self):
        raise NotImplementedError()


class RandomDestroyHeuristic(DestroyHeuristic):
    def generate_subset(self, initial_subset = []):
        # Remove chosen agents from available options
        random_ids = [agent_id for agent_id in range(1, self.instance.num_agents+1) if agent_id not in initial_subset]


        # Calculate number of agents left to choose
        random_size = self.subset_size - len(initial_subset)

        # Choose random agents
        subset = initial_subset + np.random.choice(random_ids, random_size, replace=False).tolist()
        return subset

class ConnectedComponentDestroyHeuristic(DestroyHeuristic):
    def generate_subset(self, complete_random = True):
        # Generate collisions matrix
        adj_matrix = self.path_table.get_collisions_matrix(self.instance.num_agents)
        largest_cc = get_largest_connected_component(adj_matrix)
        if len(largest_cc) > self.subset_size:
            return random_walk_until_neighborhood_is_full(adj_matrix, largest_cc, subset_size=self.subset_size)
        if len(largest_cc) < self.subset_size:
            random_dh = RandomDestroyHeuristic(self.instance, self.path_table, self.subset_size)
            return random_dh.generate_subset(initial_subset=largest_cc)
        return largest_cc


class ArgmaxDestroyHeuristic(DestroyHeuristic):
    def generate_subset(self):
        cols = []
        for agent in self.instance.agents.values():
            path = agent.paths[agent.path_id]
            cols += [self.path_table.count_collisions_points_along_existing_path(path)]
        cols = np.array(cols)
        argmax_incides = np.argpartition(cols, -self.subset_size)[-self.subset_size:]
        argmax_ids = (argmax_incides + 1).tolist()
        return argmax_ids


class PriorityDestroyHeuristic(DestroyHeuristic):
    def generate_subset(self, argmax_size = 0):
        random_size = self.subset_size - argmax_size
        cols = []
        argmax_ids = []
        if argmax_size > 0:
            argmax_dh = ArgmaxDestroyHeuristic(self.instance, self.path_table, argmax_size)
            argmax_ids = argmax_dh.generate_subset()
        random_ids = [agent_id for agent_id in range(1, self.instance.num_agents+1) if agent_id not in argmax_ids]
        subset = argmax_ids + np.random.choice(random_ids, random_size, replace=False).tolist()
        return subset


class RandomWeightedDestroyHeuristic(DestroyHeuristic):
    def generate_subset(self, initial_subset = []):
        adj_matrix = self.path_table.get_collisions_matrix(self.instance.num_agents)

        # Weighting of vertex degrees proportional to deg(v_i) + 1 as shown in https://ojs.aaai.org/index.php/AAAI/article/view/21266
        prob_weights = get_degrees_of_all_vertices(adj_matrix) + 1

        # Remove chosen agents from available options (mask is 0 if already chosen and 1 otherwise)
        random_ids_mask = np.array([0 if agent_id in initial_subset else 1 for agent_id in range(1, self.instance.num_agents+1)])

        # Mask out already chosen agents
        prob_weights = random_ids_mask * prob_weights

        # Normalize probabilities to sum to 1
        prob_weights = prob_weights/prob_weights.sum()

        # Calculate number of agents left to choose
        random_size = self.subset_size - len(initial_subset)


        # Choose random agents
        random_subset = np.random.choice(range(1, self.instance.num_agents+1), random_size, replace=False, p=prob_weights)

        subset = initial_subset + random_subset.tolist()
        return subset