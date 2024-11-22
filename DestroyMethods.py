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
            #random_dh = RandomDestroyHeuristic(self.instance, self.path_table, self.subset_size)
            return self.fill_cc_with_random_walks(largest_cc)#random_dh.generate_subset(initial_subset=largest_cc)
        return largest_cc
    def fill_cc_with_random_walks(self, connected_component):
        # Make copy of agents in connected component
        chosen_ids = connected_component[:]

        # Iteratively choose a random agent and start a random walk from  a point on its path until it collides with another agent
        while len(chosen_ids) < self.subset_size:
            agent_id = self.random_walk_from_random_point_until_collision(chosen_ids, connected_component)

            if agent_id > -1:
                assert agent_id not in chosen_ids
                chosen_ids += [agent_id]
        return chosen_ids

    def random_walk_from_random_point_until_collision(self, chosen_ids, connected_component):
        # Get pre-calculated make-span of solution
        makespan = self.path_table.makespan

        # Get graph representation of map
        map_graph = self.instance.map_graph

        # Choose random robot from connected component
        agent_id = int(np.random.choice(connected_component, 1))

        # Retrieve path for agent
        path_id = self.instance.agents[agent_id].path_id
        path = self.instance.agents[agent_id].paths[path_id]

        # Choose random point on agents path
        initial_t = int(np.random.choice(len(path), 1))

        loc = path[initial_t].tolist()

        # TODO: fix paths going through obstacles so this is unnecessary
        while tuple(loc) not in map_graph.nodes:
            initial_t = int(np.random.choice(len(path), 1))
            loc = path[initial_t].tolist()

        # Random walk along time-space grid to find colliding agent
        for t in range (initial_t, makespan+1):
            point = (*loc,t)
            agents_at_point = self.path_table.table[point]
            colliding_agents = set.difference(agents_at_point, chosen_ids)
            if len(colliding_agents) > 0:
                return int(np.random.choice(list(colliding_agents), 1))
            try:
                loc = get_random_neighbor(map_graph, tuple(loc))
            except:
                print('oh')


        return -1




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