import instance
import PathTable
import numpy as np
import itertools


class NeighborhoodRepair:
    def __init__(self,instance:instance.instance, path_table: PathTable.PathTable, agent_subset, verbose = True):
        self.instance = instance
        self.path_table = path_table
        self.agent_subset = agent_subset
        self.verbose = verbose

    def destroy_neighborhood(self):
        for agent_id in self.agent_subset:
            path_id = self.instance.agents[agent_id].path_id
            self.path_table.remove_path(agent_id, self.instance.agents[agent_id].paths[path_id])
            self.instance.agents[agent_id].path_id = -1

    def run(self):
        raise NotImplementedError()
class PPNeighborhoodRepair(NeighborhoodRepair):
    def reroute_agent(self, agent_id):
        if self.verbose:
            print(f'\n**** Rerouting agent {agent_id} ****')

        best_path_id = 0
        best_path = self.instance.agents[agent_id].paths[best_path_id]
        best_path_cols = np.inf

        for path_id in range(len(self.instance.agents[agent_id].paths)):
            path = self.instance.agents[agent_id].paths[path_id]
            path_cols = self.path_table.count_collisions_along_path(path)
            if path_cols < best_path_cols:
                best_path_id = path_id
                best_path_cols = path_cols
                best_path = path
            if best_path_cols == 0:
                break

        if self.verbose:
            print(f'\n       New number of collisions: {best_path_cols} ****')

        self.path_table.insert_path(agent_id, best_path)
        self.instance.agents[agent_id].path_id = best_path_id

    def run(self):
        self.destroy_neighborhood()
        print(f'     num_cols:{self.path_table.num_collisions()}')

        for agent_id in self.agent_subset:#np.random.permutation(self.agent_subset):
            self.reroute_agent(agent_id)

class ExhaustiveNeighborhoodRepair(NeighborhoodRepair):

    def reroute_agents(self, agent_path_selection):
        self.destroy_neighborhood()
        for agent_id, path_id in zip(self.agent_subset, agent_path_selection):
            path = self.instance.agents[agent_id].paths[path_id]
            self.path_table.insert_path(agent_id, path)
            self.instance.agents[agent_id].path_id = path_id

    def run(self):
        if self.verbose:
            print(f'\n**** Starting Exhaustive search ****')

        best_num_cols = np.inf
        best_path_selection = -1
        path_range = list(range(len(self.instance.agents[1].paths)))
        subset_size = len(self.agent_subset)
        path_range_list = subset_size * [path_range]
        path_combinations = itertools.product(*path_range_list)

        # Iterate over all path selections
        for path_selection in path_combinations:
            self.reroute_agents(path_selection)
            num_cols = self.path_table.num_collisions()
            if num_cols < best_num_cols:
                best_num_cols = num_cols
                best_path_selection = path_selection
                if self.verbose:
                    print(f'\n      Found new path selection: {path_selection} ****')
                    print(f'\n      New number of collisions: {best_num_cols} ****')

        self.reroute_agents(best_path_selection)
