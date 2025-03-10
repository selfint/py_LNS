import copy

import instance
import PathTable
import numpy as np
import itertools
import CBS
import torch

from benchmark_utils import benchmark

class NeighborhoodRepair:
    def __init__(self,*args, **kwargs):
        self.instance = kwargs['instance']
        self.path_table = kwargs['path_table']
        self.agent_subset = kwargs['agent_subset']
        self.verbose = kwargs['verbose']

    def destroy_neighborhood(self):
        for agent_id in self.agent_subset:
            path_id = self.instance.agents[agent_id].path_id
            self.path_table.remove_path(agent_id, self.instance.agents[agent_id].paths[path_id])
            self.instance.agents[agent_id].path_id = -1

    def run(self):
        raise NotImplementedError()
class PPNeighborhoodRepair(NeighborhoodRepair):
    # reroute_agent rolling mean execution time: 258.485603 μs (over 10000 runs)
    # @benchmark(n=10_000)
    def reroute_agent(self, agent_id):
        if self.verbose:
            print(f'\n**** Rerouting agent {agent_id} ****')

        best_path_id = 0
        best_path = self.instance.agents[agent_id].paths[best_path_id]
        best_path_cols = np.inf

        for path_id in range(len(self.instance.agents[agent_id].paths)):
            path = self.instance.agents[agent_id].paths[path_id]
            path_cols = self.path_table.count_collisions_points_along_path(path)
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
        if self.verbose:
            print(f'     num_cols:{self.path_table.num_collisions(self.instance.num_agents)}')

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
            num_cols = self.path_table.num_collisions(self.instance.num_agents)
            if num_cols < best_num_cols:
                best_num_cols = num_cols
                best_path_selection = path_selection
                if self.verbose:
                    print(f'\n      Found new path selection: {path_selection} ****')
                    print(f'\n      New number of collisions: {best_num_cols} ****')

        self.reroute_agents(best_path_selection)


class RankPPNeighborhoodRepair(NeighborhoodRepair):
    def __init__(self, *args, **kwargs):
        self.agent_cost_type = kwargs.pop('agent_cost_type')
        super(RankPPNeighborhoodRepair,self).__init__(*args, **kwargs)


    def calculate_agent_path_costs(self):
        return {agent_id: self.path_table.get_agent_collisions_for_paths(self.instance.agents[agent_id], self.instance.num_agents) for agent_id in self.temp_subset}

    def calculate_total_agent_cost(self, path_costs_dict):
        res_dict = dict()
        if self.agent_cost_type == 'min':
            res_dict =  {agent_id: min(path_degrees) for agent_id, path_degrees in path_costs_dict.items()}
        if self.agent_cost_type == 'mean':
            res_dict =  {agent_id: sum(path_degrees)/len(path_degrees) for agent_id, path_degrees in path_costs_dict.items()}
        return dict(sorted(res_dict.items(), key=lambda item: item[1]))
    def reroute_agent(self, agent_id, path_costs_dict):
        if self.verbose:
            print(f'\n**** Rerouting agent {agent_id} ****')

        # Choose path with minimum
        new_path_id = int(np.argmin(np.array(path_costs_dict[agent_id])))
        new_path = self.instance.agents[agent_id].paths[new_path_id]


        self.path_table.insert_path(agent_id, new_path)
        self.instance.agents[agent_id].path_id = new_path_id

    def run(self):
        self.destroy_neighborhood()
        self.temp_subset = self.agent_subset.copy()
        while len(self.temp_subset) > 0:
            # Calculate cost of paths of each robot w.r.t each path
            # Cost = number of collisions (with other robots)  if this path is chosen
            path_costs_dict = self.calculate_agent_path_costs()

            # Aggregates cost into one number and sorts by it
            agent_costs = self.calculate_total_agent_cost(path_costs_dict)

            chosen_agent_id = list(agent_costs.keys())[0]

            self.reroute_agent(chosen_agent_id, path_costs_dict)

            self.temp_subset.remove(chosen_agent_id)

class CBSNeighborhoodRepair(NeighborhoodRepair):

    def reroute_agents(self, agent_path_selection):
        self.destroy_neighborhood()
        for agent_id, path_id in zip(self.agent_subset, agent_path_selection):
            path = self.instance.agents[agent_id].paths[path_id]
            self.path_table.insert_path(agent_id, path)
            self.instance.agents[agent_id].path_id = path_id

    def format_solution_to_paths(self, cbs_solution):
        path_ids = cbs_solution[self.agent_subset_normalized].argmax(-1).tolist()
        self.reroute_agents(path_ids)

    def run(self):
        self.agent_subset_normalized = [a_id-1 for a_id in self.agent_subset]
        if self.verbose:
            print(f'\n**** Starting CBS search ****')

        solution = torch.tensor(
            [self.instance.agents[i + 1].path_id for i in range(self.instance.num_agents)]
        )
        solution_one_hot = torch.nn.functional.one_hot(solution, self.instance.n_paths)
        solver = CBS.CBS(self.instance, solution_one_hot, self.verbose, self.agent_subset_normalized)
        cbs_solution, cbs_lns_col_count = solver.search()
        self.format_solution_to_paths(cbs_solution)

