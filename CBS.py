import instance
import torch_parallel_lns
import torch
from copy import deepcopy
import numpy as np
from scipy.sparse import csr_matrix
import sys




class CBS:
    def __init__(self, instance : instance.instance, initial_solution = None, verbose = True):
        self.open = set()
        self.closed = set()
        self.instance = instance
        self.cmatrix = torch_parallel_lns.build_cmatrix(list(self.instance.agents.values()))
        self.cost_matrix = torch_parallel_lns.build_cost_matrix(list(self.instance.agents.values())).int()
        self.initial_solution = initial_solution
        self.verbose = verbose
        self.expanded = 0

    def search(self):
        initial_node = CTNode(self.instance.num_agents, self.instance.n_paths)
        if self.initial_solution is not None:
            initial_node.solution = self.initial_solution.int()
        # Line 3 of CBS
        initial_node.cost = self.compute_solution_cost(initial_node.solution)

        solution_matrix = torch_parallel_lns.solution_cmatrix(self.cmatrix, initial_node.solution)
        initial_node.collisions = solution_matrix.nonzero()
        initial_node.col_count = len(initial_node.collisions)//2
        best_sol = initial_node

        # Line 4 of CBS
        self.open |= {initial_node}

        # Line 5 of CBS
        while self.open:

            # Line 6 of CBS
            p = min(self.open)
            # Line 7 of CBS
            self.open -= {p}
            self.expanded += 1
            if self.expanded > 10000:
                return best_sol.solution, best_sol.col_count

            if (len(self.open) & (len(self.open) - 1) == 0) and len(self.open) != 0 and self.verbose:
                print(f'**** popped node with cost {p.cost} from open of size {len(self.open)}')

            if self.verbose:
                pass#print(f'**** popped node with cost {p.cost} from open of size {len(self.open)}')

            solution_matrix = torch_parallel_lns.solution_cmatrix(self.cmatrix, p.solution.clip(0,1))
            p.collisions = solution_matrix.nonzero()
            p.col_count = len(p.collisions) // 2

            if best_sol == None or best_sol.col_count > p.col_count:
                if self.verbose:
                    print(f'**** New best solution found!:')
                    print(f'     Collisions: {p.col_count}')

                best_sol = p

            # Lines 7-8 of CBS
            if p.col_count == 0:
                # Line 9 of CBS
                if self.verbose:
                    print(f'Found solution!')
                    print(f'Expanded nodes: {self.expanded}')
                    print(f'{p.solution}')
                return p.solution, p.col_count

            # Line 10 of CBS
            #c = p.collisions[torch.randperm(p.col_count)[0]].tolist()
            c = p.collisions[0].tolist()

            # Line 11 of CBS
            for agent_id, other_agent in zip(c, c[-1::-1]):
                # Lines 12-15 of CBS
                a = deepcopy(p)
                a.add_restriction(agent_id,a.get_path_id(agent_id), other_agent, a.get_path_id(other_agent))
                #other_agent_path_id = a.get_path_id(other_agent)
                path_restriction = 0#self.compute_collisions_between_path_and_agent(other_agent, other_agent_path_id, agent_id)
                if a.update_solution(agent_id, path_restriction):
                    # Line 16 of CBS
                    a.cost = self.compute_solution_cost(a.solution)

                    # Line 17 of CBS
                    self.open |= {a}
        print(best_sol.col_count)
        print(f'Expanded nodes: {self.expanded}')
        return best_sol.solution, best_sol.col_count

    def compute_solution_cost(self, solution : torch_parallel_lns.Solution):
        cost_matrix = solution @ (self.cost_matrix.T)
        return torch.diag(cost_matrix).sum().item()
    def compute_collisions_between_path_and_agent(self, agent_1_id, agent1_path_id, agent_2_id):
        cmatrix_view = self.cmatrix.view(self.instance.num_agents, self.instance.n_paths, self.instance.num_agents, self.instance.n_paths)
        return cmatrix_view[agent_1_id,agent1_path_id,agent_2_id,:]

class CTNode:
    def __init__(self, num_agents, n_paths):
        self.n_paths = n_paths
        # Line 1 of CBS
        # Constraints are part of computing solutions
        # TODO: if first path of each agent sucks nothing will be checked
        # Line 2 of CBS
        # Chosen : 1, Available : 0, Unallowed: -1
        self._solution = torch.zeros((num_agents, n_paths)).int()
        self._solution[:, 0] = 1

        self._restrictions = {i: [] for i in range(num_agents)}

        self.cost = 0
        self.collisions = 0
        self.col_count = 0

    def update_solution(self, agent_id, path_restriction):
        # Apply Restriction
        self._solution[agent_id] = self.get_restrictions(agent_id)
        if len((self._solution[agent_id] == 0).nonzero()) == 0:
            return False
        index = (self._solution[agent_id] == 0).nonzero()[0].item()
        self._solution[agent_id][index] = 1
        return True

        '''# Apply Restriction
        self._solution[agent_id] = self._solution[agent_id].masked_fill(path_restriction == 1, -1)
        if len((self._solution[agent_id] == 0).nonzero()) == 0:
            return False
        index = (self._solution[agent_id] == 0).nonzero()[0].item()
        self._solution[agent_id][index] = 1
        assert len((self._solution[agent_id] == 1).nonzero()) == 1
        return True'''

    def get_path_id(self, agent_id):
        return ((self.solution[agent_id] == 1).nonzero(as_tuple=True)[0]).item()

    def add_restriction(self, agent1_id, agent1_path_id, agent2_id, agent2_path_id):
        self._restrictions[agent1_id] += [(agent1_path_id, agent2_id, agent2_path_id)]

    def get_restrictions(self, agent_id):
        paths = torch.zeros(self.n_paths)
        for agent1_path_id, agent2_id, agent2_path_id in self._restrictions[agent_id]:
            if self._solution[agent2_id, agent2_path_id] == 1:
                paths[agent1_path_id] = -1
        return paths
    @property
    def solution(self):
        return self._solution.clip(0,1)

    @solution.setter
    def solution(self, val):
        self._solution = val

    def __lt__(self, other):
        #return self.cost < other.cost or (self.cost == other.cost and self.col_count < other.col_count)
        return self.col_count < other.col_count or (self.col_count == other.col_count and self.cost < other.cost)
