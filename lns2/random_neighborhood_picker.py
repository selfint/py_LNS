from numpy.random import choice
from typing import List
from PathTable import PathTable
from neighborhood_picker import NeighborhoodPicker


class RandomNeighborhoodPicker(NeighborhoodPicker):
    def __init__(self, n_size):
        self.n_size = n_size

    def pick(self, paths: dict[int, List[int]]):
        p_table = PathTable(0, 0, len(paths))
        for agent_id, path in paths.items():
            p_table.insert_path(agent_id, path)

        agents_collisions_list: list[(int, int)] = []
        for agent_id, path in paths.items():
            agents_collisions_list.append((agent_id, sum(p_table.get_collisions_matrix(None))[agent_id]))
        number_of_total_collisions = sum([agent_collisions_pair[1] + 1 for agent_collisions_pair in agents_collisions_list])
        weights = [(collisions + 1)/number_of_total_collisions for _agent, collisions in agents_collisions_list]
        agents = [agent for agent, _collisions in agents_collisions_list]
        return choice(agents, self.n_size, p=weights, replace=False)

