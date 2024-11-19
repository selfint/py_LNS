import itertools
import numpy as np
import Agent
from collections import defaultdict
class PathTable:
    """
    Represents paths in the grid.

    The table is a dictionary with keys as tuples (i,j) representing the
    location in the grid.

    Each value is a list of sets. The list is indexed by time t, and the set
    contains the agent ids that are at location (i,j) at time t.
    """

    table: dict[tuple[int, int, int], list[set[int]]]

    def __init__(self, num_of_rows, num_of_cols):
        self.table = defaultdict(lambda: [])

    def insert_path(self, agent_id, path):
        for (x, y), t in zip(path, range(len(path))):
            self.insert_point(agent_id, x,y ,t)

    def insert_point(self,agent_id, x, y, t):
        self.table[x, y, t].append(agent_id)

    def remove_path(self, agent_id, path):
        for (x, y), t in zip(path, range(len(path))):
            del self.table[x, y, t][agent_id]

    def is_path_available(self, path):
        for (x, y), t in zip(path, range(len(path))):
            if self.table[x, y, t]:
                return False

    def count_collisions_points_along_path(self, path):
        return len([self.table[x, y, t] for x, y, t in zip(path, range(len(path)))])

    def count_collisions_points_along_existing_path(self, path):
        return len([self.table[x, y, t] for x, y, t in zip(path, range(len(path))) if len(self.table[x, y, t]) > 1])

    def num_collisions(self):
        return self.num_collisions_in_robots()
        count = 0
        for time_list in self.table.values():
            for point_set in time_list:
                if len(point_set) > 1:
                    count += 1
        return count

    def num_collisions_in_robots(self, num_robots = 90):
        return self.get_collisions_matrix(num_robots).sum()//2

    def get_collisions_matrix(self, num_robots):
        matrix = np.zeros((num_robots+1,num_robots+1))
        for ((_x, _y), t), agent_ids in self.table.items():
            if len(agent_ids) > 1:
                for i,j in itertools.combinations(agent_ids, 2):
                    matrix[i, j] = 1
                    matrix[j, i] = 1
        return matrix.astype(int)

    def get_agent_collisions_for_paths(self, agent: Agent.Agent, num_robots):
        n_paths = agent.n_paths
        matrix = np.zeros((n_paths,num_robots+1))
        for path_index, path in enumerate(agent.paths):
            for (x, y), t in zip(path, range(len(path))):
                if self.table[x, y, t]:
                    for colliding_agent_id in self.table[x, y, t]:
                        if colliding_agent_id != agent.id:
                            #print(f'**** agent {agent.id} collides with agent {colliding_agent_id}')
                            matrix[path_index][colliding_agent_id] = 1
        return matrix.sum(axis = 1).astype(int).tolist()







