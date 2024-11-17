import itertools
import numpy as np
import Agent
class PathTable:
    """
    Represents paths in the grid.

    The table is a dictionary with keys as tuples (i,j) representing the
    location in the grid.

    Each value is a list of sets. The list is indexed by time t, and the set
    contains the agent ids that are at location (i,j) at time t.
    """

    table: dict[tuple[int, int], list[set[int]]]

    def __init__(self, num_of_rows, num_of_cols):
        self.table = dict()
        for i in range(0, num_of_rows+1):
            for j in range(0, num_of_cols+1):
                self.table[(i,j)] = []

    def insert_path(self, agent_id, path):
        for loc, t in zip(path, range(len(path))):
            self.insert_point(agent_id, tuple(loc), t)

    def insert_point(self,agent_id, loc, t):
        self.extend_table_to_time(loc, t+1) # Make sure point can contain the length of the path
        self.table[loc][t].add(agent_id)

    def remove_path(self, agent_id, path):
        for loc, t in zip(path, range(len(path))):
            self.remove_point(agent_id, tuple(loc), t)

    def remove_point(self,agent_id, loc, t):
        if len(self.table[loc]) > t and agent_id in self.table[loc][t]:
            self.table[loc][t].remove(agent_id)


    def extend_table_to_time(self, loc, t):
        if len(self.table[loc]) < t: # need to extend
            addition = [set() for i in range(t - len(self.table[loc]))]
            self.table[loc] += addition

    def is_path_available(self, path):
        for loc, t in zip(path, range(len(path))):
            if len(self.table[tuple(loc)]) > t and len(self.table[tuple(loc)][t]) > 0:
                return False
        return True

    def count_collisions_points_along_path(self, path):
        count = 0
        for loc, t in zip(path, range(len(path))):
            if len(self.table[tuple(loc)]) > t and len(self.table[tuple(loc)][t]) > 0:
                count += 1
        return count

    def count_collisions_points_along_existing_path(self, path):
        count = 0
        for loc, t in zip(path, range(len(path))):
            if len(self.table[tuple(loc)]) > t and len(self.table[tuple(loc)][t]) > 1:
                count += 1
        return count

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
        for time_list in self.table.values():
            for point_set in time_list:
                if len(point_set) > 1:
                    for i,j in itertools.combinations(point_set, 2):
                        matrix[i, j] = 1
                        matrix[j, i] = 1
        return matrix.astype(int)

    def get_agent_collisions_for_paths(self, agent: Agent.Agent, num_robots):
        n_paths = agent.n_paths
        matrix = np.zeros((n_paths,num_robots+1))
        for path_index, path in enumerate(agent.paths):
            for loc, t in zip(path, range(len(path))):
                if len(self.table[tuple(loc)]) > t:
                    for colliding_agent_id in self.table[tuple(loc)][t]:
                        if colliding_agent_id != agent.id:
                            #print(f'**** agent {agent.id} collides with agent {colliding_agent_id}')
                            matrix[path_index][colliding_agent_id] = 1
        return matrix.sum(axis = 1).astype(int).tolist()







