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

    table: defaultdict[tuple[int, int, int], set[int]]
    collision_matrix: np.ndarray
    num_of_collision_points: int

    def __init__(self, num_of_rows, num_of_cols, num_of_agents):
        self.table = defaultdict(set)
        self.collisions_matrix = np.zeros((num_of_agents + 1, num_of_agents + 1))
        self.num_of_collision_points = 0
        self.makespan = -1

    def insert_path(self, agent_id, path):
        for (x, y), t in zip(path, range(len(path))):
            self.insert_point(agent_id, x, y, t)

    def insert_point(self, agent_id, x, y, t):
        # print(f"Inserting agent {agent_id} at ({x},{y}) at time {t}")
        if self.table[x, y, t]:
            for agent in self.table[x, y, t]:
                self.collisions_matrix[agent_id, agent] = 1
                self.collisions_matrix[agent, agent_id] = 1
                self.num_of_collision_points += 1
        self.table[x, y, t].add(agent_id)

    def remove_path(self, agent_id, path):
        for (x, y), t in zip(path, range(len(path))):
            self.table[x, y, t].remove(agent_id)
            for agent in self.table[x, y, t]:
                self.collisions_matrix[agent_id, agent] = 0
                self.collisions_matrix[agent, agent_id] = 0
                self.num_of_collision_points -= 1

    def is_path_available(self, path):
        for (x, y), t in zip(path, range(len(path))):
            if self.table[x, y, t]:
                return False

    def count_collisions_points_along_path(self, path):
        return sum(len(self.table[x, y, t]) > 0 for (x, y), t in zip(path, range(len(path))))

    def count_collisions_points_along_existing_path(self, path):
        return sum(len(self.table[x, y, t]) > 1 for (x, y), t in zip(path, range(len(path))))

    def num_collisions(self, num_robots = 90):
        return self.num_collisions_in_robots(num_robots)

    def num_collisions_in_robots(self, num_robots = 90):
        return self.num_of_collision_points

    def get_collisions_matrix(self, num_robots):
        return self.collisions_matrix

    def get_agent_collisions_for_paths(self, agent: Agent.Agent, num_robots):
        n_paths = agent.n_paths
        matrix = np.zeros((n_paths,num_robots+1))
        for path_index, path in enumerate(agent.paths):
            for (x, y), t in zip(path, range(len(path))):
                if self.table[x, y, t]:
                    for colliding_agent_id in self.table[x, y, t]:
                        if colliding_agent_id != agent.id:
                            # print(f'**** agent {agent.id} collides with agent {colliding_agent_id}')
                            matrix[path_index][colliding_agent_id] = 1
        return matrix.sum(axis = 1).astype(int).tolist()

    def calculate_makespan(self):
        makespan = -1
        for (_,_,t), _ in self.table.items():
            makespan = max(t, makespan)
        self.makespan = makespan