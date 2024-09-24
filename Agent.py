from PathGenerator import *


class Agent:
    def __init__(self, instance, agent_id, start_loc, end_loc, n_paths):
        self.id = agent_id
        self.start = start_loc
        self.end = end_loc
        self.n_paths = n_paths
        self.path_id = -1
        self.instance = instance

    def generate_paths(self, num_rows, num_cols, temp):
        self.paths = [create_random_step_path(self.instance, self.start, self.end, num_rows, num_cols, temp) for i in range(self.n_paths)]

