import aux
import numpy as np
from Agent import Agent


class instance:
    def __init__(self, map_f_name,agent_fname, instance_name, verbose = True, n_paths = 10, agent_path_temp = 1):
        self.map_f_name = map_f_name
        self.agent_fname = agent_fname
        self.instance_name = instance_name
        self.verbose = verbose
        self.n_paths = n_paths
        self.agent_path_temp = agent_path_temp
        np.random.seed(2705)

        self.load_map()
        self.load_agents()
    def load_map(self):
        with open(self.map_f_name) as map_file:
            headers = [map_file.readline() for i in range(4)]
            self.num_of_rows = aux.extract_digits(headers[1])
            self.num_of_cols = aux.extract_digits(headers[2])
            file_string = map_file.read()
        self.map_size = self.num_of_rows * self.num_of_cols
        file_string = [c for c in file_string if c != '\n']
        self.map = np.array(list(map(lambda c: int(c != '.'), file_string))).reshape(self.num_of_rows, self.num_of_cols)
        if self.verbose:
            print(f'\n**** Successfully loaded map from: {self.map_f_name} ****')
            print(f'     number of rows: {self.num_of_rows}')
            print(f'     number of columns: {self.num_of_cols}\n')
    def print_map(self):
        for line in self.map:
            print(''.join(aux.bin_to_char(num) for num in line))

    def print_map_with_agents(self):
        map = self.map.copy()
        for loc in zip(self.start_locations, self.goal_locations):
            print(loc)
            map[loc[0][0], loc[0][1]] = 2
            map[loc[1][0], loc[1][1]] = 3

        for line in map:
            print(''.join(aux.bin_to_char(num) for num in line))

    def print_agents(self):
        map = np.zeros_like(self.map)
        for loc in zip(self.start_locations, self.goal_locations):
            print(loc)
            map[loc[0][0], loc[0][1]] = 2
            map[loc[1][0], loc[1][1]] = 3

        for line in map:
            print(''.join(aux.bin_to_char(num) for num in line))

    def load_agents(self):
        with open(self.agent_fname) as agent_file:
            lines = agent_file.readlines()[1:]
        lines = [line.split('\t')[4:-1] for line in lines]
        lines = [[int(num) for num in line] for line in lines]
        self.start_locations = [line[:2] for line in lines]
        self.goal_locations = [line[2:] for line in lines]
        self.num_agents = len(self.start_locations)
        if self.verbose:
            print(f'**** Successfully loaded {self.num_agents} agents! ****\n')
        self.agents = {i: Agent(i, s, t, self.n_paths) for i, s, t in (zip(range(1, self.num_agents+1), self.start_locations, self.goal_locations))}
        for agent in self.agents.values():
            agent.generate_paths(self.num_of_rows, self.num_of_cols, self.agent_path_temp)
    def is_in_map(self, loc):
        pos_int = loc[0] * loc[1]
        return pos_int >= 0 and pos_int < self.map_size
    def is_valid_move(self, cur_loc, next_loc):
        # Check if remains in map
        if not self.is_in_map(next_loc):
            return False

        # Check if obstacle
        if self.map[next_loc[0], next_loc[1]]:
            return False

        # Make sure next loc is reachable in single move
        return aux.manhattan_dist(cur_loc, next_loc) < 2
