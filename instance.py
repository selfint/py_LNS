import aux
import numpy as np
from Agent import Agent
from graphMethods import create_graph_from_map


class instance:
    def __init__(self, map_f_name,agent_fname, instance_name, verbose = True, n_paths = 3, agent_path_temp = 1):
        self.map_f_name = map_f_name
        self.agent_fname = agent_fname
        self.instance_name = instance_name
        self.verbose = verbose
        self.n_paths = n_paths
        self.agent_path_temp = agent_path_temp
        np.random.seed(2717)

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
        self.create_map_graph()
        if self.verbose:
            print(f'\n**** Successfully loaded map from: {self.map_f_name} ****')
            print(f'     number of rows: {self.num_of_rows}')
            print(f'     number of columns: {self.num_of_cols}\n')

    def create_map_graph(self):
        self.map_graph = create_graph_from_map(self.map, self.num_of_rows, self.num_of_cols, verbose = self.verbose)
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
        self.verify_and_repair_agents()
        self.num_agents = len(self.start_locations)
        if self.verbose:
            print(f'**** Successfully loaded {self.num_agents} agents! ****\n')
        self.agents = {i: Agent(self, i, s, t, self.n_paths) for i, s, t in (zip(range(1, self.num_agents+1), self.start_locations, self.goal_locations))}
        for agent in self.agents.values():
            agent.generate_paths()

    def verify_and_repair_agents(self):
        new_starts = []
        for start in self.start_locations:
            if tuple(start) not in self.map_graph.nodes:
                start_x = start[0]
                start_y = start[1]
                new_locs = [(start_x-1,start_y),
                            (start_x+1,start_y),
                            (start_x,start_y-1),
                            (start_x,start_y+1)]
                for new_loc in new_locs:
                    if new_loc in self.map_graph.nodes:
                        start = list(new_loc)
                        break
            new_starts += [start]
        self.start_locations = [tuple(start) for start in new_starts]

        new_goals = []
        for goal in self.goal_locations:
            if tuple(goal) not in self.map_graph.nodes:
                goal_x = goal[0]
                goal_y = goal[1]
                new_locs = [(goal_x - 1, goal_y),
                            (goal_x + 1, goal_y),
                            (goal_x, goal_y - 1),
                            (goal_x, goal_y + 1)]
                for new_loc in new_locs:
                    if new_loc in self.map_graph.nodes:
                        goal = list(new_loc)
                        break
            new_goals += [goal]
        self.goal_locations = [tuple(goal) for goal in new_goals]
        for start, goal in zip (self.start_locations, self.goal_locations):
            assert tuple(start) in self.map_graph.nodes
            assert tuple(goal) in self.map_graph.nodes
        pass



    def is_in_map(self, loc):
        return 0 <= loc[0] < self.num_of_rows and 0 <= loc[1] < self.num_of_cols
    def is_valid_move(self, cur_loc, next_loc):
        # Check if remains in map
        if not self.is_in_map(next_loc):
            return False

        # Check if obstacle
        if self.map[next_loc[0], next_loc[1]] == 1:
            return False

        # Make sure next loc is reachable in single move
        return aux.manhattan_dist(cur_loc, next_loc) < 2
