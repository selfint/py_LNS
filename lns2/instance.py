import aux
import numpy as np
from Agent import Agent
from graphMethods import create_graph_from_map


class Instance:
    def __init__(self, map_file_name, agent_file_name):
        self.map_file_name = map_file_name
        self.agent_file_name = agent_file_name
        np.random.seed(2717)
        self.load_map()
        self.n_paths = 3
        self.load_agents()

    def load_map(self):
        with open(self.map_file_name) as map_file:
            headers = [map_file.readline() for i in range(4)]
            self.num_of_rows = aux.extract_digits(headers[1])
            self.num_of_cols = aux.extract_digits(headers[2])
            file_string = map_file.read()
        self.map_size = self.num_of_rows * self.num_of_cols
        file_string = [c for c in file_string if c != '\n']
        self.map = np.array(list(map(lambda c: int(c != '.'), file_string))).reshape(self.num_of_rows, self.num_of_cols)
        self.create_map_graph()
        print('**** Successfully loaded map from ****')

    def create_map_graph(self):
        self.map_graph = create_graph_from_map(self.map, self.num_of_rows, self.num_of_cols)

    def load_agents(self):
        with open(self.agent_file_name) as agent_file:
            lines = agent_file.readlines()[1:]
        lines = [line.split('\t')[4:-1] for line in lines]
        lines = [[int(num) for num in line] for line in lines]
        self.start_locations = [line[:2] for line in lines]
        self.goal_locations = [line[2:] for line in lines]
        self.verify_and_repair_agents()
        self.num_agents = len(self.start_locations)
        print(f'**** Successfully loaded {self.num_agents} agents! ****\n')
        self.agents = {i: Agent(self, i, s, t, self.n_paths) for i, s, t in (zip(range(1, self.num_agents+1), self.start_locations, self.goal_locations))}
        for agent in self.agents.values():
            agent.generate_paths()

    def get_map_graph(self):
        return self.map_graph

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

    def get_agents(self):
        return [(agent_id, start, goal) for agent_id, start, goal in zip(range(1, self.num_agents+1), self.start_locations, self.goal_locations)]