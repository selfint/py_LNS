class PathTable:
    def __init__(self, num_of_rows, num_of_cols):
        self.table = dict()
        for i in range(1, num_of_rows+1):
            for j in range(1, num_of_cols+1):
                self.table[(i,j)] = []

    def insert_path(self, agent_id, path):
        for loc, t in zip(path, range(len(path))):
            self.insert_point(agent_id, tuple(loc), t)

    def insert_point(self,agent_id, loc, t):
        self.extend_table_to_time(loc, t+1) # Make sure point can contain the length of the path
        self.table[loc][t].add(agent_id)


    def extend_table_to_time(self, loc, t):
        if len(self.table[loc]) < t: # need to extend
            addition = [set() for i in range(t - len(self.table[loc]))]
            self.table[loc] += addition



