import pandas as pd
import numpy as np
import instance
import os
import time

def log():
    dates = pd.date_range("20130101", periods=6)
    df = pd.DataFrame(np.random.randn(6, 4), columns=list("ABCD"))
    print(df)
    df.to_csv('experiments.csv', mode='a', header=False, index = False)


class Logger:
    def __init__(self, save_path, instance:instance.instance):
        self.save_path = save_path
        self.instance = instance
        self.instance_column_names = ['map_name','agent_name', 'n_agents', 'solver_name']
        solver_name = self.instance.instance_name
        n_agents = self.instance.num_agents
        map_name = os.path.basename(self.instance.map_f_name)
        agent_name = os.path.basename(self.instance.agent_fname)
        self.instance_columns = [map_name,agent_name,n_agents, solver_name]

    def start(self):
        self.start_time = time.time()
    def end(self):
        self.end_time = time.time()

    def runtime(self):
        return 1000*(self.end_time-self.start_time)
    def log(self, n_solved):
        solved_percent = 100*(n_solved/self.instance.num_agents)
        other_columns = [self.runtime(), solved_percent]
        other_column_names = ['runtime (ms)', 'percent_solved']
        columns = self.instance_columns + other_columns
        column_names = self.instance_column_names + other_column_names
        header = os.stat(self.save_path).st_size == 0

        df = pd.DataFrame([columns], columns=column_names)
        df.to_csv(self.save_path, mode='a', index=False, header = header)




