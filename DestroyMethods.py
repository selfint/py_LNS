import numpy as np

class DestroyHeuristic:
    def __init__(self, instance, subset_size):
        self.instance = instance
        self.subset_size = subset_size

    def generate_subset(self):
        raise NotImplementedError()


class RandomDestroyHeuristic(DestroyHeuristic):
    def generate_subset(self):
        return np.random.choice(range(1, self.instance.num_agents+1), self.subset_size, replace=False)


class PriorityDestroyHeuristic(DestroyHeuristic):
    def generate_subset(self):
        return np.random.choice(range(1, self.instance.num_agents+1), self.subset_size, replace=False)


