import networkx as nx

from collision_based_neighborhood_picker import CollisionBasedNeighborhoodPicker
from failure_based_neighborhood_picker import FailureBasedNeighborhoodPicker
from neighborhood_picker import NeighborhoodPicker
from numpy.random import choice
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from random_neighborhood_picker import RandomNeighborhoodPicker


class AdaptiveLNSNeighborhoodPicker(NeighborhoodPicker):
    def __init__(self, n_size, graph: nx.Graph, gamma: float = 0.1):
        self.n_size = n_size
        self.gamma: float = gamma
        self.neighborhood_pickers: list[NeighborhoodPicker] = [CollisionBasedNeighborhoodPicker(n_size, graph),
                                                               RandomNeighborhoodPicker(n_size)]
                                                               # FailureBasedNeighborhoodPicker(n_size)]
        self.weights: list[int] = [1 for _ in range(len(self.neighborhood_pickers))]
        self.current_picker_index = -1

    def pick(self, paths: dict[int, list[int]]):
        weights = [self.weights[i]/sum(self.weights) for i in range(len(self.weights))]
        self.current_picker_index: int = choice(list(range(len(self.neighborhood_pickers))), 1, p=weights)[0]
        print(f"used {self.current_picker_index}")
        return self.neighborhood_pickers[self.current_picker_index].pick(paths)

    def update(self, cp_diff: int):
        """
        cp_diff: collisions before - collisions after
        """
        self.weights[self.current_picker_index] = self.gamma * max(0, cp_diff) + (1 - self.gamma) * self.weights[self.current_picker_index]