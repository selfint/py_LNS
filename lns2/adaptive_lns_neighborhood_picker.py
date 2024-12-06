from lns2.collision_based_neighborhood_picker import CollisionBasedNeighborhoodPicker
from lns2.failure_based_neighborhood_picker import FailureBasedNeighborhoodPicker
from lns2.neighborhood_picker import NeighborhoodPicker
from numpy.random import choice

from lns2.random_neighborhood_picker import RandomNeighborhoodPicker


class AdaptiveLNSNeighborhoodPicker(NeighborhoodPicker):
    def __init__(self, n_size, gamma: float = 0.1):
        self.n_size = n_size
        self.weights: list[int] = [1, 1, 1]
        self.gamma: float = gamma
        self.neighborhood_pickers: list[NeighborhoodPicker] = [CollisionBasedNeighborhoodPicker(n_size),
                                                               RandomNeighborhoodPicker(n_size),
                                                               FailureBasedNeighborhoodPicker(n_size)]
        self.current_picker = None

    def pick(self, paths: dict[int, list[int]]):
        weights = [self.weights[i]/sum(self.weights) for i in range(len(self.weights))]
        self.current_picker_index: int = choice(list(range(1, len(self.neighborhood_pickers))), 1, p=weights)[0]
        return self.neighborhood_pickers[self.current_picker_index].pick(paths)

    def update(self, cp_diff: int):
        """
        cp_diff: collisions before - collisions after
        """
        self.weights[self.current_picker_index] = self.gamma * max(0, cp_diff) + (1 - self.gamma) * self.weights[self.current_picker_index]