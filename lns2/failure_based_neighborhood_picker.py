from lns2.collision_based_neighborhood_picker import CollisionBasedNeighborhoodPicker
from lns2.neighborhood_picker import NeighborhoodPicker
from numpy.random import choice

from lns2.random_neighborhood_picker import RandomNeighborhoodPicker


class FailureBasedNeighborhoodPicker(NeighborhoodPicker):
    def __init__(self, n_size, ):
        self.n_size = n_size


    def pick(self, paths: dict[int, list[int]]):
        raise NotImplementedError