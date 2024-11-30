from typing import List

from lns2.neighborhood_picker import NeighborhoodPicker


class RandomNeighborhoodPicker(NeighborhoodPicker):
    def __init__(self, n_size):
        self.n_size = n_size

    def pick(self, paths: dict[int, List[int]]):
        assert self.n_size <= len(paths)
        return list(paths.keys())[:self.n_size]
