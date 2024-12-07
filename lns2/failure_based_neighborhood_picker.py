from neighborhood_picker import NeighborhoodPicker


class FailureBasedNeighborhoodPicker(NeighborhoodPicker):
    def __init__(self, n_size):
        self.n_size = n_size

    def pick(self, paths: dict[int, list[int]]):
        # TODO it is unclear from the paper how should we implement this because its seems like you can get neighborhood
        #  of size 0
        raise NotImplementedError