from abc import ABC, abstractmethod
from typing import List


class NeighborhoodPicker(ABC):
    @abstractmethod
    def pick(self, paths: dict[int: List[int]]) -> list[int]:
        """
        Given a dict of agents paths e.g: {agent_id: [v1, v2, ...]}, pick a subset of agents to be in the neighborhood.
        """
        pass
