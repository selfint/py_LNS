from collections import defaultdict
from typing import List, Tuple
from intervaltree import Interval, IntervalTree


class SafeIntervalTable:
    def __init__(self, vertices: List[int], paths: dict[int, List[int]], hard_obstacles: List[Tuple[int, int]]):
        self.paths = paths
        self.hard_obstacles = hard_obstacles
        self.mapping: dict[int, IntervalTree] = dict()
        for vertex in vertices:
            safe_interval = IntervalTree()
            safe_interval[0: float('inf')] = 'safe'
            self.mapping[vertex] = safe_interval

        for hard_obstacle in hard_obstacles:
            self.mapping[hard_obstacle[0]].chop(hard_obstacle[1], hard_obstacle[1] + 1)

        for agent_id, path in paths.items():
            for i, (vertex, time) in enumerate(path):
                self.mapping[vertex].chop(time, time + 1)

    def is_safe(self, neighbor, next_time):
        return self.mapping[neighbor].at(next_time)





