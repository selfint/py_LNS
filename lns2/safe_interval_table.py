import copy
from collections import defaultdict
from typing import List, Tuple
from intervaltree import Interval, IntervalTree


class SafeIntervalTable:
    def __init__(self, vertices: list[object], soft_obstacles: list[tuple[int, int]], hard_obstacles: list[tuple[int, int]]):
        self.soft_obstacles = soft_obstacles
        self.hard_obstacles = hard_obstacles
        self.mapping: dict[object, IntervalTree] = dict()
        for vertex in vertices:
            safe_interval = IntervalTree()
            safe_interval[0: float('inf')] = 'safe'
            vertex_soft_obstacles_timestamps = [t for (v, t) in soft_obstacles if v == vertex]
            for i in self.find_range_changes(vertex_soft_obstacles_timestamps):
                safe_interval.slice(i)
            self.mapping[vertex] = safe_interval

        for hard_obstacle in hard_obstacles:
            self.mapping[hard_obstacle[0]].chop(hard_obstacle[1], hard_obstacle[1] + 1)

    def is_safe(self, neighbor, next_time):
        return self.mapping[neighbor].at(next_time)

    def find_range_changes(self, timestamps):
        if not timestamps:
            return []
        result = []
        for i in range(len(timestamps)):
            if i == 0:  # Add the first number
                result.append(timestamps[i])
            elif timestamps[i] != timestamps[i - 1] + 1:  # Gap detected
                result.append(timestamps[i - 1] + 1)  # End of previous range
                result.append(timestamps[i])  # Start of new range
        result.append(timestamps[-1] + 1)
        return result

    def __getitem__(self, item):
        return copy.deepcopy(sorted(list(self.mapping[item])))



