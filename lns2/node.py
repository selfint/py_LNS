from dataclasses import dataclass
from intervaltree import Interval


@dataclass
class Node:
    """A node in a linked list."""
    vertex: int
    safe_interval: Interval
    id: int
    is_goal: bool
    g: int
    h: int
    f: int
    c: int
    path: list[int]
    _priority: tuple[int, int] = None

    def __post_init__(self):
        # Comparing c and then f
        self._priority = (self.c, self.f)

    def __lt__(self, other: "Node"):
        # for heapq to give the smallest element
        return self._priority < other._priority
