from typing import Generator, NamedTuple
import itertools
import numpy as np
import Agent
from collections import defaultdict


class Vertex(NamedTuple):
    x: int
    y: int
    t: int

    def to_int(self) -> "Vertex":
        return int(self.x), int(self.y), int(self.t)


class Edge(NamedTuple):
    px: int
    py: int
    pt: int
    x: int
    y: int
    t: int

    def reverse(self) -> "Edge":
        """
        Given this edge:
        (px, py, pt) -> (x, y, t)

        The 'reversed' edge is:
        (x, y, pt)   -> (px, py, t)

        Meaning an agent came from my next location to my previous location,
        in reversed timestamps (and where t - pt == 1).
        """

        assert self.t - self.pt == 1, "got invalid edge, timestamp diff > 1"

        return self.x, self.y, self.pt, self.px, self.py, self.t

    def to_int(self) -> "Edge":
        return int(self.px), int(self.py), int(self.pt), int(self.x), int(self.y), int(self.t)


def iter_vertices(path: list[tuple[int, int]]) -> Generator[Vertex, None, None]:
    for (x, y), t in zip(path, range(len(path))):
        yield Vertex(x, y, t)


def iter_edges(path: list[tuple[int, int]]) -> Generator[Edge, None, None]:
    prev = None
    for (x, y), t in zip(path, range(len(path))):
        if prev is not None:
            px, py, pt = prev

            yield Edge(px, py, pt, x, y, t)

        prev = x, y, t


def iter_path(path: list[tuple[int, int]]) -> Generator[tuple[Vertex, Edge | None], None, None]:
    """
    Yields vertices and edges of a path, where the edge is the connection
    between the previous vertex to the current vertex.

    Note:
        The first edge is None, since there is no previous vertex for the
        first vertex.

    Args:
        path (list[tuple[int, int]]): the path to iterate over

    Yields:
        tuple[Vertex, Edge | None]: the vertex and edge (edge from previous vertex)
    """

    vertices = iter_vertices(path)
    edges = itertools.chain([None], iter_edges(path))

    for vertex, edge in zip(vertices, edges):
        yield vertex, edge


class PathTable:
    """
    Represents paths in the grid.

    The table is a dictionary with keys as tuples (i,j) representing the
    location in the grid.

    Each value is a list of sets. The list is indexed by time t, and the set
    contains the agent ids that are at location (i,j) at time t.
    """

    table: defaultdict[Vertex, set[int]]
    edges: defaultdict[Edge, set[int]]
    collision_matrix: np.ndarray
    num_of_collision_points: int
    makespan: int

    def __init__(self, num_of_rows, num_of_cols, num_of_agents):
        self.table = defaultdict(set)
        self.edges = defaultdict(set)
        self.collisions_matrix = np.zeros((num_of_agents + 1, num_of_agents + 1))
        self.num_of_collision_points = 0
        self.makespan = -1

    def insert_path(self, agent_id, path):
        for vertex, edge in iter_path(path):
            self.insert_vertex(agent_id, vertex)

            if edge is not None:
                self.insert_edge(agent_id, edge)

    def insert_vertex(self, agent_id: int, vertex: Vertex):
        # print(f"Inserting agent {agent_id} at ({x},{y}) at time {t}")
        if self.table[vertex]:
            for agent in self.table[vertex]:
                self.collisions_matrix[agent_id, agent] = 1
                self.collisions_matrix[agent, agent_id] = 1
                self.num_of_collision_points += 1
        self.table[vertex].add(agent_id)

    def insert_edge(self, agent_id: int, edge: Edge):
        """
        Note:
            Detects 'swap' collisions only, since other collisions are
            handled by insert_vertex.

            A 'swap' collision occurs if these two edges exist:
            (px, py, pt) -> (x, y, t)
            (x, y, pt)   -> (px, py, t)

            Meaning an agent came from my next location to my previous location,
            in reversed timestamps (and where t - pt == 1).
        """

        collision_edge = edge.reverse()

        self.edges[edge].add(agent_id)

        for agent in self.edges[collision_edge]:
            self.collisions_matrix[agent_id, agent] = 1
            self.collisions_matrix[agent, agent_id] = 1
            self.num_of_collision_points += 1

    def remove_path(self, agent_id, path):
        for vertex, edge in iter_path(path):
            self.table[vertex].remove(agent_id)
            for agent in self.table[vertex]:
                self.collisions_matrix[agent_id, agent] = 0
                self.collisions_matrix[agent, agent_id] = 0
                self.num_of_collision_points -= 1

            if edge is not None:
                self.edges[edge].remove(agent_id)
                collision_edge = edge.reverse()

                for agent in self.edges[collision_edge]:
                    self.collisions_matrix[agent_id, agent] = 0
                    self.collisions_matrix[agent, agent_id] = 0
                    self.num_of_collision_points -= 1

    def is_path_available(self, path):
        for vertex, edge in iter_path(path):
            if self.table[vertex]:
                return False

            if edge is not None and self.edges[edge.reverse()]:
                return False

    def count_collisions_points_along_path(self, path):
        vertices = set.union(*(self.table[v] for v in iter_vertices(path)))
        edges = set.union(*(self.edges[e.reverse()] for e in iter_edges(path)))

        return len(set.union(vertices, edges))

    def count_collisions_points_along_existing_path(self, path):
        return self.count_collisions_points_along_path(path)

    def num_collisions(self, num_robots = 90):
        return self.num_collisions_in_robots(num_robots)

    def num_collisions_in_robots(self, num_robots = 90):
        return self.num_of_collision_points

    def get_collisions_matrix(self, num_robots):
        return self.collisions_matrix[: num_robots + 1, : num_robots + 1]

    def get_agent_collisions_for_path(
        self, agent_id: int, path
    ) -> tuple[list[tuple[Vertex, set[int]]], list[tuple[Edge, set[int]]]]:
        """
        Get all the collisions for a path.

        Args:
            agent_id (int): the agent id
            path (list[tuple[int, int]]): the path

        Returns:
            tuple[list[tuple[Vertex, set[int]], tuple[Edge, set[int]]]]:
                the vertices and edges with the agents that
                collide with the given agent_id
        """

        vertices = [
            (v, self.table[v] - {agent_id})
            for v in iter_vertices(path)
            if len(self.table[v] - {agent_id}) > 0
        ]
        edges = [
            (e, self.edges[e.reverse()] - {agent_id})
            for e in iter_edges(path)
            if len(self.edges[e.reverse()] - {agent_id}) > 0
        ]

        return vertices, edges

    def get_agent_collisions_for_paths(self, agent: Agent.Agent, num_robots):
        n_paths = agent.n_paths
        matrix = np.zeros((n_paths,num_robots+1))
        for path_index, path in enumerate(agent.paths):
            for vertex, edge in iter_path(path):
                vertex_collisions = self.table[vertex]
                edge_collisions = set() if edge is None else self.edges[edge.reverse()]
                collisions = set.union(vertex_collisions, edge_collisions)

                for colliding_agent_id in collisions:
                    if colliding_agent_id != agent.id:
                        # print(f'**** agent {agent.id} collides with agent {colliding_agent_id}')
                        matrix[path_index][colliding_agent_id] = 1

        return matrix.sum(axis = 1).astype(int).tolist()

    def calculate_makespan(self):
        makespan = -1
        for (_,_,t), _ in self.table.items():
            makespan = max(t, makespan)
        self.makespan = makespan
