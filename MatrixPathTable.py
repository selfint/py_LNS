from typing import Generator, NamedTuple
import itertools
import numpy as np
import numpy.typing as npt
import Agent
from collections import defaultdict
import time
import scipy
from PathTable import iter_path, iter_edges, iter_vertices, Vertex, Edge


def benchmark(func):
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        print(f"{func.__name__} executed in {end_time - start_time:.6f} seconds")
        return result

    return wrapper


class MatrixPathTable:
    """
    Represents paths in the grid.

    On initialization, for each (agent, path_id) we get all other
    (agent, path_id) it collides with, and cache the results.
    Cache size = (agents X paths) ^ 2

    Then, we store the current paths ((agent_id, path_id) tuple) in use.
    """

    class AgentPath(NamedTuple):
        agent_id: int
        path_id: int
        locations: list[tuple[int, int]]

    path_collisions: scipy.sparse.csr_matrix
    """Sparse matrix representing collisions between path idx"""

    current_paths: set[tuple[int, int]]
    """Set of current (agent_id, path_id) in use"""

    current_idx: npt.NDArray
    """Indicator vector for current path_idx in use"""

    path_lengths: npt.NDArray
    """Vector storing length of each path_idx, for use in makespan"""

    def __init__(self, agents: list[Agent.Agent]):
        self.makespan = -1
        self.n_agents = len(agents) + 1
        self.n_paths = agents[0].n_paths
        self.all_paths = [
            MatrixPathTable.AgentPath(agent.id, path_id, path)
            for agent in agents
            for path_id, path in enumerate(agent.paths)
        ]

        self.current_paths = set()

        size = self.n_agents * self.n_paths
        self.current_idx = np.zeros(size, dtype=np.int8)

        # convert matrix to sparse
        self.path_collisions = scipy.sparse.csr_matrix(
            self._build_path_collisions(self.n_agents, self.n_paths, self.all_paths)
        )

        self.path_makespan = np.array(
            [-1] * self.n_paths + [len(path.locations) for path in self.all_paths]
        )

    # @benchmark
    # _build_path_collisions executed in 0.082134 seconds
    def _build_path_collisions(
        self, n_agents: int, n_paths: int, all_paths: list[AgentPath]
    ) -> np.array:
        size = n_agents * n_paths

        vertices = defaultdict(set)
        edges = defaultdict(set)
        for agent_id, path_id, locations in all_paths:
            for vertex, edge in iter_path(locations):
                vertices[vertex].add((agent_id, path_id))

                if edge is not None:
                    edges[edge].add((agent_id, path_id))
                    edges[edge.reverse()].add((agent_id, path_id))

        path_collisions = np.zeros((size, size), dtype=np.int8)
        for group in itertools.chain(vertices.values(), edges.values()):
            for (a_agent, a_path), (b_agent, b_path) in itertools.combinations(
                group, 2
            ):
                if a_agent == b_agent:
                    continue

                a = a_agent * n_paths + a_path
                b = b_agent * n_paths + b_path

                path_collisions[a][b] = 1
                path_collisions[b][a] = 1

        return path_collisions

    def _get_idx(self, agent_id: int, path_id: int) -> int:
        return agent_id * self.n_paths + path_id

    def _get_agent_path(self, idx: int) -> tuple[int, int]:
        return idx // self.n_paths, idx % self.n_paths

    def insert_path(self, agent_id: int, path_id: int) -> None:
        self.current_paths.add((agent_id, path_id))
        self.current_idx[self._get_idx(agent_id, path_id)] = 1

    def remove_path(self, agent_id: int, path_id: int) -> None:
        self.current_paths.remove((agent_id, path_id))
        self.current_idx[self._get_idx(agent_id, path_id)] = 0

    def is_path_available(self, agent_id: int, path_id: int) -> bool:
        if (agent_id, path_id) in self.current_paths:
            return False

        other_idx = self._get_idx(agent_id, path_id)
        for current_agent_id, current_path_id in self.current_paths:
            current_idx = self._get_idx(current_agent_id, current_path_id)
            if self.path_collisions[current_idx, other_idx] == 1:
                return False

        return True

    # @benchmark
    # count_collisions_points_along_path executed in 0.000024 seconds
    def count_collisions_points_along_path(self, agent_id: int, path_id: int) -> int:
        idx = self._get_idx(agent_id, path_id)

        result = (self.path_collisions[idx] * self.current_idx).sum()

        return result

    def count_collisions_points_along_existing_path(
        self, agent_id: int, path_id: int
    ) -> int:
        return self.count_collisions_points_along_path(agent_id, path_id)

    def num_collisions(self):
        return self.get_collisions_matrix().sum() // 2

    # @benchmark
    # get_collisions_matrix executed in 0.000460 seconds
    def get_collisions_matrix(self, additional_paths: np.array = None) -> np.array:
        # NOTE: we don't care about agent order, since idx are strictly increasing
        # the returned matrix will be sorted by agent_id
        current_idx = self.current_idx
        if additional_paths is not None:
            current_idx += additional_paths

        current_idx = np.where(current_idx > 0)[0]

        # get collisions between current idx only
        cmatrix = self.path_collisions[np.ix_(current_idx, current_idx)]

        # TODO: remove and start agent_id at 0
        size = self.n_agents
        wrapped_matrix = np.zeros((size, size), dtype=np.int8)

        # Place the reordered matrix in the bottom-right corner
        wrapped_matrix[1:, 1:] = cmatrix.toarray()

        return wrapped_matrix

    def get_agent_collisions_for_path(
        self, agent_id: int, path_id: int
    ) -> tuple[list[tuple[Vertex, set[int]]], list[tuple[Edge, set[int]]]]:
        raise NotImplementedError()

    def get_agent_collisions_for_paths(self, agent: Agent.Agent) -> npt.NDArray:
        start = time.time()
        n_paths = agent.n_paths
        matrix = np.zeros((n_paths, self.n_agents + 1))
        for path_index, path in enumerate(agent.paths):
            idx = self._get_idx(agent_id=agent.id, path_id=path_index)

            for current_agent_id, current_path_id in self.current_paths:
                current_idx = self._get_idx(current_agent_id, current_path_id)
                if current_idx == idx:
                    continue

                row = self.path_collisions[current_idx]
                one_hot = np.zeros(row.shape, dtype=np.int8)
                one_hot[0][idx] = 1

                if (row @ one_hot.T).squeeze().item() == 1:
                    matrix[path_index][current_agent_id] = 1

        result = matrix.sum(axis=1).astype(int).tolist()

        print(f"get_agent_collisions_for_paths Time taken: {time.time() - start}")
        return result

    def calculate_makespan(self):
        self.makespan = (self.path_makespan * self.current_idx).max()
