from typing import Protocol
import itertools
import numpy as np
import numpy.typing as npt
import scipy.sparse
from Agent import Agent
from PathTable import iter_path
from collections import defaultdict
from benchmark_utils import benchmark


CMatrix = scipy.sparse.csr_matrix
"""
Sparse matrix of size (n_agents*n_paths, n_agents*n_paths) for all
(agent.id * path.id) collisions with each other.
"""

Solution = npt.NDArray[np.int8]
"""
One-hot of matrix of size (n_agents, n_paths) for currently active paths for
each agent. Each cell of size (n_paths) contains a single 1, or is all zero
for intermediate solutions.
"""

Neighborhood = npt.NDArray[np.int8]
"""Vector of size (subset_size) specifying agent ids in a neighborhood"""


class DestroyMethod(Protocol):
    def __call__(
        self, cmatrix: CMatrix, solution: Solution, n_paths: int, size: int
    ) -> Neighborhood:
        """
        Generate a mask of size (agents) specifying which agents to destroy.

        The total 1s in the mask must sum to the given size.
        """

        raise NotImplementedError()


class RepairMethod(Protocol):
    def __call__(
        self,
        cmatrix: CMatrix,
        n_agents: int,
        n_paths: int,
        solution: Solution,
        neighborhood: Neighborhood,
    ) -> Solution:
        """
        Repair a neighborhood in a solution.
        """

        raise NotImplementedError()


def build_cmatrix(agents: list[Agent]) -> CMatrix:
    """
    Build a collision matrix for all agent paths.
    """

    n_agents = len(agents)
    n_paths = agents[0].n_paths

    vertices = defaultdict(set)
    edges = defaultdict(set)
    for agent in agents:
        agent_id = agent.id - 1
        for path_id, locations in enumerate(agent.paths):
            for vertex, edge in iter_path(locations):
                vertices[vertex].add((agent_id, path_id))

                if edge is not None:
                    edges[edge].add((agent_id, path_id))
                    edges[edge.reverse()].add((agent_id, path_id))

    size = n_agents * n_paths
    path_collisions = np.zeros((size, size), dtype=np.int8)
    for group in itertools.chain(vertices.values(), edges.values()):
        for (a_agent, a_path), (b_agent, b_path) in itertools.combinations(group, 2):
            if a_agent == b_agent:
                continue

            a = a_agent * n_paths + a_path
            b = b_agent * n_paths + b_path

            path_collisions[a][b] = 1
            path_collisions[b][a] = 1

    path_collisions = CMatrix(path_collisions)

    return path_collisions


@benchmark(n=10_000)
def solution_cmatrix(cmatrix: CMatrix, solution: Solution) -> CMatrix:
    """
    Generate a solution collision matrix from a slice of the cmatrix.
    """
    solution_idx = np.where(solution.flat > 0)[0]

    return cmatrix[np.ix_(solution_idx, solution_idx)]


def argmax_destroy_method(
    cmatrix: CMatrix, solution: Solution, n_paths: int, size: int
) -> Neighborhood:
    sol_cmatrix = solution_cmatrix(cmatrix, solution)

    cols = sol_cmatrix.sum(axis=0)
    argmax_indices = np.argpartition(cols, -size)[-size:]
    argmax_ids = (argmax_indices + 1).tolist()

    return argmax_ids


def priority_destroy_method(
    cmatrix: CMatrix, solution: Solution, n_paths: int, size: int
) -> Neighborhood:
    n_agents = len(solution)

    random_size = size
    random_ids = [agent_id for agent_id in range(n_agents)]
    subset = np.random.choice(random_ids, random_size, replace=False)

    return subset


def pp_repair_method(
    cmatrix: CMatrix,
    n_agents: int,
    n_paths: int,
    solution: Solution,
    neighborhood: Neighborhood,
) -> Solution:
    """
    Repair neighborhood with prioritized planning.
    """

    def reroute_agent(current_solution: Solution, agent_id: int) -> int:
        paths = np.arange(n_paths) + agent_id * n_paths
        # cast to int16 to avoid overflow
        cols = cmatrix[paths].astype(np.int16) * current_solution.flat

        return np.argmin(cols)

    current_solution = solution.copy()

    # remove paths
    current_solution[neighborhood, :] = 0

    for agent_id in neighborhood:
        new_path_id = reroute_agent(current_solution, agent_id)

        current_solution[agent_id][new_path_id] = 1

    return current_solution


def run_iteration(
    cmatrix: CMatrix,
    n_agents: int,
    n_paths: int,
    solution: Solution,
    collisions: int,
    destroy_method: DestroyMethod,
    repair_method: RepairMethod,
    neighborhood_size: int,
) -> tuple[Solution, int]:
    """
    Runs LNS on the agent paths.

    Args:
        agents (list[Agent]): the agents

    Returns:
        Solution: the solution
    """

    neighborhood = destroy_method(cmatrix, solution, n_paths, neighborhood_size)
    new_solution = repair_method(cmatrix, n_agents, n_paths, solution, neighborhood)
    
    new_collisions = solution_cmatrix(cmatrix, new_solution).sum() // 2

    if new_collisions < collisions:
        return new_solution, new_collisions.item()
    else:
        return solution, collisions
