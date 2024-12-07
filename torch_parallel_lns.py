from typing import Protocol
import itertools
import numpy as np
import numpy.typing as npt
import scipy.sparse
from Agent import Agent
from PathTable import iter_path
from collections import defaultdict
from benchmark_utils import benchmark
import concurrent.futures
from multiprocessing import shared_memory, Process
import torch


CMatrix = torch.Tensor
"""
Sparse matrix of size (n_agents*n_paths, n_agents*n_paths) for all
(agent.id * path.id) collisions with each other. dtype=int16
"""

Solution = torch.Tensor
"""
One-hot of matrix of size (n_agents, n_paths) for currently active paths for
each agent. Each cell of size (n_paths) contains a single 1, or is all zero
for intermediate solutions.
"""

Neighborhood = torch.Tensor
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
    path_collisions = np.zeros((size, size), dtype=np.int16)
    for group in itertools.chain(vertices.values(), edges.values()):
        for (a_agent, a_path), (b_agent, b_path) in itertools.combinations(group, 2):
            if a_agent == b_agent:
                continue

            a = a_agent * n_paths + a_path
            b = b_agent * n_paths + b_path

            path_collisions[a][b] = 1
            path_collisions[b][a] = 1

    path_collisions = torch.Tensor(path_collisions)

    return path_collisions


def solution_cmatrix(cmatrix: CMatrix, solution: Solution) -> CMatrix:
    """
    Generate a solution collision matrix from a slice of the cmatrix.
    """
    solution_idx = torch.where(solution.ravel() > 0)[0]

    return cmatrix[torch.meshgrid(solution_idx, solution_idx, indexing="ij")]


def priority_destroy_method(
    cmatrix: CMatrix, solution: Solution, n_paths: int, size: int
) -> Neighborhood:
    n_agents = len(solution)

    random_size = size
    random_ids = [agent_id for agent_id in range(n_agents)]
    subset = np.random.choice(random_ids, random_size, replace=False)

    return torch.Tensor(subset).to(torch.int32)


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

    current_solution = solution.clone()

    # remove paths
    current_solution[neighborhood, :] = 0

    # flatten solution for indexing into solution matrix
    current_solution = current_solution.flatten()

    # for each agent, generate a [0, 1, ..., n_paths] array
    all_paths = torch.arange(n_paths) + neighborhood[:, None] * n_paths

    for agent_id, paths in zip(neighborhood, all_paths):
        cols = cmatrix[paths][:,torch.where(current_solution > 0)[0]].sum(axis=1)
        new_path_id = torch.argmin(cols)

        current_solution[agent_id * n_paths + new_path_id] = 1

    current_solution = current_solution.reshape(n_agents, n_paths)

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


def run_parallel_iteration(
    cmatrix: CMatrix,
    n_agents: int,
    n_paths: int,
    solution: Solution,
    collisions: int,
    destroy_method: DestroyMethod,
    repair_method: RepairMethod,
    neighborhood_size: int,
    n_threads: int,
    n_sub_iterations: int,
) -> tuple[Solution, int]:

    def create_shared_memory(array):
        shm = shared_memory.SharedMemory(create=True, size=array.nbytes)
        shared_array = np.ndarray(array.shape, dtype=array.dtype, buffer=shm.buf)
        np.copyto(shared_array, array)
        return shm

    shm_data = create_shared_memory(cmatrix.data)
    shm_indices = create_shared_memory(cmatrix.indices)
    shm_indptr = create_shared_memory(cmatrix.indptr)

    dtype = cmatrix.dtype

    def run():
        thread_cols = collisions
        thread_solution = solution.copy()

        data = np.ndarray((shm_data.size // dtype.itemsize,), dtype=dtype, buffer=shm_data.buf)
        indices = np.ndarray((shm_indices.size // dtype.itemsize,), dtype=dtype, buffer=shm_indices.buf)
        indptr = np.ndarray((shm_indptr.size // dtype.itemsize,), dtype=dtype, buffer=shm_indptr.buf)

        thread_cmatrix = scipy.sparse.csr_matrix(
            (data, indices, indptr), shape=cmatrix.shape, copy=False
        )

        print(thread_cmatrix.toarray())

    with concurrent.futures.ProcessPoolExecutor(max_workers=n_threads) as executor:
        futures = [
            executor.submit(
                run,
            )
            for _ in range(n_threads)
        ]

        for future in concurrent.futures.as_completed(futures):
            pass

    shm_data.close()
    shm_indices.close()
    shm_indptr.close()

