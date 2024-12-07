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
import torch.multiprocessing as mp
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


def build_cmatrix(agents: list[Agent], device="cpu") -> CMatrix:
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

    path_collisions = torch.tensor(path_collisions, device=device)

    return path_collisions


def solution_cmatrix(cmatrix: CMatrix, solution: Solution) -> CMatrix:
    """
    Generate a solution collision matrix from a slice of the cmatrix.
    """
    solution_idx = torch.nonzero(solution.ravel(), as_tuple=True)[0]

    return cmatrix[solution_idx][:,solution_idx]


def priority_destroy_method(
    cmatrix: CMatrix, solution: Solution, n_paths: int, size: int
) -> Neighborhood:
    n_agents = len(solution)

    random_size = size
    random_ids = [agent_id for agent_id in range(n_agents)]
    subset = np.random.choice(random_ids, random_size, replace=False)

    return torch.tensor(subset, device=cmatrix.device, dtype=torch.int32)


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
    current_solution_flat = current_solution.ravel()

    # for each agent, generate a [0, 1, ..., n_paths] array
    all_paths = torch.arange(n_paths, device=cmatrix.device) + neighborhood[:, None] * n_paths

    for agent_id, paths in zip(neighborhood, all_paths):
        current_idx = torch.nonzero(current_solution_flat, as_tuple=True)[0]
        cols = cmatrix[paths][:,current_idx].sum(axis=1)
        new_path_id = torch.argmin(cols)

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

    # print(
    #     "@",
    #     neighborhood + 1,
    #     torch.nonzero(new_solution.ravel(), as_tuple=True)[0][neighborhood],
    #     new_collisions,
    #     collisions,
    # )

    if new_collisions < collisions:
        return new_solution, new_collisions.item()
    else:
        return solution, collisions

def worker(shared_cmatrix: CMatrix, shared_solution: Solution, shared_collisions: torch.Tensor, lock, args):
    n_agents, n_paths, destroy_method, repair_method, neighborhood_size, n_sub_iterations = args

    for _ in range(n_sub_iterations):
        with lock:
            solution = shared_solution.clone()

        neighborhood = destroy_method(shared_cmatrix, solution, n_paths, neighborhood_size)
        new_solution = repair_method(shared_cmatrix, n_agents, n_paths, solution, neighborhood)
        new_collisions = solution_cmatrix(shared_cmatrix, new_solution).sum() // 2

        with lock:
            if new_collisions < shared_collisions:
                shared_solution[:] = new_solution
                shared_collisions.copy_(new_collisions)

def run_parallel(
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

    shared_cmatrix = cmatrix.share_memory_()
    shared_solution = solution.share_memory_()
    shared_collisions = torch.tensor(collisions, dtype=torch.int32, device=cmatrix.device).share_memory_()

    lock = mp.Lock()

    args = (
        shared_cmatrix,
        shared_solution,
        shared_collisions,
        lock,
        (
            n_agents,
            n_paths,
            destroy_method,
            repair_method,
            neighborhood_size,
            n_sub_iterations,
        ),
    )

    workers = []
    for _ in range(n_threads):
        p = mp.Process(target=worker, args=args)
        workers.append(p)
        p.start()

    for p in workers:
        p.join()

    sol = shared_solution.argmax(dim=1)
    print(f"Final solution: {sol} {shared_collisions=}")

    return shared_solution, shared_collisions


def iteration_worker(shared_cmatrix: CMatrix, n_sub_iterations, args) -> tuple[Solution, int]:
    new_solution = args[2].clone()
    collisions = args[3]

    for sub_iteration in range(n_sub_iterations):
        new_solution, new_collisions = run_iteration(shared_cmatrix, *args)

        if new_collisions < collisions:
            return new_solution, new_collisions
        
    return new_solution, new_collisions

def run_parallel_iteration(
    shared_cmatrix: CMatrix,
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

    args = [
        n_agents,
        n_paths,
        solution,
        collisions,
        destroy_method,
        repair_method,
        neighborhood_size
    ]

    with concurrent.futures.ProcessPoolExecutor(max_workers=n_threads) as ex:
        futures = [
            ex.submit(iteration_worker, shared_cmatrix, n_sub_iterations, args)
            for _ in range(n_threads)
        ]

        for future in concurrent.futures.as_completed(futures):
            f_solution, f_collisions = future.result()
            if f_collisions < collisions:
                return f_solution, f_collisions

    return solution, collisions
