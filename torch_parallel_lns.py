import itertools
import time
from collections import defaultdict
from typing import NamedTuple, Protocol

import numpy as np
import torch
import torch.multiprocessing as mp
from tqdm import tqdm

from Agent import Agent
from benchmark_utils import benchmark, Benchmark
from PathTable import iter_path

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
        self,
        cmatrix: CMatrix,
        solution: Solution,
        n_agents: int,
        n_paths: int,
        n_subset: int,
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
            for vertex, edge in iter_path(locations):  # type: ignore
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

    return cmatrix[solution_idx][:, solution_idx]


def random_destroy_method(
    cmatrix: CMatrix, solution: Solution, n_agents: int, n_paths: int, n_subset: int
) -> Neighborhood:
    return torch.randperm(n_agents, device=cmatrix.device)[:n_subset]


# @benchmark(n=10_000)
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
    all_paths = (
        torch.arange(n_paths, device=cmatrix.device) + neighborhood[:, None] * n_paths
    )

    for agent_id, paths in zip(neighborhood, all_paths):
        current_idx = torch.nonzero(current_solution_flat, as_tuple=True)[0]
        cols = cmatrix[paths][:, current_idx].sum(dim=1)
        new_path_id = torch.argmin(cols)

        current_solution[agent_id][new_path_id] = 1

    return current_solution


def exhaustive_repair_method(
    cmatrix: CMatrix,
    n_agents: int,
    n_paths: int,
    solution: Solution,
    neighborhood: Neighborhood,
) -> Solution:
    """
    Repair neighborhood with exhaustive solver.
    """

    current_solution = solution.clone()

    # remove paths
    current_solution[neighborhood, :] = 0

    permutations = torch.cartesian_prod(
        *(torch.arange(n_paths) + agent_id * n_paths for agent_id in neighborhood)
    )

    solutions = (
        current_solution.ravel()
        .repeat(len(permutations))
        .reshape(-1, n_agents * n_paths)
    )

    batch_size = len(solutions)

    solutions[torch.arange(len(permutations)).unsqueeze(1), permutations] = 1

    solutions = solutions.reshape(-1, n_agents, n_paths).argmax(dim=2)
    solutions += (
        (torch.arange(n_agents) * n_paths)
        .repeat(len(solutions))
        .reshape(batch_size, n_agents)
    )

    # Perform the operation to get the desired result without a for loop
    rows = solutions.unsqueeze(2).expand(batch_size, n_agents, n_agents)  # [M, X, X]
    cols = solutions.unsqueeze(1).expand(batch_size, n_agents, n_agents)  # [M, X, X]

    sub_matrices = cmatrix[rows, cols]
    result = sub_matrices.sum(dim=(1, 2))
    best_solution = solutions[result.argmin()]
    current_solution.ravel()[best_solution] = 1

    return current_solution


class Configuration(NamedTuple):
    n_agents: int
    n_paths: int
    destroy_method: list[DestroyMethod]
    repair_method: list[RepairMethod]
    neighborhood_size: int
    simulated_annealing: tuple[float, float, float] | None = None


def run_iteration(
    cmatrix: CMatrix,
    solution: Solution,
    collisions: int,
    c: Configuration,
) -> tuple[Solution, int]:
    """
    Runs iteration.
    """

    neighborhood = c.destroy_method[0](
        cmatrix, solution, c.n_agents, c.n_paths, c.neighborhood_size
    )
    new_solution = c.repair_method[0](
        cmatrix, c.n_agents, c.n_paths, solution, neighborhood
    )
    new_collisions = solution_cmatrix(cmatrix, new_solution).sum() // 2

    if new_collisions < collisions:
        return new_solution, new_collisions.item()  # type: ignore
    else:
        return solution, collisions


def worker(
    shared_cmatrix: CMatrix,
    shared_solution: Solution,
    shared_collisions: torch.Tensor,
    shared_iterations: torch.Tensor,
    lock,
    c: Configuration,
    thread_id: int,
):
    with lock:
        thread_solution = shared_solution.clone()
        thread_collisions = shared_collisions.clone()

    # bench = Benchmark(n=10_000)

    destroy_method = c.destroy_method[thread_id % len(c.destroy_method)]
    repair_method = c.repair_method[thread_id % len(c.repair_method)]

    while True:
        # with bench.benchmark(label="Worker iteration"):
        neighborhood = destroy_method(
            shared_cmatrix, thread_solution, c.n_agents, c.n_paths, c.neighborhood_size
        )
        thread_solution = repair_method(
            shared_cmatrix, c.n_agents, c.n_paths, thread_solution, neighborhood
        )
        thread_collisions = solution_cmatrix(shared_cmatrix, thread_solution).sum() // 2

        with lock:
            shared_iterations += 1

            # Exponential decay on simulated annealing probability
            if c.simulated_annealing is not None:
                A, k, s = c.simulated_annealing
                decay_factor = A * torch.exp(k * -shared_iterations / s)
                simulated_annealing = (
                    c.simulated_annealing and torch.rand(1).item() < decay_factor
                )
            else:
                simulated_annealing = False

            if thread_collisions < shared_collisions or simulated_annealing:
                shared_solution[:] = thread_solution
                shared_collisions.copy_(thread_collisions)
            else:
                thread_solution = shared_solution.clone()
                thread_collisions = shared_collisions.clone()


def run_parallel(
    cmatrix: CMatrix,
    solution: Solution,
    collisions: int,
    c: Configuration,
    n_threads: int,
    n_seconds: int,
    optimal: int = 0,
) -> tuple[Solution, int, list[float], list[int]]:

    shared_cmatrix = cmatrix.share_memory_()
    shared_solution = solution.share_memory_()
    shared_collisions = torch.tensor(
        collisions, dtype=torch.int32, device=cmatrix.device
    ).share_memory_()
    shared_iterations = torch.tensor(0).share_memory_()

    lock = mp.Lock()

    args = (
        shared_cmatrix,
        shared_solution,
        shared_collisions,
        shared_iterations,
        lock,
        c,
    )

    workers = []
    for thread_id in range(n_threads):
        p = mp.Process(target=worker, args=(*args, thread_id))
        workers.append(p)
        p.start()

    # while workers working
    start = time.time()
    log_values = []
    with tqdm(total=n_seconds) as pbar:
        while time.time() - start < n_seconds:
            with lock:
                iterations = shared_iterations.item()
                cols = int(shared_collisions.item())

            log_values.append((((time.time() - start) * 1_000, iterations)))
            pbar.set_description(
                f"P {n_threads: 2} Iterations: {iterations} Cols: {cols}"
            )

            if cols <= optimal:
                break

            time.sleep(0.5)
            pbar.update(0.5)

    for p in workers:
        p.kill()
        p.join()

    timestamps_ms = [t for t, _ in log_values]
    iterations = [c for _, c in log_values]

    sol = shared_solution.argmax(dim=1)

    return shared_solution, int(shared_collisions.item()), timestamps_ms, iterations
