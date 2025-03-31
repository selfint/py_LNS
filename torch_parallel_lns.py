import itertools
import time
from collections import defaultdict, deque
from typing import NamedTuple, Protocol

import numpy as np
import torch
import torch.multiprocessing as mp
from tqdm import tqdm
from multiprocessing import shared_memory
from concurrent.futures import ProcessPoolExecutor
from queue import Queue

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


def random_initial_solution(n_agents: int, n_paths: int) -> Solution:
    sol = torch.zeros((n_agents, n_paths), dtype=torch.int8)
    paths = torch.randint(0, n_paths, (n_agents,))
    for agent_id, path in enumerate(paths):
        sol[agent_id][path] = 1

    return sol


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


def build_cost_matrix(agents: list[Agent], device="cpu") -> CMatrix:
    """
    Build a collision matrix for all agent paths.
    """

    cost_matrix = [[len(paths) for paths in agent.paths] for agent in agents]

    cost_matrix = torch.tensor(cost_matrix, device=device)

    return cost_matrix


def build_cost_matrix_from_paths(
    paths: list[list[tuple[int, int]]], device="cpu"
) -> CMatrix:
    """
    Build a collision matrix for all agent paths.
    """

    cost_matrix = [[len(path) for path in agent_paths] for agent_paths in paths]

    cost_matrix = torch.tensor(cost_matrix, device=device, dtype=torch.int32)

    return cost_matrix


def build_cmatrix(agents: list[Agent], device="cpu") -> CMatrix:
    """
    Build a collision matrix for all agent paths.
    """

    paths = [a.paths for a in agents]

    n_agents = len(paths)
    n_paths = len(paths[0])

    vertices = defaultdict(set)
    edges = defaultdict(set)
    for agent_id, agent_paths in enumerate(tqdm(paths, desc="Scanning paths")):
        for path_id, locations in enumerate(agent_paths):
            for vertex, edge in iter_path(locations):  # type: ignore
                vertices[vertex].add((agent_id, path_id))

                if edge is not None:
                    edges[edge].add((agent_id, path_id))
                    edges[edge.reverse()].add((agent_id, path_id))

    size = n_agents * n_paths
    path_collisions = np.zeros((size, size), dtype=np.int16)
    for group in tqdm(
        itertools.chain(vertices.values(), edges.values()),
        total=len(vertices) + len(edges),
        desc="Building cmatrix",
    ):
        for (a_agent, a_path), (b_agent, b_path) in itertools.combinations(group, 2):
            if a_agent == b_agent:
                continue

            a = a_agent * n_paths + a_path
            b = b_agent * n_paths + b_path

            path_collisions[a][b] = 1
            path_collisions[b][a] = 1

    path_collisions = torch.tensor(path_collisions, device=device)

    return path_collisions


def build_cmatrix_fast(paths: list[list[tuple[int, int]]], device="cpu") -> CMatrix:
    """
    Build a collision matrix for all agent paths.
    """

    n_agents = len(paths)
    n_paths = len(paths[0])

    vertices = defaultdict(set)
    edges = defaultdict(set)
    for agent_id, agent_paths in enumerate(tqdm(paths, desc="Scanning paths")):
        for path_id, locations in enumerate(agent_paths):
            for vertex, edge in iter_path(locations):  # type: ignore
                vertices[vertex].add((agent_id, path_id))

                if edge is not None:
                    edges[edge].add((agent_id, path_id))
                    edges[edge.reverse()].add((agent_id, path_id))

    size = n_agents * n_paths

    # numpy is faster than torch for these kinds of operations
    path_collisions = np.zeros((size, size), dtype=np.int8)

    # print matrix mem usage in megabytes
    print(f"Matrix size: {size*size*np.int8().nbytes/1024/1024:.2f} MB")

    groups = [
        g for g in itertools.chain(vertices.values(), edges.values()) if len(g) > 1
    ]
    groups = [np.array([list(g) for g in group]) for group in groups]

    for group in tqdm(groups, desc="Building cmatrix"):
        agents = group[:, 0]
        paths = group[:, 1]  # type: ignore
        ids = agents * n_paths + paths
        path_collisions[ids[:, None], ids] = 1

    path_collisions = torch.tensor(path_collisions, dtype=torch.bool, device=device)

    # remove agent collisions with themselves
    view = path_collisions.view((n_agents, n_paths, n_agents, n_paths))
    for agent in range(n_agents):
        view[agent, :, agent, :] = 0

    return path_collisions


def solution_cmatrix(cmatrix: CMatrix, solution: Solution) -> CMatrix:
    """
    Generate a solution collision matrix from a slice of the cmatrix.
    """
    solution_idx = torch.nonzero(solution.ravel(), as_tuple=True)[0]
    return cmatrix[solution_idx][:, solution_idx]


class RandomDestroyMethod:
    def __init__(self, n_subset: int) -> None:
        self.n_subset = n_subset

    def __call__(
        self,
        cmatrix: CMatrix,
        solution: Solution,
        n_agents: int,
        n_paths: int,
        _n_subset: int,
    ) -> Neighborhood:
        return torch.randperm(n_agents, device=cmatrix.device)[: self.n_subset]


def random_destroy_method(
    cmatrix: CMatrix,
    solution: Solution,
    n_agents: int,
    n_paths: int,
    n_subset: int,
) -> Neighborhood:
    return torch.randperm(n_agents, device=cmatrix.device)[:n_subset]


def weighted_random_destroy_method(
    cmatrix: CMatrix,
    solution: Solution,
    n_agents: int,
    n_paths: int,
    n_subset: int,
) -> Neighborhood:
    sol_cmatrix = solution_cmatrix(cmatrix, solution)
    ranks = sol_cmatrix.sum(dim=1)
    if ranks.sum() == 0:
        ranks += 1
    probabilities = ranks / ranks.sum()

    return torch.multinomial(probabilities, n_subset, replacement=False)


def argmax_destroy_method(
    cmatrix: CMatrix, solution: Solution, n_agents: int, n_paths: int, n_subset: int
) -> Neighborhood:
    """Return agents with the most collisions"""
    sol_cmatrix = solution_cmatrix(cmatrix, solution)

    return torch.topk(sol_cmatrix.sum(dim=1), n_subset)[1]


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
    simulated_annealing: tuple[float, float, float] or None = None
    dynamic_neighborhood: int or None = None
    worker_sub_iterations: int = 1


def run_iteration(
    cmatrix: CMatrix,
    solution: Solution,
    collisions: int,
    config: Configuration,
) -> tuple[Solution, int]:
    """
    Runs iteration.
    """

    n_subset = config.neighborhood_size
    if config.dynamic_neighborhood is not None:
        n_subset = int(
            torch.randint(config.dynamic_neighborhood, n_subset + 1, (1,)).item()
        )

    neighborhood = config.destroy_method[0](
        cmatrix, solution, config.n_agents, config.n_paths, n_subset
    )
    new_solution = config.repair_method[0](
        cmatrix, config.n_agents, config.n_paths, solution, neighborhood
    )
    new_collisions = solution_cmatrix(cmatrix, new_solution).sum() // 2

    if new_collisions < collisions:
        return new_solution, new_collisions.item()  # type: ignore
    else:
        return solution, collisions


def run_iterative(
    cmatrix: CMatrix,
    config: Configuration,
    iterations: int,
    optimal: int = 0,
    initial_solution: Solution | None = None,
):
    if initial_solution is not None:
        solution = initial_solution.clone()
    else:
        solution = random_initial_solution(config.n_agents, config.n_paths)

    collisions = int(solution_cmatrix(cmatrix, solution).sum() // 2)

    pbar = tqdm(range(iterations), desc=f"Collisions: {collisions}")
    for _ in pbar:
        solution, collisions = run_iteration(cmatrix, solution, int(collisions), config)
        pbar.set_description(f"Collisions: {collisions}")
        if collisions == optimal:
            return solution, collisions

    return solution, collisions


def run_stopwatch(
    cmatrix: CMatrix,
    config: Configuration,
    seconds: int,
    optimal: int = 0,
    initial_solution: Solution | None = None,
):
    if initial_solution is not None:
        solution = initial_solution.clone()
    else:
        solution = random_initial_solution(config.n_agents, config.n_paths)

    collisions = int(solution_cmatrix(cmatrix, solution).sum() // 2)

    pbar = tqdm(total=seconds, desc=f"Collisions: {collisions}")
    start = time.time()
    while (elapsed := time.time() - start) < seconds:
        solution, collisions = run_iteration(cmatrix, solution, int(collisions), config)
        pbar.set_description(f"Collisions: {collisions}")
        pbar.n = elapsed

        if collisions == optimal:
            return solution, collisions

    return solution, collisions


class SharedState(NamedTuple):
    cmatrix: CMatrix
    solution: Solution
    collisions: torch.Tensor
    iterations: torch.Tensor
    best_solution: Solution
    best_collisions: torch.Tensor
    lock: mp.Lock  # type: ignore


def worker(
    shared: SharedState,
    config: Configuration,
    thread_id: int,
):
    torch.manual_seed(thread_id)

    with shared.lock:
        solution = shared.solution.clone()
        collisions = shared.collisions.clone()

    # bench = Benchmark(n=10_000)

    destroy_method = config.destroy_method[thread_id % len(config.destroy_method)]
    repair_method = config.repair_method[thread_id % len(config.repair_method)]

    iteration = 0
    while True:
        # with bench.benchmark(label="Worker iteration"):
        n_subset = config.neighborhood_size
        if config.dynamic_neighborhood is not None:
            n_subset = int(
                torch.randint(config.dynamic_neighborhood, n_subset + 1, (1,)).item()
            )

        neighborhood = destroy_method(
            shared.cmatrix, solution, config.n_agents, config.n_paths, n_subset
        )
        solution = repair_method(
            shared.cmatrix,
            config.n_agents,
            config.n_paths,
            solution,
            neighborhood,
        )
        sol_cmatrix = solution_cmatrix(shared.cmatrix, solution)
        collisions = sol_cmatrix.sum() // 2

        iteration += 1
        if iteration < config.worker_sub_iterations:
            continue
        else:
            iteration = 0

        with shared.lock:
            shared.iterations.add_(config.worker_sub_iterations)

            # Exponential decay on simulated annealing probability
            if config.simulated_annealing is not None:
                A, k, s = config.simulated_annealing
                decay_factor = A * torch.exp(k * -shared.iterations / s)
                simulated_annealing = (
                    config.simulated_annealing and torch.rand(1).item() < decay_factor
                )
            else:
                simulated_annealing = False

            # record best all-time solution
            if collisions < shared.best_collisions:
                shared.best_collisions.copy_(collisions)
                shared.best_solution[:] = solution

            # update current solution, optionally with simulated annealing
            if collisions < shared.collisions or simulated_annealing:
                shared.solution[:] = solution
                shared.collisions.copy_(collisions)
            else:
                solution = shared.solution.clone()
                collisions = shared.collisions.clone()


def run_parallel(
    cmatrix: CMatrix,
    solution: Solution,
    collisions: int,
    config: Configuration,
    n_threads: int,
    n_seconds: int,
    optimal: int or None = None,
) -> tuple[Solution, int, list[float], list[int]]:

    shared_cmatrix = cmatrix.share_memory_()
    shared_solution = solution.clone().share_memory_()
    shared_collisions = torch.tensor(
        collisions, dtype=torch.int32, device=cmatrix.device
    ).share_memory_()
    shared_iterations = torch.tensor(0).share_memory_()
    shared_best_solution = solution.clone().share_memory_()
    shared_best_collisions = torch.tensor(
        collisions, dtype=torch.int32, device=cmatrix.device
    ).share_memory_()

    lock = mp.Lock()

    shared = SharedState(
        shared_cmatrix,
        shared_solution,
        shared_collisions,
        shared_iterations,
        shared_best_solution,
        shared_best_collisions,
        lock,
    )

    if not mp.get_start_method(allow_none=True):
        mp.set_start_method("spawn")

    workers = []
    for thread_id in range(n_threads):
        p = mp.Process(target=worker, args=(shared, config, thread_id))
        workers.append(p)
        p.start()

    start = time.time()
    log_values = []
    sample_rate = 10 / 1_000  # 10 ms
    with tqdm(total=n_seconds) as pbar:
        while (time_passed := time.time() - start) < n_seconds:
            with lock:
                iterations = shared_iterations.item()
                cols = int(shared_collisions.item())
                best_cols = int(shared_best_collisions.item())

            log_values.append(((time_passed * 1_000, best_cols, iterations)))
            pbar.set_description(
                f"{config.n_agents=: 2} {n_threads=} {iterations=} {cols=} {best_cols=}"
            )

            if optimal is not None and best_cols <= optimal:
                break

            time.sleep(sample_rate)
            pbar.update(round(time_passed - pbar.n))

    for p in workers:
        p.kill()
        p.join()

    timestamps_ms = [t for t, _, _ in log_values]
    log_collisions = [c for _, c, _ in log_values]
    log_iterations = [i for _, _, i in log_values]

    sol = shared_best_solution.argmax(dim=1)

    return (
        shared_best_solution,
        int(shared_best_collisions.item()),
        timestamps_ms,
        log_collisions,
        log_iterations,
    )
