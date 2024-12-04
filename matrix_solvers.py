from MatrixPathTable import MatrixPathTable
from instance import instance
import numpy as np
from MatrixLowLevelSolvers import (
    PPNeighborhoodRepair,
    ExhaustiveNeighborhoodRepair,
    RankPPNeighborhoodRepair,
)
from MatrixDestroyMethods import (
    DestroyHeuristic,
    RandomDestroyHeuristic,
    PriorityDestroyHeuristic,
    RandomWeightedDestroyHeuristic,
    ConnectedComponentDestroyHeuristic,
)
import copy
from Agent import Agent
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import typing
import PathTable


def random_initial_solution(inst: instance, path_table: MatrixPathTable) -> None:
    random_path_selection = np.random.randint(inst.n_paths, size=inst.num_agents)

    for agent_id, path_id in zip(inst.agents.keys(), random_path_selection):
        path_table.insert_path(agent_id, path_id)
        inst.agents[agent_id].path_id = int(path_id)

    path_table.calculate_makespan()


def generate_random_random_solution_iterative(
    n_agents: int, n_paths: int, path_table: MatrixPathTable, iterations=200
):
    raise NotImplementedError("TODO")
    random_initial_solution(n_agents, n_paths, path_table)
    solver = MatrixIterativeRandomLNS(
        instance, path_table, 3, num_iterations=iterations
    )
    solver.run()


class MatrixRandomPP:
    def __init__(
        self, path_table: MatrixPathTable, name="MatrixRandomPP", verbose=False
    ):
        self.path_table = path_table
        self.name = name
        self.verbose = verbose

        self.n_solved = 0

    def solve_for_agent(self, agent_id: int, n_paths: int) -> bool:
        for path_id in range(n_paths):
            if self.path_table.is_path_available(agent_id, path_id):
                self.path_table.insert_path(agent_id, path_id)
                self.n_solved += 1
                return True

        return False

    def solve(self):
        if self.verbose:
            print(f"\n**** Using Random PP Solver on {self.name}: ****")

        for agent_id in range(1, self.path_table.n_agents):
            self.solve_for_agent(agent_id, self.path_table.n_paths)

        if self.verbose:
            print(f"\n**** Successfully solved for {self.n_solved} agents: ****")
            print(
                f"\n     Failed for {self.path_table.n_agents - 1 - self.n_solved} agents\n"
            )


class MatrixIterativeRandomLNS:

    def __init__(
        self,
        instance: instance,
        path_table: MatrixPathTable,
        subset_size,
        num_iterations=1000,
        destroy_method_name="priority",
        low_level_solver_name="pp",
        verbose=False,
    ):
        self.instance = instance
        self.path_table = path_table
        self.subset_size = subset_size
        self.verbose = verbose
        self.num_iterations = num_iterations
        self.num_collisions = self.path_table.num_collisions()
        self.collision_statistics = [self.num_collisions]
        destroy_methods = {
            "random": RandomDestroyHeuristic,
            "w-random": RandomWeightedDestroyHeuristic,
            "priority": PriorityDestroyHeuristic,
            "cc": ConnectedComponentDestroyHeuristic,
        }
        dm = destroy_methods[destroy_method_name]
        self.destroy_heuristic: DestroyHeuristic = dm(instance, path_table, subset_size)

        solvers_list = {
            "pp": PPNeighborhoodRepair,
            "rank-pp": RankPPNeighborhoodRepair,
            "exhaustive": ExhaustiveNeighborhoodRepair,
        }
        self.low_level_solver = solvers_list[low_level_solver_name]

    def run_iteration(self):
        subset = self.destroy_heuristic.generate_subset()

        if self.verbose:
            print(subset)

        subset_path_ids = [
            int(self.instance.agents[agent_id].path_id) for agent_id in subset
        ]

        if self.verbose:
            print(subset_path_ids)
            print(f"\n**** Initial number of collisions: {self.num_collisions} ****")

        low_level_solver = self.low_level_solver(
            agent_cost_type="mean",
            instance=self.instance,
            path_table=self.path_table,
            agent_subset=subset,
            verbose=False,
        )
        low_level_solver.run()

        new_num_collisions = self.path_table.num_collisions()
        if new_num_collisions < self.num_collisions:
            self.num_collisions = new_num_collisions
            if self.verbose:
                print(f"**** Iteration successful! ****")
                print(
                    f"        New path ids: {[int(self.instance.agents[agent_id].path_id) for agent_id in subset]} ****"
                )
                print(f"        New collision count: {self.num_collisions} ****\n")

        else:
            low_level_solver.destroy_neighborhood()
            for agent_id, path_id in zip(subset, subset_path_ids):
                agent = self.instance.agents[agent_id]
                agent.path_id = path_id
                self.path_table.insert_path(agent_id, path_id)
            if self.verbose:
                print(f"**** Iteration failed! \n****")

        self.collision_statistics += [self.num_collisions]

    def run(self, early_stopping=False) -> tuple[typing.Self, int]:
        for iteration in range(self.num_iterations):
            prev_collisions = self.num_collisions
            self.run_iteration()
            if early_stopping and (self.num_collisions < prev_collisions):
                return self, iteration + 1

        return self, self.num_iterations
