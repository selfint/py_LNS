import PathTable
import instance
import PathTable
import numpy as np
from LowLevelSolvers import PPNeighborhoodRepair, ExhaustiveNeighborhoodRepair, RankPPNeighborhoodRepair, NeighborhoodRepair
import DestroyMethods
import numpy.typing as npt
import copy
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import typing

def random_initial_solution(instance: instance.instance, path_table: PathTable.PathTable):
    random_path_selection = np.random.randint(instance.n_paths, size=instance.num_agents)
    for agent_id, path_idx in zip(instance.agents.keys(), random_path_selection):
        path_table.insert_path(agent_id, instance.agents[agent_id].paths[path_idx])
        instance.agents[agent_id].path_id = int(path_idx)
    path_table.calculate_makespan()

def generate_random_random_solution_iterative(instance: instance.instance, path_table: PathTable.PathTable, iterations = 200):
    random_initial_solution(instance, path_table) # Choose random paths as initial solution
    solver = IterativeRandomLNS(instance, path_table, 3, num_iterations = iterations)
    solver.run()
class RandomPP:
    def __init__(self, instance: instance.instance, path_table:PathTable.PathTable):
        self.instance = instance
        self.path_table = path_table
        self.n_solved = 0
        self.verbose = self.instance.verbose
    def solve_for_agent(self, agent_id, agent):
        for i in range(len(agent.paths)):
            if self.path_table.is_path_available(agent.paths[i]):
                self.path_table.insert_path(agent_id, agent.paths[i])
                agent.path_id = i
                self.n_solved += 1
                return True
        return False
    def solve(self):
        if self.verbose:
            print(f'\n**** Using Random PP Solver on {self.instance.instance_name}: ****')

        for agent_id, agent in self.instance.agents.items():
            self.solve_for_agent(agent_id, agent)

        if self.verbose:
            print(f'\n**** Successfully solved for {self.n_solved} agents: ****')
            print(f'\n     Failed for {self.instance.num_agents - self.n_solved} agents\n')


class IterativeRandomLNS:
    def __init__(self, instance: instance.instance, path_table:PathTable.PathTable, subset_size, num_iterations = 1000, destroy_method_name = 'priority', low_level_solver_name = 'pp'):
        self.instance = instance
        self.path_table = path_table
        self.subset_size = subset_size
        self.verbose = self.instance.verbose
        self.num_iterations = num_iterations
        self.num_collisions = self.path_table.num_collisions(self.instance.num_agents)
        self.collision_statistics = [self.num_collisions]
        destroy_methods = {'random': DestroyMethods.RandomDestroyHeuristic,
                           'w-random': DestroyMethods.RandomWeightedDestroyHeuristic,
                           'priority': DestroyMethods.PriorityDestroyHeuristic,
                           'cc': DestroyMethods.ConnectedComponentDestroyHeuristic}
        dm = destroy_methods[destroy_method_name]
        self.destroy_heuristic = dm(instance, path_table, subset_size)

        solvers_list = {'pp': PPNeighborhoodRepair,
                        'rank-pp': RankPPNeighborhoodRepair,
                        'exhaustive': ExhaustiveNeighborhoodRepair}
        self.low_level_solver = solvers_list[low_level_solver_name]



    def run_iteration(self):
        subset = self.destroy_heuristic.generate_subset()#np.random.choice(range(1, self.instance.num_agents+1), self.subset_size, replace=False)
        if self.verbose:
            print(subset)
        subset_path_ids = [int(self.instance.agents[agent_id].path_id) for agent_id in subset]
        if self.verbose:
            print(subset_path_ids)
            print(f'\n**** Initial number of collisions: {self.num_collisions} ****')
        low_level_solver = self.low_level_solver(agent_cost_type = 'mean', instance = self.instance,path_table =  self.path_table, agent_subset = subset, verbose=False)
        low_level_solver.run()
        new_num_collisions = self.path_table.num_collisions(self.instance.num_agents)
        if new_num_collisions < self.num_collisions:
            self.num_collisions = new_num_collisions
            if self.verbose:
                print(f'**** Iteration successful! ****')
                print(f'        New path ids: {[int(self.instance.agents[agent_id].path_id) for agent_id in subset]} ****')
                print(f'        New collision count: {self.num_collisions} ****\n')
        else:
            low_level_solver.destroy_neighborhood()
            for agent_id, path_id in zip(subset, subset_path_ids):
                agent = self.instance.agents[agent_id]
                agent.path_id = path_id
                self.path_table.insert_path(agent_id, agent.paths[path_id])
            if self.verbose:
                print(f'**** Iteration failed! \n****')
        self.collision_statistics += [self.num_collisions]


    def run(self):
        for iteration in range(self.num_iterations):
            self.run_iteration()


class ParallelIterativeRandomLNS:
    """
    Evaluates multiple neighborhoods in parallel.

    Uses the destroy method to get a neighborhood of size `subset_size * parallelism`,
    then evaluates each `subset_size` chunk in parallel.

    Takes the result with the largest collision reduction, if any exist.
    Discards all other results.
    """

    def __init__(
        self,
        instance: instance.instance,
        path_table: PathTable.PathTable,
        subset_size: int,
        parallelism: int,
        num_iterations=1000,
        destroy_method_name="priority",
        low_level_solver_name="pp",
        executor_name: typing.Literal["thread"] | typing.Literal["process"] = "process",
    ):
        self.instance = instance
        self.path_table = path_table
        self.subset_size = subset_size
        self.verbose = self.instance.verbose
        self.num_iterations = num_iterations
        self.num_collisions = self.path_table.num_collisions(self.instance.num_agents)
        self.collision_statistics = [self.num_collisions]
        destroy_methods = {'random': DestroyMethods.RandomDestroyHeuristic,
                           'w-random': DestroyMethods.RandomWeightedDestroyHeuristic,
                           'priority': DestroyMethods.PriorityDestroyHeuristic,
                           'cc': DestroyMethods.ConnectedComponentDestroyHeuristic}
        dm = destroy_methods[destroy_method_name]
        self.destroy_heuristic = dm(instance, path_table, subset_size * parallelism)

        solvers_list = {'pp': PPNeighborhoodRepair,
                        'rank-pp': RankPPNeighborhoodRepair,
                        'exhaustive': ExhaustiveNeighborhoodRepair}
        self.low_level_solver = solvers_list[low_level_solver_name]

        self.parallelism = parallelism
        self.executor_name: typing.Literal["thread"] | typing.Literal["process"] = (
            executor_name
        )
        if executor_name == "thread":
            self.executor = ThreadPoolExecutor(max_workers=self.parallelism)
        else:
            self.executor = ProcessPoolExecutor(max_workers=self.parallelism)

    @staticmethod
    def evaluate_subset(
        instance: instance.instance,
        path_table: PathTable.PathTable,
        low_level_solver: NeighborhoodRepair,
        subset: npt.NDArray,
    ) -> tuple[int, PathTable.PathTable]:
        """
        Evaluate a subset of agents.

        Note:
            The value agent.path_id after this method is **will be the new
            agent path**.

        Args:
            instance (instance.instance): the instance
            path_table (PathTable.PathTable): the path table (will be modified in-place)
            low_level_solver (NeighborhoodRepair): the low level solver
            subset (npt.NDArray): the subset of agent ids to evaluate
        
        Returns:
            tuple[int, PathTable.PathTable]: the number of collisions and the path table
        """

        low_level_solver = low_level_solver(
            agent_cost_type="mean",
            instance=instance,
            path_table=path_table,
            agent_subset=subset,
            verbose=False,
        )
        low_level_solver.run()
        new_num_collisions = path_table.num_collisions(instance.num_agents)

        return new_num_collisions, path_table, instance

    def __del__(self):
        self.executor.shutdown(wait=True, cancel_futures=True)

    def run_iteration(self):
        subsets_flat = self.destroy_heuristic.generate_subset()
        subsets_path_ids = [int(self.instance.agents[agent_id].path_id) for agent_id in subsets_flat]

        # evaluate each subset in parallel
        subsets = [
            subsets_flat[i : i + self.subset_size]
            for i in range(0, len(subsets_flat), self.subset_size)
        ]

        if self.verbose:
            print(subsets)
            print(subsets_path_ids)
            print(f'\n**** Initial number of collisions: {self.num_collisions} ****')

        futures = [
            self.executor.submit(
                self.evaluate_subset,
                # if we are using a ProcessPoolExecutor
                # arguments will be auto-copied, otherwise we use copy.deepcopy
                self.instance if self.executor_name == "process" else copy.deepcopy(self.instance),
                self.path_table if self.executor_name == "process" else copy.deepcopy(self.path_table),
                self.low_level_solver,
                subset,
            )
            for subset in subsets
        ]

        results = [future.result() for future in futures]

        new_num_collisions, new_path_table, new_instance = min(results, key=lambda x: x[0])
        if new_num_collisions < self.num_collisions:
            self.num_collisions = new_num_collisions
            self.path_table = new_path_table
            self.instance = new_instance

            if self.verbose:
                print(f'**** Iteration successful! ****')
                print(f'        New path ids: {[int(self.instance.agents[agent_id].path_id) for agent_id in subsets_flat]} ****')
                print(f'        New collision count: {self.num_collisions} ****\n')
        else:
            if self.verbose:
                print(f'**** Iteration failed! \n****')

        self.collision_statistics += [self.num_collisions]

    def run(self):
        for iteration in range(self.num_iterations):
            self.run_iteration()
