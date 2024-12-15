from typing import TypedDict
import instance
from logger import Logger
import solvers
from PathTable import PathTable
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from plotter import *
from graphMethods import *
import itertools
from visualization import visualize, draw_graph_highlight_paths
from PathGenerator import k_shortest_paths
import tqdm
from pathlib import Path

def run_scenario(map_path, agent_path, solver, log_file = 'experiments.csv', verbose = True, n_paths = 2, temp = 1):
    inst = instance.instance(map_path, agent_path, solver, verbose, n_paths, agent_path_temp = temp)
    logger = Logger(log_file, inst)
    t = PathTable(inst.num_of_rows, inst.num_of_cols)
    solver = solvers.RandomPP(inst, t)
    logger.start()
    solver.solve()
    n_solved = solver.n_solved
    logger.end()
    logger.log(n_solved)

def run_all_scenarios(map_name, agent_directory, solver, verbose = True, n_paths = 2):
    for agent_name in os.listdir(agent_directory):
        agent_path = os.path.join(agent_directory, agent_name)
        print(agent_path)
        run_scenario(map_name, agent_path, solver, verbose = True, n_paths = 2)


def temp_ablation(map_path, agent_path, solver, verbose = True, n_paths = 2, temp = 1):
    open('temp_ablation.csv', "a")
    temps = np.logspace(-1, 0, 100)
    #temps = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.8, 1, 1.1, 1.2 , 1.3 , 1.5]
    for temp in temps:
        run_scenario(map_path, agent_path, solver,'temp_ablation.csv', verbose, n_paths, temp)


def pandas_exp():
    df = pd.read_csv('n_path_ablation.csv')
    #df = df[['percent_solved', 'agent_temp']]
    df.plot(x = 'n_paths', y = 'percent_solved')
    plt.show()
    #print(df.to_string())
    #print(df['percent_solved'].to_string())

def n_path_ablation(map_path, agent_path, solver, verbose = True, temp = 1):
    open('n_path_ablation.csv', "a")
    n_paths = range(1,33)
    #temps = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.8, 1, 1.1, 1.2 , 1.3 , 1.5]
    for n_path in n_paths:
        run_scenario(map_path, agent_path, solver,'n_path_ablation.csv', verbose, n_path, temp)

def group_size_ablation(map_path, agent_path, solver_name, verbose = True, n_paths = 2, temp = 1):
    open('group_size_ablation.csv', "a")
    group_sizes = [10,15,20, 25, 30]#range(1, 12)
    collision_counts = []
    x_axis = []
    for size in group_sizes:
        s = instance.instance(map_path, agent_path, solver_name)
        t = PathTable(s.num_of_rows, s.num_of_cols)
        solvers.random_initial_solution(s, t)
        solver = solvers.IterativeRandomLNS(s, t, size)
        x_axis = range(solver.num_iterations + 1)
        solver.run()
        collision_counts += [solver.collision_statistics]
    label = 'num_of_cols in random PP LNS (varying group sizes, 1000 iteration)'
    x_axis_label = 'iterations'
    y_axis_label = 'collision counts'
    y_labels = [f'subset size = {size}' for size in group_sizes]
    plot_line_graphs(x_axis, collision_counts, label, x_axis_label,y_axis_label, y_labels)

def exhaustive_vs_random_exp(map_path, agent_path, solver_name, verbose = True, n_paths = 3, temp = 1):
    #open('test.csv', "a")
    solvers_list = ['pp', 'exhaustive']#range(1, 12)
    collision_counts = []
    x_axis = []
    for solver in solvers_list:
        s = instance.instance(map_path, agent_path, solver, verbose, n_paths, temp)
        t = PathTable(s.num_of_rows, s.num_of_cols)
        solvers.random_initial_solution(s, t)
        #adj_matrix = t.get_collisions_matrix(s.num_agents)
        #subset = get_largest_connected_component(adj_matrix)
        #print(get_degrees_of_vertices_dict(adj_matrix, subset))
        solver = solvers.IterativeRandomLNS(s, t, 3, destroy_method_name='w-random', num_iterations= 1000, low_level_solver_name=solver)
        x_axis = range(solver.num_iterations + 1)
        solver.run()
        collision_counts += [solver.collision_statistics]
    label = 'num_of_cols in random PP LNS (varying solvers, 1000 iteration)'
    x_axis_label = 'iterations'
    y_axis_label = 'collision counts'
    y_labels = solvers_list
    plot_line_graphs(x_axis, collision_counts, label, x_axis_label,y_axis_label, y_labels)

def destroy_method_ablation_exp(map_path, agent_path, solver_name, verbose = True, n_paths = 8, temp = 1):
    #open('test.csv', "a")
    ds_list = ['cc', 'random', 'w-random']#range(1, 12)
    collision_counts = []
    x_axis = []
    for ds in ds_list:
        s = instance.instance(map_path, agent_path, solver_name, verbose, n_paths, temp)
        t = PathTable(s.num_of_rows, s.num_of_cols, num_of_agents=s.num_agents)
        paths = k_shortest_paths(s.map_graph, (1,26), (25,1), 7)
        [print(path) for path in paths]
        draw_graph_highlight_paths(s.map_graph, paths)
        solvers.random_initial_solution(s, t)
        solver = solvers.IterativeRandomLNS(s, t, 10, destroy_method_name=ds, num_iterations= 500, low_level_solver_name=solver_name)
        x_axis = range(solver.num_iterations + 1)
        solver.run()
        collision_counts += [solver.collision_statistics]
    label = 'num_of_cols in random PP LNS (varying destroy methods, 1000 iteration)'
    x_axis_label = 'iterations'
    y_axis_label = 'collision counts'
    y_labels = ds_list
    plot_line_graphs(x_axis, collision_counts, label, x_axis_label,y_axis_label, y_labels)


def test_exp(map_path, agent_path, solver_name, verbose = True, n_paths = 3, temp = 1):
    #open('test.csv', "a")
    group_sizes = [20]#range(1, 12)
    solvers_list = ['pp','rank-pp']#range(1, 12)
    variants = list(itertools.product(group_sizes, solvers_list))
    collision_counts = []
    x_axis = []
    for size, solver_name in variants:
        s = instance.instance(map_path, agent_path, solver_name, n_paths = n_paths)
        t = PathTable(s.num_of_rows, s.num_of_cols)
        solvers.random_initial_solution(s, t)
        adj_matrix = t.get_collisions_matrix(s.num_agents)
        largest_cc = get_largest_connected_component(adj_matrix)
        random_walk_until_neighborhood_is_full(adj_matrix, largest_cc, subset_size=5)
        #print(t.get_collisions_matrix(s.num_agents).sum(axis=1))
        #t.get_agent_collisions_for_paths(s.agents[2], s.num_agents)
        solver = solvers.IterativeRandomLNS(s, t, size, destroy_method_name='cc',low_level_solver_name = solver_name, num_iterations= 3000)
        x_axis = range(solver.num_iterations + 1)
        solver.run()
        collision_counts += [solver.collision_statistics]
        visualize(s, t, )
    label = 'num_of_cols in random PP LNS (varying group sizes and solvers, 1000 iteration)'
    x_axis_label = 'iterations'
    y_axis_label = 'collision counts'
    y_labels = [f'subset size = {size}, solver = {solver}' for size, solver in variants]
    plot_line_graphs(x_axis, collision_counts, label, x_axis_label,y_axis_label, y_labels)


class OptimisticIterationResult(TypedDict):
    total_colliding_agents: int
    total_improved_agents: int
    total_improvement: int


def optimistic_iteration_exp(
    map_path,
    agent_path,
    n_paths=3,
    temp=1,
    verbose=True,
    n_iterations=1000,
    max_agents=None,
    on_result_callback=None,
) -> list[instance.instance, PathTable, OptimisticIterationResult]:
    inst = instance.instance(
        map_path,
        agent_path,
        n_paths=n_paths,
        instance_name="Optimistic Iteration",
        agent_path_temp=temp,
        verbose=verbose
    )

    if max_agents is not None:
        new_agents = dict([(k, v) for k, v in inst.agents.items()][:max_agents])
        inst.agents = new_agents
        inst.num_agents = len(new_agents)

    table = PathTable(inst.num_of_rows, inst.num_of_cols, inst.num_agents)

    group_size = 20

    solvers.random_initial_solution(inst, table)

    solver = solvers.IterativeRandomLNS(inst, table, group_size, num_iterations=n_iterations)

    results: list[OptimisticIterationResult] = []

    p_bar = tqdm.tqdm(range(n_iterations))
    for iteration in p_bar:
        # get improvement from changing a single agent
        total_colliding_agents = 0
        total_improved_agents = 0
        total_improvement = 0
        for agent_id, agent in inst.agents.items():
            initial_path = agent.path_id

            assert initial_path != -1, "got agent without path"

            # get amount of robots this robot collides with
            collision_matrix = table.get_collisions_matrix(inst.num_agents)
            initial_collisions = collision_matrix[agent_id].sum()

            # ignore robots without collisions
            if initial_collisions == 0:
                continue

            if verbose:
                print(f'\n**** Agent {agent_id} ****')
                print(f'\nInitial number of collisions: {initial_collisions}')

            total_colliding_agents += 1

            # iterate over all other path selections
            table.remove_path(agent_id, agent.paths[initial_path])
            best_collisions = initial_collisions
            for path_id in range(inst.n_paths):
                if path_id == initial_path:
                    continue

                table.insert_path(agent_id, agent.paths[path_id])

                iteration_collision_matrix = table.get_collisions_matrix(
                    inst.num_agents
                )
                new_collisions = iteration_collision_matrix[agent_id].sum()
                if new_collisions < best_collisions:
                    best_collisions = new_collisions
                    if verbose:
                        print(f'\n      Found new path selection: {path_id} ****')
                        print(f'\n      New number of collisions: {best_collisions} ****')

                table.remove_path(agent_id, agent.paths[path_id])

            collision_reduction = initial_collisions - best_collisions
            if collision_reduction > 0:
                total_improved_agents += 1
                total_improvement += collision_reduction

            # restore initial state
            table.insert_path(agent_id, agent.paths[initial_path])

        # print results
        if verbose:
            print(f'\n**** Iteration {iteration} ****')
            print(f'\nTotal colliding agents: {total_colliding_agents}')
            print(f'\nTotal improved agents: {total_improved_agents}')
            print(f'\nTotal improvement: {total_improvement}')

        result: OptimisticIterationResult = {
            "total_colliding_agents": int(total_colliding_agents),
            "total_improved_agents": int(total_improved_agents),
            "total_improvement": int(total_improvement),
        }

        results.append(result)

        on_result_callback(result)

        # improve with solver
        solver.run_iteration()

        p_bar.set_description(f'Collisions: {solver.num_collisions}')

    return inst, table, results


def stateless_solver_test_exp(
    map_path, agent_path, n_paths, test_iterations, temp, verbose
):
    import torch_parallel_lns as parallel_lns
    import time
    import torch

    test_iterations = 1000

    subset_size = 20

    inst = instance.instance(
        map_path,
        agent_path,
        n_paths=n_paths,
        instance_name="Parallel solver experiment",
        agent_path_temp=temp,
        verbose=verbose,
    )

    table = PathTable(
        inst.num_of_rows,
        inst.num_of_cols,
        inst.num_agents,
    )

    np.random.seed(42)

    solvers.random_initial_solution(inst, table)

    n_agents = inst.num_agents
    p_solution = torch.zeros((n_agents, n_paths), dtype=torch.int8)
    for agent in inst.agents.values():
        p_solution[agent.id - 1][agent.path_id] = 1

    p_cmatrix = parallel_lns.build_cmatrix(list(inst.agents.values()))

    for agent, m_agent in zip(
        sorted(inst.agents.values(), key=lambda a: a.id), p_solution
    ):
        assert agent.path_id == np.where(m_agent > 0)[0]

    solver = solvers.IterativeRandomLNS(
        inst,
        table,
        subset_size=subset_size,
        num_iterations=0,
        destroy_method_name="priority",
        low_level_solver_name="pp",
    )

    p_collisions = parallel_lns.solution_cmatrix(p_cmatrix, p_solution)
    p_num_collisions = (p_collisions.sum() // 2).item()

    assert solver.num_collisions == p_num_collisions, (
        "Invalid initial collisions",
        solver.num_collisions,
        p_num_collisions,
    )

    durations = []
    p_durations = []
    for iteration in tqdm.tqdm(range(test_iterations)):
        random_state = np.random.get_state()

        start = time.time()
        solver.run_iteration()
        duration_ms = (time.time() - start) * 1000

        np.random.set_state(random_state)
        start = time.time()
        p_solution, p_num_collisions = parallel_lns.run_iteration(
            p_cmatrix,
            p_solution,
            p_num_collisions,
            parallel_lns.Configuration(
                n_agents,
                n_paths,
                destroy_method=parallel_lns.random_destroy_method,
                repair_method=parallel_lns.pp_repair_method,
                neighborhood_size=subset_size,
            ),
        )
        p_duration_ms = (time.time() - start) * 1000

        durations.append(duration_ms)
        p_durations.append(p_duration_ms)

        cmatrix = solver.path_table.get_collisions_matrix(inst.num_agents)
        p_collisions = parallel_lns.solution_cmatrix(p_cmatrix, p_solution).numpy()

        for agent, m_agent in zip(
            sorted(inst.agents.values(), key=lambda a: a.id), p_solution
        ):
            p_path_id = np.where(m_agent > 0)[0]
            if agent.path_id != p_path_id:
                print(agent.id - 1, agent.path_id, p_path_id)
                assert False

        assert solver.num_collisions == p_num_collisions, (
            "Invalid initial collisions",
            solver.num_collisions,
            p_num_collisions,
        )

        if not np.array_equal(cmatrix[1:, 1:], p_collisions):
            for agent_id, (row, p_row) in enumerate(zip(cmatrix[1:, 1:], p_collisions)):
                if not np.array_equal(row, p_row):
                    row = np.where(row > 0)[0]
                    p_row = np.where(p_row > 0)[0]
                    print(agent_id, row, p_row)

            assert False


def stateless_solver_parallelism_exp(
    map_path,
    agent_path,
    n_paths,
    temp,
    verbose,
    n_seconds,
    n_threads,
    results_dir: Path,
    optimal: int = 0,
):
    """
    NOTE: Uses multiprocessing, must be executed after if __name__ == "__main__"
    """
    import time
    import json

    import torch

    import torch_parallel_lns as parallel_lns

    results_dir.mkdir(parents=True, exist_ok=True)

    inst = instance.instance(
        map_path,
        agent_path,
        n_paths=n_paths,
        instance_name="Optimistic Iteration",
        agent_path_temp=temp,
        verbose=verbose,
    )

    subset_size = 20

    agents = list(sorted(inst.agents.values(), key=lambda a: a.id))

    p_cmatrix = parallel_lns.build_cmatrix(agents)

    table = PathTable(
        inst.num_of_rows,
        inst.num_of_cols,
        inst.num_agents,
    )

    solvers.random_initial_solution(inst, table)
    n_agents = inst.num_agents

    results = []

    # no threads
    start = time.time()
    pbar = tqdm.tqdm(range(n_seconds))
    total = 0
    log_t = 0
    timestamps = []
    iterations = []

    p_solution = torch.zeros((n_agents, n_paths), dtype=torch.int8)
    for agent in inst.agents.values():
        p_solution[agent.id - 1][agent.path_id] = 1

    p_cols = int(parallel_lns.solution_cmatrix(p_cmatrix, p_solution).sum().item() // 2)

    p_solution = torch.zeros((n_agents, n_paths), dtype=torch.int8)
    for agent in inst.agents.values():
        p_solution[agent.id - 1][agent.path_id] = 1

    p_cols = int(parallel_lns.solution_cmatrix(p_cmatrix, p_solution).sum().item() // 2)

    p_solution, p_cols, timestamps, iterations = parallel_lns.run_parallel(
        p_cmatrix,
        p_solution,
        p_cols,
        c=parallel_lns.Configuration(
            n_agents,
            n_paths,
            destroy_method=parallel_lns.random_destroy_method,
            repair_method=parallel_lns.pp_repair_method,
            neighborhood_size=subset_size,
        ),
        n_threads=n_threads,
        n_seconds=n_seconds,
        optimal=optimal,
    )

    iterations = np.array(iterations)
    timestamps = np.array(timestamps)
    last_zero_col_index = np.where(iterations >= n_threads)[0][0]

    rate = np.round(
        np.mean(
            np.diff(iterations[last_zero_col_index:])
            / np.diff(timestamps[last_zero_col_index:])
        )
        * 1000
    )

    results.append((n_threads, timestamps, iterations, rate, int(p_cols)))
    print(
        "\nParallelism: ",
        n_threads,
        total,
        int(p_cols),
        rate,
        (timestamps[-1]) / 1000,
        p_solution.argmax(dim=1),
    )


def stateless_solver_parallelism_ablation_exp(
    map_path,
    agent_path,
    n_paths,
    temp,
    verbose,
    n_seconds,
    results_dir: Path,
):
    """
    NOTE: Uses multiprocessing, must be executed after if __name__ == "__main__"
    """

    import time
    import json

    import torch

    import torch_parallel_lns as parallel_lns

    results_dir.mkdir(parents=True, exist_ok=True)

    inst = instance.instance(
        map_path,
        agent_path,
        n_paths=n_paths,
        instance_name="Optimistic Iteration",
        agent_path_temp=temp,
        verbose=verbose,
    )

    subset_size = 20

    agents = list(sorted(inst.agents.values(), key=lambda a: a.id))

    p_cmatrix = parallel_lns.build_cmatrix(agents)

    table = PathTable(
        inst.num_of_rows,
        inst.num_of_cols,
        inst.num_agents,
    )

    solvers.random_initial_solution(inst, table)
    n_agents = inst.num_agents

    results = []

    # no threads
    start = time.time()
    pbar = tqdm.tqdm(range(n_seconds))
    total = 0
    log_t = 0
    timestamps = []
    iterations = []

    p_solution = torch.zeros((n_agents, n_paths), dtype=torch.int8)
    for agent in inst.agents.values():
        p_solution[agent.id - 1][agent.path_id] = 1

    p_cols = int(parallel_lns.solution_cmatrix(p_cmatrix, p_solution).sum().item() // 2)

    while time.time() - start < n_seconds:
        p_solution, p_cols = parallel_lns.run_iteration(
            p_cmatrix,
            p_solution,
            p_cols,
            parallel_lns.Configuration(
                n_agents,
                n_paths,
                destroy_method=parallel_lns.random_destroy_method,
                repair_method=parallel_lns.pp_repair_method,
                neighborhood_size=subset_size,
            ),
        )
        total += 1
        timestamp = time.time()

        if timestamp - start > log_t:
            timestamps.append((timestamp - start) * 1000)
            iterations.append(total)
            log_t += 0.5

        pbar.set_description(f"Iterations: {total} Cols: {int(p_cols)}")
        pbar.n = timestamp - start
        pbar.refresh()

    rate = int(np.mean(np.diff(iterations) / np.diff(timestamps)) * 1000)
    results.append((0, timestamps, iterations, rate, int(p_cols)))

    print("\nNo parallelism: ", total, int(p_cols), rate)

    # with threads
    for n_threads in range(1, 11):
        p_solution = torch.zeros((n_agents, n_paths), dtype=torch.int8)
        for agent in inst.agents.values():
            p_solution[agent.id - 1][agent.path_id] = 1

        p_cols = int(
            parallel_lns.solution_cmatrix(p_cmatrix, p_solution).sum().item() // 2
        )

        _, p_cols, timestamps, iterations = parallel_lns.run_parallel(
            p_cmatrix,
            p_solution,
            p_cols,
            c=parallel_lns.Configuration(
                n_agents,
                n_paths,
                destroy_method=parallel_lns.random_destroy_method,
                repair_method=parallel_lns.pp_repair_method,
                neighborhood_size=subset_size,
            ),
            n_threads=n_threads,
            n_seconds=n_seconds,
        )

        iterations = np.array(iterations)
        timestamps = np.array(timestamps)
        last_zero_col_index = np.where(iterations >= n_threads)[0][0]

        rate = int(
            np.mean(
                np.diff(iterations[last_zero_col_index:])
                / np.diff(timestamps[last_zero_col_index:])
            )
            * 1000
        )

        results.append((n_threads, timestamps, iterations, rate, int(p_cols)))
        print("\nParallelism: ", n_threads, total, int(p_cols), rate)

    for n_threads, timestamps, iterations, rate, p_cols in results:
        plt.plot(
            timestamps,
            iterations,
            label=f"P={n_threads} ({rate} iter/s)",
        )

    with (results_dir / "p_iterations_ablation_results.json").open("w") as f:
        json_results = {
            f"P{n_threads}": {
                "n_threads": int(n_threads),
                "timestamps": np.array(timestamps).tolist(),
                "iterations": np.array(iterations).tolist(),
                "rate": int(rate),
                "cols": int(p_cols),
            }
            for n_threads, timestamps, iterations, rate, p_cols in results
        }

        json.dump(json_results, f)

    plt.title("Parallelism Ablation", fontsize=16)
    plt.xlabel("Time (ms)", fontsize=12)
    plt.ylabel("Iterations", fontsize=12)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / "parallelism_ablation.png")
