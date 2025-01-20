from typing import TypedDict, Literal

import torch.nn.functional
import torch

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
import CBS
from pathlib import Path
import torch_parallel_lns as lns
from Agent import Agent


def run_scenario(
    map_path,
    agent_path,
    solver,
    log_file="experiments.csv",
    verbose=True,
    n_paths=2,
    temp=1,
):
    inst = instance.instance(
        map_path, agent_path, solver, verbose, n_paths, agent_path_temp=temp
    )
    logger = Logger(log_file, inst)
    t = PathTable(inst.num_of_rows, inst.num_of_cols)
    solver = solvers.RandomPP(inst, t)
    logger.start()
    solver.solve()
    n_solved = solver.n_solved
    logger.end()
    logger.log(n_solved)


def run_all_scenarios(map_name, agent_directory, solver, verbose=True, n_paths=2):
    for agent_name in os.listdir(agent_directory):
        agent_path = os.path.join(agent_directory, agent_name)
        print(agent_path)
        run_scenario(map_name, agent_path, solver, verbose=True, n_paths=2)


def temp_ablation(map_path, agent_path, solver, verbose=True, n_paths=2, temp=1):
    open("temp_ablation.csv", "a")
    temps = np.logspace(-1, 0, 100)
    # temps = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.8, 1, 1.1, 1.2 , 1.3 , 1.5]
    for temp in temps:
        run_scenario(
            map_path, agent_path, solver, "temp_ablation.csv", verbose, n_paths, temp
        )


def pandas_exp():
    df = pd.read_csv("n_path_ablation.csv")
    # df = df[['percent_solved', 'agent_temp']]
    df.plot(x="n_paths", y="percent_solved")
    plt.show()
    # print(df.to_string())
    # print(df['percent_solved'].to_string())


def n_path_ablation(map_path, agent_path, solver, verbose=True, temp=1):
    open("n_path_ablation.csv", "a")
    n_paths = range(1, 33)
    # temps = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.8, 1, 1.1, 1.2 , 1.3 , 1.5]
    for n_path in n_paths:
        run_scenario(
            map_path, agent_path, solver, "n_path_ablation.csv", verbose, n_path, temp
        )


def group_size_ablation(
    map_path, agent_path, solver_name, verbose=True, n_paths=2, temp=1
):
    open("group_size_ablation.csv", "a")
    group_sizes = [10, 15, 20, 25, 30]  # range(1, 12)
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
    label = "num_of_cols in random PP LNS (varying group sizes, 1000 iteration)"
    x_axis_label = "iterations"
    y_axis_label = "collision counts"
    y_labels = [f"subset size = {size}" for size in group_sizes]
    plot_line_graphs(
        x_axis, collision_counts, label, x_axis_label, y_axis_label, y_labels
    )


def exhaustive_vs_random_exp(
    map_path, agent_path, solver_name, verbose=True, n_paths=3, temp=1
):
    # open('test.csv', "a")
    solvers_list = ["pp", "exhaustive"]  # range(1, 12)
    collision_counts = []
    x_axis = []
    for solver in solvers_list:
        s = instance.instance(map_path, agent_path, solver, verbose, n_paths, temp)
        t = PathTable(s.num_of_rows, s.num_of_cols)
        solvers.random_initial_solution(s, t)
        # adj_matrix = t.get_collisions_matrix(s.num_agents)
        # subset = get_largest_connected_component(adj_matrix)
        # print(get_degrees_of_vertices_dict(adj_matrix, subset))
        solver = solvers.IterativeRandomLNS(
            s,
            t,
            3,
            destroy_method_name="w-random",
            num_iterations=1000,
            low_level_solver_name=solver,
        )
        x_axis = range(solver.num_iterations + 1)
        solver.run()
        collision_counts += [solver.collision_statistics]
    label = "num_of_cols in random PP LNS (varying solvers, 1000 iteration)"
    x_axis_label = "iterations"
    y_axis_label = "collision counts"
    y_labels = solvers_list
    plot_line_graphs(
        x_axis, collision_counts, label, x_axis_label, y_axis_label, y_labels
    )


def destroy_method_ablation_exp(
    map_path, agent_path, solver_name, verbose=True, n_paths=8, temp=1
):
    # open('test.csv', "a")
    ds_list = ["cc", "random", "w-random"]  # range(1, 12)
    collision_counts = []
    x_axis = []
    for ds in ds_list:
        s = instance.instance(map_path, agent_path, solver_name, verbose, n_paths, temp)
        t = PathTable(s.num_of_rows, s.num_of_cols, num_of_agents=s.num_agents)
        paths = k_shortest_paths(s.map_graph, (1, 26), (25, 1), 7)
        [print(path) for path in paths]
        draw_graph_highlight_paths(s.map_graph, paths)
        solvers.random_initial_solution(s, t)
        solver = solvers.IterativeRandomLNS(
            s,
            t,
            10,
            destroy_method_name=ds,
            num_iterations=500,
            low_level_solver_name=solver_name,
        )
        x_axis = range(solver.num_iterations + 1)
        solver.run()
        collision_counts += [solver.collision_statistics]
    label = "num_of_cols in random PP LNS (varying destroy methods, 1000 iteration)"
    x_axis_label = "iterations"
    y_axis_label = "collision counts"
    y_labels = ds_list
    plot_line_graphs(
        x_axis, collision_counts, label, x_axis_label, y_axis_label, y_labels
    )


def test_exp(map_path, agent_path, solver_name, verbose=True, n_paths=3, temp=1):
    # open('test.csv', "a")
    group_sizes = [20]  # range(1, 12)
    solvers_list = ["pp", "rank-pp"]  # range(1, 12)
    variants = list(itertools.product(group_sizes, solvers_list))
    collision_counts = []
    x_axis = []
    for size, solver_name in variants:
        s = instance.instance(map_path, agent_path, solver_name, n_paths=n_paths)
        t = PathTable(s.num_of_rows, s.num_of_cols)
        solvers.random_initial_solution(s, t)
        adj_matrix = t.get_collisions_matrix(s.num_agents)
        largest_cc = get_largest_connected_component(adj_matrix)
        random_walk_until_neighborhood_is_full(adj_matrix, largest_cc, subset_size=5)
        # print(t.get_collisions_matrix(s.num_agents).sum(axis=1))
        # t.get_agent_collisions_for_paths(s.agents[2], s.num_agents)
        solver = solvers.IterativeRandomLNS(
            s,
            t,
            size,
            destroy_method_name="cc",
            low_level_solver_name=solver_name,
            num_iterations=3000,
        )
        x_axis = range(solver.num_iterations + 1)
        solver.run()
        collision_counts += [solver.collision_statistics]
        visualize(
            s,
            t,
        )
    label = (
        "num_of_cols in random PP LNS (varying group sizes and solvers, 1000 iteration)"
    )
    x_axis_label = "iterations"
    y_axis_label = "collision counts"
    y_labels = [f"subset size = {size}, solver = {solver}" for size, solver in variants]
    plot_line_graphs(
        x_axis, collision_counts, label, x_axis_label, y_axis_label, y_labels
    )


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
        verbose=verbose,
    )

    if max_agents is not None:
        new_agents = dict([(k, v) for k, v in inst.agents.items()][:max_agents])
        inst.agents = new_agents
        inst.num_agents = len(new_agents)

    table = PathTable(inst.num_of_rows, inst.num_of_cols, inst.num_agents)

    group_size = 20

    solvers.random_initial_solution(inst, table)

    solver = solvers.IterativeRandomLNS(
        inst, table, group_size, num_iterations=n_iterations
    )

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
                print(f"\n**** Agent {agent_id} ****")
                print(f"\nInitial number of collisions: {initial_collisions}")

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
                        print(f"\n      Found new path selection: {path_id} ****")
                        print(
                            f"\n      New number of collisions: {best_collisions} ****"
                        )

                table.remove_path(agent_id, agent.paths[path_id])

            collision_reduction = initial_collisions - best_collisions
            if collision_reduction > 0:
                total_improved_agents += 1
                total_improvement += collision_reduction

            # restore initial state
            table.insert_path(agent_id, agent.paths[initial_path])

        # print results
        if verbose:
            print(f"\n**** Iteration {iteration} ****")
            print(f"\nTotal colliding agents: {total_colliding_agents}")
            print(f"\nTotal improved agents: {total_improved_agents}")
            print(f"\nTotal improvement: {total_improvement}")

        result: OptimisticIterationResult = {
            "total_colliding_agents": int(total_colliding_agents),
            "total_improved_agents": int(total_improved_agents),
            "total_improvement": int(total_improvement),
        }

        results.append(result)

        on_result_callback(result)

        # improve with solver
        solver.run_iteration()

        p_bar.set_description(f"Collisions: {solver.num_collisions}")

    return inst, table, results


def CBS_EXP(map_path, agent_path, solver_name, verbose=True, n_paths=20, temp=1):
    s = instance.instance(map_path, agent_path, solver_name, n_paths=n_paths)
    solver = CBS.CBS(s)
    solver.search()


def CBS_num_agents_exp(
    map_path, agent_path, solver_name, verbose=False, n_paths=20, temp=1
):
    s = instance.instance(map_path, agent_path, solver_name, n_paths=n_paths)
    agents_list = range(2, s.num_agents)
    expanded_list = []
    for num_agents in agents_list:
        with open(agent_path) as agent_file:
            head = [next(agent_file) for _ in range(num_agents + 1)]
            with open("temp.txt", "w") as temp:
                for line in head:
                    temp.write(line)
        s_temp = instance.instance(map_path, "temp.txt", solver_name, n_paths=n_paths)
        solver = CBS.CBS(s_temp, verbose=verbose)
        solver.search()
        expanded_list += [solver.expanded]
    print({a: e for a, e in zip(agents_list, expanded_list)})
    title = "Expanded nodes in CBS"
    x_axis_label = "Number of agents"
    y_axis_label = "Expanded nodes"
    x_axis = agents_list
    y_axis = expanded_list
    plot_line_graph(
        x_axis, y_axis, title, x_axis_label=x_axis_label, y_axis_label=y_axis_label
    )


def CBS_lns_improvement_exp(
    map_path, agent_path, solver_name, verbose=False, n_paths=20, temp=1
):
    s = instance.instance(map_path, agent_path, solver_name, n_paths=n_paths)
    agents_list = range(40, 45, 2)
    lns_collisions = []
    cbs_collisions = []
    for num_agents in agents_list:
        with open(agent_path) as agent_file:
            head = [next(agent_file) for _ in range(num_agents + 1)]
            with open("temp.txt", "w") as temp:
                for line in head:
                    temp.write(line)
        s_temp = instance.instance(map_path, "temp.txt", solver_name, n_paths=n_paths)
        t = PathTable(
            s_temp.num_of_rows, s_temp.num_of_cols, num_of_agents=s_temp.num_agents
        )
        solver_t, _ = solvers.generate_random_random_solution_iterative(s_temp, t)
        lns_num_cols = solver_t.num_collisions
        solution = torch.tensor(
            [s_temp.agents[i + 1].path_id for i in range(s_temp.num_agents)]
        )
        solution_one_hot = torch.nn.functional.one_hot(solution, n_paths)
        solver = CBS.CBS(s_temp, solution_one_hot, verbose=verbose)
        _, cbs_col_count = solver.search()
        lns_collisions += [lns_num_cols]
        cbs_collisions += [cbs_col_count]
    title = "Collisions in Lns and LNS+CBS"
    x_axis = agents_list
    y_axis = [lns_collisions, cbs_collisions]
    x_axis_label = "Number of agents"
    y_axis_label = "collision counts"
    y_labels = ["LNS", "LNS + CBS"]
    plot_line_graphs(x_axis, y_axis, title, x_axis_label, y_axis_label, y_labels)


def CBS_lns_init_cols_exp(
    map_path, agent_path, solver_name, verbose=False, n_paths=20, temp=1
):
    s = instance.instance(map_path, agent_path, solver_name, n_paths=n_paths)
    agents_list = range(3, s.num_agents, 5)
    lns_collisions = []
    cbs_collisions = []
    cbs_lns_collisions = []
    for num_agents in agents_list:
        with open(agent_path) as agent_file:
            head = [next(agent_file) for _ in range(num_agents + 1)]
            with open("temp.txt", "w") as temp:
                for line in head:
                    temp.write(line)
        s_temp = instance.instance(map_path, "temp.txt", solver_name, n_paths=n_paths)
        t = PathTable(
            s_temp.num_of_rows, s_temp.num_of_cols, num_of_agents=s_temp.num_agents
        )

        solver = CBS.CBS(s_temp, verbose=verbose)
        _, cbs_col_count = solver.search()
        cbs_collisions += [cbs_col_count]

        solver_t, _ = solvers.generate_random_random_solution_iterative(s_temp, t)
        lns_num_cols = solver_t.num_collisions
        lns_collisions += [lns_num_cols]
        solution = torch.tensor(
            [s_temp.agents[i + 1].path_id for i in range(s_temp.num_agents)]
        )
        solution_one_hot = torch.nn.functional.one_hot(solution, n_paths)
        solver = CBS.CBS(s_temp, solution_one_hot, verbose=verbose)
        _, cbs_lns_col_count = solver.search()
        cbs_lns_collisions += [cbs_lns_col_count]
    title = "Collisions in CBS and CBSw/LNSInit"
    x_axis = agents_list
    y_axis = [cbs_collisions, lns_collisions, cbs_lns_collisions]
    x_axis_label = "Number of agents"
    y_axis_label = "collision counts"
    y_labels = ["CBS", "LNS", "CBSw/LNSInit"]
    plot_line_graphs(x_axis, y_axis, title, x_axis_label, y_axis_label, y_labels)


def CBS_lns_init_expanded_nodes_exp(
    map_path, agent_path, solver_name, verbose=False, n_paths=20, temp=1
):
    s = instance.instance(map_path, agent_path, solver_name, n_paths=n_paths)
    agents_list = range(3, s.num_agents, 5)
    cbs_expanded = []
    cbs_lns_expanded = []
    for num_agents in agents_list:
        with open(agent_path) as agent_file:
            head = [next(agent_file) for _ in range(num_agents + 1)]
            with open("temp.txt", "w") as temp:
                for line in head:
                    temp.write(line)
        s_temp = instance.instance(map_path, "temp.txt", solver_name, n_paths=n_paths)
        t = PathTable(
            s_temp.num_of_rows, s_temp.num_of_cols, num_of_agents=s_temp.num_agents
        )

        solver = CBS.CBS(s_temp, verbose=verbose)
        _, cbs_col_count = solver.search()
        cbs_expanded += [solver.expanded]

        solver_t, _ = solvers.generate_random_random_solution_iterative(s_temp, t)
        solution = torch.tensor(
            [s_temp.agents[i + 1].path_id for i in range(s_temp.num_agents)]
        )
        solution_one_hot = torch.nn.functional.one_hot(solution, n_paths)
        solver = CBS.CBS(s_temp, solution_one_hot, verbose=verbose)
        _, cbs_lns_col_count = solver.search()
        cbs_lns_expanded += [solver.expanded]

    title = "Expanded nodes in CBS and CBSw/LNSInit"
    x_axis = agents_list
    y_axis = [cbs_expanded, cbs_lns_expanded]
    x_axis_label = "Number of agents"
    y_axis_label = "Expanded nodes"
    y_labels = ["CBS", "CBSw/LNSInit"]
    plot_line_graphs(x_axis, y_axis, title, x_axis_label, y_axis_label, y_labels)


def cbs_lns_exp(map_path, agent_path, solver_name, verbose=True, n_paths=15, temp=1):
    s = instance.instance(map_path, agent_path, solver_name, verbose, n_paths, temp)
    t = PathTable(s.num_of_rows, s.num_of_cols, num_of_agents=s.num_agents)
    solvers.generate_random_random_solution_iterative(s, t)
    solution = torch.tensor([s.agents[i + 1].path_id for i in range(s.num_agents)])
    solution_one_hot = torch.nn.functional.one_hot(solution, n_paths)
    cbs = CBS.CBS(s, solution_one_hot)
    cbs.search()


def stateless_solver_test_exp(
    map_path, agent_path, n_paths, test_iterations, temp, verbose
):
    import torch_parallel_lns as lns
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

    p_cmatrix = lns.build_cmatrix(list(inst.agents.values()))

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

    p_collisions = lns.solution_cmatrix(p_cmatrix, p_solution)
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
        p_solution, p_num_collisions = lns.run_iteration(
            p_cmatrix,
            p_solution,
            p_num_collisions,
            lns.Configuration(
                n_agents,
                n_paths,
                destroy_method=[lns.random_destroy_method],
                repair_method=[lns.pp_repair_method],
                neighborhood_size=subset_size,
            ),
        )
        p_duration_ms = (time.time() - start) * 1000

        durations.append(duration_ms)
        p_durations.append(p_duration_ms)

        cmatrix = solver.path_table.get_collisions_matrix(inst.num_agents)
        p_collisions = lns.solution_cmatrix(p_cmatrix, p_solution).numpy()

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


def stateless_solver_no_parallelism_exp(
    map_path,
    agent_path,
    n_paths,
    temp,
    verbose,
    n_seconds,
    config: lns.Configuration,
    results_dir: Path,
    optimal: int | None = None,
):
    import time
    import json

    import torch

    results_dir.mkdir(parents=True, exist_ok=True)

    inst = instance.instance(
        map_path,
        agent_path,
        n_paths=n_paths,
        instance_name="Optimistic Iteration",
        agent_path_temp=temp,
        verbose=verbose,
    )

    agents = list(sorted(inst.agents.values(), key=lambda a: a.id))
    assert config.n_agents == len(
        agents
    ), f"Invalid number of agents: {config.n_agents} expected {len(agents)}"

    p_cmatrix = lns.build_cmatrix(agents)

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

    p_cols = int(lns.solution_cmatrix(p_cmatrix, p_solution).sum().item() // 2)

    while time.time() - start < n_seconds:
        p_solution, p_cols = lns.run_iteration(
            p_cmatrix,
            p_solution,
            p_cols,
            config,
        )
        total += 1
        timestamp = time.time()

        if timestamp - start > log_t:
            timestamps.append((timestamp - start) * 1000)
            iterations.append(total)
            log_t += 0.5

        if optimal is not None and p_cols <= optimal:
            break

        pbar.set_description(f"Iterations: {total} Cols: {int(p_cols)}")
        pbar.n = timestamp - start
        pbar.refresh()

    assert len(iterations) > 1, "Ran 1 iteration"
    rate = int(np.mean(np.diff(iterations) / np.diff(timestamps)) * 1000)
    results.append((0, timestamps, iterations, rate, int(p_cols)))

    print(f"\nNo parallelism: {total}, {p_cols=}, {rate=}")


def stateless_solver_parallelism_exp(
    map_path,
    agent_path,
    n_paths,
    temp,
    verbose,
    n_seconds,
    config: lns.Configuration,
    n_threads,
    results_dir: Path | None = None,
    optimal: int | None = None,
):
    """
    NOTE: Uses multiprocessing, must be executed after if __name__ == "__main__"
    """
    import time
    import json

    import torch

    import torch_parallel_lns as lns

    if results_dir is not None:
        results_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()
    print("Loading instance")
    inst = instance.instance(
        map_path,
        agent_path,
        n_paths=n_paths,
        instance_name="Optimistic Iteration",
        agent_path_temp=temp,
        verbose=verbose,
    )
    print(f"Loaded instance in {time.time() - start:.2f}s")

    agents = list(sorted(inst.agents.values(), key=lambda a: a.id))
    assert config.n_agents == len(
        agents
    ), f"Invalid number of agents: {config.n_agents} expected {len(agents)}"

    p_cmatrix = lns.build_cmatrix(agents)

    table = PathTable(
        inst.num_of_rows,
        inst.num_of_cols,
        inst.num_agents,
    )

    solvers.random_initial_solution(inst, table)
    n_agents = inst.num_agents

    timestamps = []
    collisions = []

    p_solution = torch.zeros((n_agents, n_paths), dtype=torch.int8)
    for agent in inst.agents.values():
        p_solution[agent.id - 1][agent.path_id] = 1

    p_cols = int(lns.solution_cmatrix(p_cmatrix, p_solution).sum().item() // 2)

    p_solution = torch.zeros((n_agents, n_paths), dtype=torch.int8)
    for agent in inst.agents.values():
        p_solution[agent.id - 1][agent.path_id] = 1

    p_cols = int(lns.solution_cmatrix(p_cmatrix, p_solution).sum().item() // 2)

    p_solution, p_cols, timestamps, collisions = lns.run_parallel(
        p_cmatrix,
        p_solution,
        p_cols,
        config=config,
        n_threads=n_threads,
        n_seconds=n_seconds,
        optimal=optimal,
    )

    collisions = np.array(collisions)
    timestamps = np.array(timestamps)

    print(
        f"\nResults: p_cols={int(p_cols)}, "
        f"time_elapsed={(timestamps[-1]) / 1000}, p_solution={p_solution.argmax(dim=1)}"
    )

    if results_dir is not None:
        with (results_dir / "data.json").open("w") as f:
            json_results = {
                "n_threads": int(n_threads),
                "timestamps": np.array(timestamps).tolist(),
                "collisions": np.array(collisions).tolist(),
                "cols": int(p_cols),
            }

            json.dump(json_results, f)


def stateless_solver_parallelism_ablation_exp(
    map_path,
    agent_path,
    n_paths,
    temp,
    verbose,
    n_seconds,
    config: lns.Configuration,
    results_dir: Path,
):
    """
    NOTE: Uses multiprocessing, must be executed after if __name__ == "__main__"
    """

    import time
    import json

    import torch

    import torch_parallel_lns as lns

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

    p_cmatrix = lns.build_cmatrix(agents)

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

    p_cols = int(lns.solution_cmatrix(p_cmatrix, p_solution).sum().item() // 2)

    while time.time() - start < n_seconds:
        p_solution, p_cols = lns.run_iteration(
            p_cmatrix,
            p_solution,
            p_cols,
            lns.Configuration(
                n_agents,
                n_paths,
                destroy_method=[lns.random_destroy_method],
                repair_method=[lns.pp_repair_method],
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

        p_cols = int(lns.solution_cmatrix(p_cmatrix, p_solution).sum().item() // 2)

        _, p_cols, timestamps, iterations = lns.run_parallel(
            p_cmatrix,
            p_solution,
            p_cols,
            config=lns.Configuration(
                n_agents,
                n_paths,
                destroy_method=[lns.random_destroy_method],
                repair_method=[lns.pp_repair_method],
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


def collisions_by_ms_aggregate_exp(
    scenarios: Path,
    n_seconds: int,
    n_threads: int,
):
    import json
    import time
    import torch
    import torch_parallel_lns as lns

    for old_result in scenarios.glob("**/*.results"):
        old_result.unlink(missing_ok=True)

    pbar = tqdm.tqdm(scenarios.glob("**/*.path"))
    # np.random.seed(42)
    # np.random.seed(7)  # n_paths = 500, n_agents = 100
    seed = 12
    np.random.seed(seed)
    torch.manual_seed(seed)
    for path_file in pbar:
        pbar.set_description(f"{path_file.parent}/{path_file.name}")
        start = time.time()
        agent_paths = json.loads(path_file.read_text())
        # shuffle agent paths
        # np.random.shuffle(agent_paths)
        agents = []
        for i, paths in enumerate(agent_paths[:50]):
            # np.random.shuffle(paths)
            paths = paths[:400]
            agent = Agent(
                instance=None,
                agent_id=i + 1,
                start_loc=paths[0][0],
                end_loc=paths[0][-1],
                n_paths=len(paths),
            )
            agent.paths = paths
            agents.append(agent)

        n_agents = len(agents)
        n_paths = agents[0].n_paths
        neighborhood_size = 30
        config = lns.Configuration(
            n_agents=n_agents,
            n_paths=n_paths,
            destroy_method=[
                # parallel_lns.argmax_destroy_method,
                lns.weighted_random_destroy_method,
                # parallel_lns.random_destroy_method,
                # parallel_lns.random_destroy_method,
            ],
            repair_method=[
                # parallel_lns.pp_repair_method,
                # parallel_lns.pp_repair_method,
                # parallel_lns.pp_repair_method,
                lns.pp_repair_method,
            ],
            neighborhood_size=neighborhood_size,
            simulated_annealing=(10, 1, 5000),
            dynamic_neighborhood=2,
        )

        cmatrix = lns.build_cmatrix_fast(agents)
        solution = lns.random_initial_solution(n_agents, n_paths)
        sol_cmatrix = lns.solution_cmatrix(cmatrix, solution)
        initial_collisions = int(sol_cmatrix.sum() // 2)
        setup_time = time.time() - start

        _, final_collisions, timestamps, collisions = lns.run_parallel(
            cmatrix,
            solution,
            initial_collisions,
            config,
            n_threads,
            n_seconds,
            optimal=0,
        )

        timestamps = np.array(timestamps) + setup_time

        results_file = path_file.with_suffix(".results")
        results_file.write_text(
            json.dumps(
                {
                    "timestamps": np.array(timestamps).tolist(),
                    "collisions": np.array(collisions).tolist(),
                    "initial_collisions": int(initial_collisions),
                    "final_collisions": int(final_collisions),
                    "n_threads": int(n_threads),
                    "n_agents": int(n_agents),
                    "n_paths": int(n_paths),
                    "neighborhood_size": int(neighborhood_size),
                }
            )
        )


class UniformExperimentParams(TypedDict):
    n_agents: int
    n_paths: int
    destroy_method: Literal["random"]
    repair_method: Literal["pp"]
    neighborhood: int
    simulated_annealing: tuple[float, float, float] | None
    repetitions: int
    n_seconds: int
    n_threads: int
    cbs_max_expanded: int
    lns_initial_iterations: int
    map_file: str
    agent_file: str | None


def uniform_experiment(
    log_dir: Path,
    solve: bool,
    index: int,
    params: UniformExperimentParams,
    seed: int = 42,
    overwrite: bool = False,
):
    import path_generator
    import scenario_generator
    import networkx as nx
    import torch
    import json
    import CBS2

    map_file = Path(params["map_file"])
    methods = {"random": lns.random_destroy_method, "pp": lns.pp_repair_method}
    config = lns.Configuration(
        n_agents=params["n_agents"],
        n_paths=params["n_paths"],
        destroy_method=[methods[params["destroy_method"]]],
        repair_method=[methods[params["repair_method"]]],
        neighborhood_size=params["neighborhood"],
        simulated_annealing=params["simulated_annealing"],
    )

    # import research

    log_dir.mkdir(exist_ok=True, parents=True)
    experment_dir = log_dir / f"experiment_{index}"
    experment_dir.mkdir(exist_ok=overwrite)

    print(
        f"Running uniform expirment {index}: dir={experment_dir}"
        f" {seed=} params={json.dumps(params)}"
    )
    (experment_dir / "params.json").write_text(json.dumps(params))
    np.random.seed(seed=seed)
    torch.manual_seed(seed=seed)

    lns_initial_iterations = params["lns_initial_iterations"]
    cbs_max_expanded_nodes = params["cbs_max_expanded"]

    # files
    real_results_file = experment_dir / "real_results.json"
    synthetic_results_file = experment_dir / "synthetic_results.json"
    paths_file = experment_dir / "paths.json"
    real_cmatrix_file = experment_dir / "real_cmatrix.json"
    real_cmatrix_viz_file = experment_dir / "real_cmatrix_viz.txt"
    synthetic_cmatrix_file = experment_dir / "synthetic_cmatrix.json"
    synthetic_cmatrix_viz_file = experment_dir / "synthetic_cmatrix_viz.txt"

    # load map
    map_graph, _, _ = scenario_generator.load_map(map_file)
    assert nx.is_connected(map_graph), "Map is not connected"

    # generate paths
    while True:
        if params["agent_file"] is None:
            n_agents = int(config.n_agents * 1.5)
            agents = scenario_generator.generate_scenario(map_graph, n_agents)
        else:
            agents = scenario_generator.load_agents(
                Path(params["agent_file"]), flip_xy=True
            )
            assert len(agents) >= config.n_agents
            agents = agents[: config.n_agents]
            n_agents = config.n_agents

        paths = path_generator.generate_paths(map_graph, agents, config.n_paths)

        # build cmatrix
        cmatrix = lns.build_cmatrix_fast(paths)
        # print("\nOriginal:")
        # print("\n".join(["".join(["#" if c else "_" for c in r]) for r in cmatrix]))
        original_density = float(cmatrix.sum() / (cmatrix.shape[0] ** 2))
        print(f"{original_density=:.4f}\n")

        # remove all impossible pairs
        colliding_agents = (
            cmatrix.view((n_agents, config.n_paths, n_agents, config.n_paths))
            .amin(dim=(1, 3))
            .nonzero()
            .flatten()
            .unique()
        )

        # remove colliding agents
        paths = [p for i, p in enumerate(paths) if i not in colliding_agents]
        if len(paths) < config.n_agents:
            print(
                f"Generated {n_agents} agent had {len(paths)} valid agents,"
                f" needed {config.n_agents}, retrying...\n"
            )
            continue

        lens = [len(p) for p in paths]
        if not all(l == config.n_paths for l in lens):
            print(
                f"Generated paths had invalid lengths: {lens} {config.n_paths=}, retrying...\n"
            )
            continue

        print(
            f"{n_agents} agents had {len(colliding_agents)}"
            f" collisions - OK, using {config.n_agents}\n"
        )
        paths = paths[: config.n_agents]
        break

    assert len(paths) == config.n_agents
    lens = [len(p) for p in paths]
    assert all(l == config.n_paths for l in lens), (lens, config.n_paths)

    # log paths used
    paths_file.write_text(json.dumps(paths))

    # start evaluation
    cmatrix = lns.build_cmatrix_fast(paths)
    real_cmatrix_file.write_text(json.dumps(cmatrix.tolist()))
    # print("\nUpdated:")
    # print("\n".join(["".join(["#" if c else "_" for c in r]) for r in cmatrix]))
    real_cmatrix_viz_file.write_text(
        "\n".join(["".join(["#" if c else "_" for c in r]) for r in cmatrix])
    )

    # get density
    real_density = float(cmatrix.sum() / (cmatrix.shape[0] ** 2))
    print(f"{real_density=:.4f} {original_density=:.4f}\n")

    # get collisions
    if solve:
        solution = lns.random_initial_solution(config.n_agents, config.n_paths)
        collisions = int(lns.solution_cmatrix(cmatrix, solution).sum() // 2)

        # use lns for initial solution
        pbar = tqdm.tqdm(
            range(lns_initial_iterations), desc="Generating LNS initial solution"
        )
        for _ in pbar:
            solution, collisions = lns.run_iteration(
                cmatrix, solution, int(collisions), config
            )
            pbar.set_description(f"LNS Collisions: {collisions}")
            if collisions == 0:
                break

        if collisions > 0:
            print("\nRunning CBS...\n")
            print(f"CBS initial collisions: {collisions}\n")
            solver = CBS2.CBS(
                paths, solution, verbose=False, max_expanded=cbs_max_expanded_nodes
            )
            solution, collisions = solver.search()

        print(f"\nCollisions: {collisions}\n")
    else:
        collisions = -1

    real_results_file.write_text(
        json.dumps({"density": real_density, "collisions": int(collisions)})
    )

    # generate random collision matrix with mean density
    size = config.n_agents * config.n_paths
    cmatrix = (torch.rand(size, size) < (real_density / 2)).to(torch.bool)

    # ensure cmatrix is symmetric
    cmatrix = cmatrix | cmatrix.T
    # print("\nSynthetic:")
    # print("\n".join(["".join(["#" if c else "_" for c in r]) for r in cmatrix]))

    # get generate density
    synthetic_density = float(cmatrix.sum() / (cmatrix.shape[0] ** 2))
    ratio = real_density / synthetic_density
    print(f"{real_density=:.4f} {synthetic_density=:.4f} {ratio=:.4f}")

    synthetic_cmatrix_file.write_text(json.dumps(cmatrix.tolist()))
    synthetic_cmatrix_viz_file.write_text(
        "\n".join(["".join(["#" if c else "_" for c in r]) for r in cmatrix])
    )

    # get collisions
    solution = lns.random_initial_solution(config.n_agents, config.n_paths)
    initial_collisions = int(lns.solution_cmatrix(cmatrix, solution).sum() // 2)
    _, collisions, _, _ = lns.run_parallel(
        cmatrix=cmatrix,
        solution=solution,
        collisions=initial_collisions,
        config=config,
        n_threads=params["n_threads"],
        n_seconds=params["n_seconds"],
        optimal=0,
    )

    # solution = research.solve_ortools(cmatrix, config.n_agents, config.n_paths)
    # collisions = int(lns.solution_cmatrix(cmatrix, solution).sum() // 2)

    synthetic_results_file.write_text(
        json.dumps({"density": real_density, "collisions": collisions})
    )


class DensityExperimentParams(TypedDict):
    n_agents: int
    n_paths: int
    destroy_method: Literal["random"]
    repair_method: Literal["pp"]
    neighborhood: int
    simulated_annealing: tuple[float, float, float] | None
    repetitions: int
    n_iterations: int
    map_file: str


def density_experiment(
    log_dir: Path,
    params: DensityExperimentParams,
    density: float,
    seed: int = 42,
):
    import json

    log_dir.mkdir(exist_ok=False, parents=True)

    print(
        f"Running density expirment: dir={log_dir}"
        f" {seed=} {density=} params={json.dumps(params)}"
    )

    methods = {"random": lns.random_destroy_method, "pp": lns.pp_repair_method}
    config = lns.Configuration(
        n_agents=params["n_agents"],
        n_paths=params["n_paths"],
        destroy_method=[methods[params["destroy_method"]]],
        repair_method=[methods[params["repair_method"]]],
        neighborhood_size=params["neighborhood"],
        simulated_annealing=params["simulated_annealing"],
    )

    for i in range(params["repetitions"]):
        repetition_dir = log_dir / f"repetition_{i}.json"
        repetition_dir.mkdir(exist_ok=False, parents=False)

        synthetic_cmatrix_file = repetition_dir / f"cmatrix.json"
        synthetic_cmatrix_viz_file = repetition_dir / f"cmatrix_viz.txt"
        results_file = repetition_dir / f"results.json"

        size = config.n_agents * config.n_paths
        cmatrix = (torch.rand(size, size) < (density / 2)).to(torch.bool)

        # ensure cmatrix is symmetric
        cmatrix = cmatrix | cmatrix.T

        # get generate density
        synthetic_density = float(cmatrix.sum() / (cmatrix.shape[0] ** 2))
        ratio = density / synthetic_density
        print(f"{density=:.4f} {synthetic_density=:.4f} {ratio=:.4f}")

        synthetic_cmatrix_file.write_text(json.dumps(cmatrix.tolist()))
        synthetic_cmatrix_viz_file.write_text(
            "\n".join(["".join(["#" if c else "_" for c in r]) for r in cmatrix])
        )

        # get collisions
        solution = lns.random_initial_solution(config.n_agents, config.n_paths)
        collisions = int(lns.solution_cmatrix(cmatrix, solution).sum() // 2)
        for _ in range(params["n_iterations"]):
            solution, collisions = lns.run_iteration(
                cmatrix=cmatrix,
                solution=solution,
                collisions=collisions,
                config=config,
            )

            if collisions == 0:
                break

        result = {"density": density, "collisions": collisions}

        results_file.write_text(json.dumps(result))
