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


def matrix_solver_test_exp(
    map_path, agent_path, n_paths, test_iterations, temp, verbose
):
    from MatrixPathTable import MatrixPathTable
    import matrix_solvers
    import copy
    import time

    def format_cmatrix_for_diff(cm: list[list[int]]) -> str:
        return "\n".join(" ".join(f"{int(x):3}" for x in row) for row in cm)

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

    m_inst = copy.deepcopy(inst)
    m_table = MatrixPathTable(
        agents = list(sorted(m_inst.agents.values(), key=lambda a: a.id))
    )

    np.random.seed(42)

    solvers.random_initial_solution(inst, table)
    for agent in inst.agents.values():
        m_table.insert_path(agent.id, agent.path_id)
        m_inst.agents[agent.id].path_id = agent.path_id

    for agent, m_agent in zip(inst.agents.values(), m_inst.agents.values()):
        assert (agent.id, agent.path_id) == (m_agent.id, m_agent.path_id)
        np.testing.assert_equal(
            agent.paths[agent.path_id], m_agent.paths[m_agent.path_id]
        )

    solver = solvers.IterativeRandomLNS(
        inst,
        table,
        subset_size=subset_size,
        num_iterations=0,
        destroy_method_name="priority",
        low_level_solver_name="pp",
    )

    m_solver = matrix_solvers.MatrixIterativeRandomLNS(
        m_inst,
        m_table,
        subset_size=subset_size,
        num_iterations=0,
        destroy_method_name="priority",
        low_level_solver_name="pp",
        verbose=True
    )

    assert solver.num_collisions == m_solver.num_collisions, (
        "Invalid initial collisions",
        solver.num_collisions,
        m_solver.num_collisions,
    )

    durations = []
    m_durations = []
    for iteration in tqdm.tqdm(range(test_iterations)):
        random_state = np.random.get_state()

        start = time.time()
        solver.run_iteration()
        duration_ms = (time.time() - start) * 1000

        np.random.set_state(random_state)
        start = time.time()
        m_solver.run_iteration()
        m_duration_ms = ((time.time() - start) * 1000)

        durations.append(duration_ms)
        m_durations.append(m_duration_ms)

        assert solver.num_collisions == m_solver.num_collisions, (
            solver.num_collisions, m_solver.num_collisions
        )

        for agent, m_agent in zip(inst.agents.values(), m_inst.agents.values()):
            value = (m_agent.id, m_agent.path_id)
            expected = (agent.id, agent.path_id)
            assert value == expected, (value, expected)
            np.testing.assert_equal(
                agent.paths[agent.path_id], m_agent.paths[m_agent.path_id]
            )

        cmatrix = solver.path_table.get_collisions_matrix(inst.num_agents)
        m_cmatrix = m_solver.path_table.get_collisions_matrix()

        cmatrix = format_cmatrix_for_diff(cmatrix)
        m_cmatrix = format_cmatrix_for_diff(m_cmatrix)

        assert cmatrix == m_cmatrix


def parallel_solver_test_exp(
    map_path, agent_path, n_paths, test_iterations, temp, verbose, executor_name
):
    """
    Test that parallel solver with parallelism=1 is identical to
    standard solver.
    """

    import copy
    import time
    from PathTable import iter_edges, iter_vertices

    subset_size = 20

    # if too low then p_solver might not accept a solution
    # and will cause test to fail, >100 is a good amount of iterations
    n_iterations = 100

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
    solvers.random_initial_solution(inst, table)

    p_inst = copy.deepcopy(inst)
    p_table = copy.deepcopy(table)

    for agent, p_agent in zip(inst.agents.values(), p_inst.agents.values()):
        assert (agent.id, agent.path_id) == (p_agent.id, p_agent.path_id)
        np.testing.assert_equal(
            agent.paths[agent.path_id], p_agent.paths[p_agent.path_id]
        )

    assert table.table == p_table.table
    assert table.edges == p_table.edges

    solver = solvers.IterativeRandomLNS(
        inst,
        table,
        subset_size=subset_size,
        num_iterations=n_iterations,
        random_seed=0
    )

    p_solver = solvers.ParallelIterativeRandomLNS(
        p_inst,
        p_table,
        subset_size=subset_size,
        parallelism=1,
        num_iterations=n_iterations,
        executor_name=executor_name
    )

    durations = []
    p_durations = []
    for iteration in tqdm.tqdm(range(test_iterations)):
        start = time.time()
        _, solver_iterations = solver.run(early_stopping=True)
        duration_ms = (time.time() - start) * 1000

        start = time.time()
        p_iterations = p_solver.run_iteration(do_seed=True)
        p_duration_ms = ((time.time() - start) * 1000) / p_iterations

        durations.append(duration_ms)
        p_durations.append(p_duration_ms)

        table = copy.deepcopy(solver.path_table)
        inst = copy.deepcopy(solver.instance)
        p_table = copy.deepcopy(p_solver.path_table)
        p_inst = copy.deepcopy(p_solver.instance)

        assert solver_iterations == p_iterations, (solver_iterations, p_iterations)
        assert solver.num_collisions == p_solver.num_collisions, (
            solver.num_collisions, p_solver.num_collisions
        )

        for agent, p_agent in zip(inst.agents.values(), p_inst.agents.values()):
            value = (p_agent.id, p_agent.path_id)
            expected = (agent.id, agent.path_id)
            assert value == expected, (value, expected)
            np.testing.assert_equal(
                agent.paths[agent.path_id], p_agent.paths[p_agent.path_id]
            )

            vertices = list(iter_vertices(agent.paths[agent.path_id]))
            edges = list(iter_edges(agent.paths[agent.path_id]))

            for vertex in table.table:
                if vertex in vertices:
                    assert agent.id in table.table[vertex]
                else:
                    assert agent.id not in table.table[vertex]

            for edge in table.edges:
                if edge in edges:
                    assert agent.id in table.edges[edge]
                else:
                    assert agent.id not in table.edges[edge]

            for vertex in p_table.table:
                if vertex in vertices:
                    assert p_agent.id in p_table.table[vertex]
                else:
                    assert p_agent.id not in p_table.table[vertex]

            for edge in p_table.edges:
                if edge in edges:
                    assert p_agent.id in p_table.edges[edge]
                else:
                    assert p_agent.id not in p_table.edges[edge]

        if table.table != p_table.table:
            keys = list(table.table.keys())
            p_keys = list(p_table.table.keys())

            diff = set(keys) ^ set(p_keys)

            assert len(keys) == len(p_keys), (len(keys), len(p_keys))

            for vertex in diff:
                print(vertex in table.table, vertex in p_table.table)
                cell = table.table[vertex]
                p_cell = p_table.table[vertex]

                assert cell == p_cell, (vertex.to_int(), cell, p_cell)

                if cell != p_cell:
                    # get difference in sets
                    diff = cell ^ p_cell

                    for agent_id in diff:
                        path_id = inst.agents[agent_id].path_id
                        p_path_id = p_inst.agents[agent_id].path_id

                        assert path_id == p_path_id, (agent, vertex.to_int(), path_id, p_path_id)

            assert table.table == p_table.table

        assert table.edges == p_table.edges


def parallelism_ablation_exp(
    map_path,
    agent_path,
    n_paths,
    n_iterations,
    sub_iterations,
    temp,
    verbose,
    results_dir,
    executor_name,
):
    import time
    from pathlib import Path

    subset_size = 20

    random_state = np.random.get_state()

    def run_no_parallel():
        np.random.set_state(random_state)

        # run without parallelism
        inst = instance.instance(
            map_path,
            agent_path,
            n_paths=n_paths,
            instance_name="Parallelism ablation experiment",
            agent_path_temp=temp,
            verbose=verbose,
        )

        table = PathTable(
            inst.num_of_rows,
            inst.num_of_cols,
            inst.num_agents,
        )

        solvers.random_initial_solution(inst, table)
        solver = solvers.IterativeRandomLNS(
            inst,
            table,
            subset_size=subset_size,
            num_iterations=sub_iterations,
            destroy_method_name="w-random",
        )

        durations = []
        collisions = []
        pbar = tqdm.tqdm(range(n_iterations))
        for iteration in pbar:
            start = time.time()
            _, sub_iterations_ran = solver.run(early_stopping=True)
            duration = time.time() - start
            duration /= sub_iterations_ran

            collisions.append(solver.num_collisions)

            pbar.set_description(
                f"Parallelism: {parallelism} "
                f"Collisions: {solver.num_collisions} "
            )
            duration_ms = duration * 1000
            durations.append(duration_ms)

        return durations, collisions


    def run(parallelism: int) -> tuple[list[int], list[int]]:
        np.random.set_state(random_state)

        inst = instance.instance(
            map_path,
            agent_path,
            n_paths=n_paths,
            instance_name="Parallelism ablation experiment",
            agent_path_temp=temp,
            verbose=verbose,
        )

        table = PathTable(
            inst.num_of_rows,
            inst.num_of_cols,
            inst.num_agents,
        )

        solvers.random_initial_solution(inst, table)
        solver = solvers.ParallelIterativeRandomLNS(
            inst,
            table,
            subset_size=subset_size,
            parallelism=parallelism,
            num_iterations=sub_iterations,
            destroy_method_name="w-random",
            executor_name=executor_name
        )

        durations = []
        collisions = []
        pbar = tqdm.tqdm(range(n_iterations))
        for iteration in pbar:
            start = time.time()
            sub_iterations_ran = solver.run_iteration()
            duration = time.time() - start
            duration /= sub_iterations_ran

            collisions.append(solver.num_collisions)

            pbar.set_description(
                f"Parallelism: {parallelism} "
                f"Collisions: {solver.num_collisions} "
            )

            duration_ms = duration * 1000
            durations.append(duration_ms)

        return durations, collisions

    parallelisms = list(range(0, 9))

    results = []
    for parallelism in parallelisms:
        if parallelism == 0:
            durations, collisions = run_no_parallel()
        else:
            durations, collisions = run(parallelism)

        results.append((parallelism, durations, collisions))

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    for parallelism, durations, collisions in results:
        with (results_dir / f"p_{parallelism}.csv").open("w") as f:
            f.write("duration,collisions\n")
            for duration, collision in zip(durations, collisions):
                f.write(f"{duration},{collision}\n")
