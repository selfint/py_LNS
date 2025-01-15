from typing import TypedDict

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



def CBS_EXP(map_path, agent_path, solver_name, verbose = True, n_paths = 20, temp = 1):
    s = instance.instance(map_path, agent_path, solver_name, n_paths = n_paths)
    solver = CBS.CBS(s)
    solver.search()


def CBS_num_agents_exp(map_path, agent_path, solver_name, verbose = False, n_paths = 20, temp = 1):
    s = instance.instance(map_path, agent_path, solver_name, n_paths = n_paths)
    agents_list = range(2, s.num_agents)
    expanded_list = []
    for num_agents in agents_list:
        with open(agent_path) as agent_file:
            head = [next(agent_file) for _ in range(num_agents+1)]
            with open('temp.txt', 'w') as temp:
                for line in head:
                    temp.write(line)
        s_temp = instance.instance(map_path, 'temp.txt', solver_name, n_paths = n_paths)
        solver = CBS.CBS(s_temp, verbose=verbose)
        solver.search()
        expanded_list += [solver.expanded]
    print({a: e for a, e in zip(agents_list, expanded_list)})
    title = 'Expanded nodes in CBS'
    x_axis_label = 'Number of agents'
    y_axis_label = 'Expanded nodes'
    x_axis = agents_list
    y_axis = expanded_list
    plot_line_graph(x_axis, y_axis, title, x_axis_label = x_axis_label, y_axis_label = y_axis_label)

def CBS_lns_improvement_exp(map_path, agent_path, solver_name, verbose = False, n_paths = 20, temp = 1):
    s = instance.instance(map_path, agent_path, solver_name, n_paths = n_paths)
    agents_list = range(40, 45, 2)
    lns_collisions = []
    cbs_collisions = []
    for num_agents in agents_list:
        with open(agent_path) as agent_file:
            head = [next(agent_file) for _ in range(num_agents+1)]
            with open('temp.txt', 'w') as temp:
                for line in head:
                    temp.write(line)
        s_temp = instance.instance(map_path, 'temp.txt', solver_name, n_paths = n_paths)
        t = PathTable(s_temp.num_of_rows, s_temp.num_of_cols, num_of_agents=s_temp.num_agents)
        solver_t, _ = solvers.generate_random_random_solution_iterative(s_temp, t)
        lns_num_cols = solver_t.num_collisions
        solution = torch.tensor([s_temp.agents[i + 1].path_id for i in range(s_temp.num_agents)])
        solution_one_hot = torch.nn.functional.one_hot(solution, n_paths)
        solver = CBS.CBS(s_temp,solution_one_hot, verbose=verbose)
        _, cbs_col_count = solver.search()
        lns_collisions += [lns_num_cols]
        cbs_collisions += [cbs_col_count]
    title = 'Collisions in Lns and LNS+CBS'
    x_axis = agents_list
    y_axis = [lns_collisions, cbs_collisions]
    x_axis_label = 'Number of agents'
    y_axis_label = 'collision counts'
    y_labels = ['LNS', "LNS + CBS"]
    plot_line_graphs(x_axis, y_axis, title, x_axis_label, y_axis_label, y_labels)


def CBS_lns_init_cols_exp(map_path, agent_path, solver_name, verbose = False, n_paths = 20, temp = 1):
    s = instance.instance(map_path, agent_path, solver_name, n_paths = n_paths)
    agents_list = range(3, s.num_agents, 5)
    lns_collisions = []
    cbs_collisions = []
    cbs_lns_collisions = []
    for num_agents in agents_list:
        with open(agent_path) as agent_file:
            head = [next(agent_file) for _ in range(num_agents+1)]
            with open('temp.txt', 'w') as temp:
                for line in head:
                    temp.write(line)
        s_temp = instance.instance(map_path, 'temp.txt', solver_name, n_paths = n_paths)
        t = PathTable(s_temp.num_of_rows, s_temp.num_of_cols, num_of_agents=s_temp.num_agents)

        solver = CBS.CBS(s_temp, verbose=verbose)
        _, cbs_col_count = solver.search()
        cbs_collisions += [cbs_col_count]

        solver_t, _ = solvers.generate_random_random_solution_iterative(s_temp, t)
        lns_num_cols = solver_t.num_collisions
        lns_collisions += [lns_num_cols]
        solution = torch.tensor([s_temp.agents[i + 1].path_id for i in range(s_temp.num_agents)])
        solution_one_hot = torch.nn.functional.one_hot(solution, n_paths)
        solver = CBS.CBS(s_temp,solution_one_hot, verbose=verbose)
        _, cbs_lns_col_count = solver.search()
        cbs_lns_collisions += [cbs_lns_col_count]
    title = 'Collisions in CBS and CBSw/LNSInit'
    x_axis = agents_list
    y_axis = [cbs_collisions,lns_collisions, cbs_lns_collisions]
    x_axis_label = 'Number of agents'
    y_axis_label = 'collision counts'
    y_labels = ["CBS", 'LNS', 'CBSw/LNSInit' ]
    plot_line_graphs(x_axis, y_axis, title, x_axis_label, y_axis_label, y_labels)


def CBS_lns_init_expanded_nodes_exp(map_path, agent_path, solver_name, verbose = False, n_paths = 20, temp = 1):
    s = instance.instance(map_path, agent_path, solver_name, n_paths = n_paths)
    agents_list = range(3, s.num_agents, 5)
    cbs_expanded = []
    cbs_lns_expanded = []
    for num_agents in agents_list:
        with open(agent_path) as agent_file:
            head = [next(agent_file) for _ in range(num_agents+1)]
            with open('temp.txt', 'w') as temp:
                for line in head:
                    temp.write(line)
        s_temp = instance.instance(map_path, 'temp.txt', solver_name, n_paths = n_paths)
        t = PathTable(s_temp.num_of_rows, s_temp.num_of_cols, num_of_agents=s_temp.num_agents)

        solver = CBS.CBS(s_temp, verbose=verbose)
        _, cbs_col_count = solver.search()
        cbs_expanded += [solver.expanded]

        solver_t, _ = solvers.generate_random_random_solution_iterative(s_temp, t)
        solution = torch.tensor([s_temp.agents[i + 1].path_id for i in range(s_temp.num_agents)])
        solution_one_hot = torch.nn.functional.one_hot(solution, n_paths)
        solver = CBS.CBS(s_temp,solution_one_hot, verbose=verbose)
        _, cbs_lns_col_count = solver.search()
        cbs_lns_expanded += [solver.expanded]

    title = 'Expanded nodes in CBS and CBSw/LNSInit'
    x_axis = agents_list
    y_axis = [cbs_expanded, cbs_lns_expanded]
    x_axis_label = 'Number of agents'
    y_axis_label = 'Expanded nodes'
    y_labels = ["CBS", 'CBSw/LNSInit' ]
    plot_line_graphs(x_axis, y_axis, title, x_axis_label, y_axis_label, y_labels)
def cbs_lns_exp(map_path, agent_path, solver_name, verbose = True, n_paths = 15, temp = 1):
    s = instance.instance(map_path, agent_path, solver_name, verbose, n_paths, temp)
    t = PathTable(s.num_of_rows, s.num_of_cols, num_of_agents=s.num_agents)
    solvers.generate_random_random_solution_iterative(s, t)
    solution = torch.tensor([s.agents[i+1].path_id for i in range(s.num_agents)])
    solution_one_hot = torch.nn.functional.one_hot(solution, n_paths)
    cbs = CBS.CBS(s, solution_one_hot)
    cbs.search()
