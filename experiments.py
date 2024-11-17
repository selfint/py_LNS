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

def destroy_method_ablation_exp(map_path, agent_path, solver_name, verbose = True, n_paths = 3, temp = 1):
    #open('test.csv', "a")
    ds_list = ['cc', 'w-random', 'random']#range(1, 12)
    collision_counts = []
    x_axis = []
    for ds in ds_list:
        s = instance.instance(map_path, agent_path, solver_name, verbose, n_paths, temp)
        t = PathTable(s.num_of_rows, s.num_of_cols)
        solvers.random_initial_solution(s, t)
        solver = solvers.IterativeRandomLNS(s, t, 20, destroy_method_name=ds, num_iterations= 3500, low_level_solver_name=solver_name)
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
    label = 'num_of_cols in random PP LNS (varying group sizes and solvers, 1000 iteration)'
    x_axis_label = 'iterations'
    y_axis_label = 'collision counts'
    y_labels = [f'subset size = {size}, solver = {solver}' for size, solver in variants]
    plot_line_graphs(x_axis, collision_counts, label, x_axis_label,y_axis_label, y_labels)