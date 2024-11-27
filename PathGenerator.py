import numpy as np
from itertools import islice
import aux
import networkx as nx


def softmax(x):
    return np.exp(x) / sum(np.exp(x))

def create_random_point_toward_end(instance, cur_point,end, temp):
    end_x = end[0]
    end_y = end[1]
    cur_x = cur_point[0]
    cur_y = cur_point[1]
    if int(np.random.randint(0,2)) %2 == 0: # x coord
        delta = end_x-cur_x
        probs = softmax(np.array([0,delta/temp])) # coord 0 means left, 1 means right
                                             # negative delta, high chance for left, else right
        choice = int(np.random.choice(2, 1, p=probs))
        choice = 2 * choice -1 # -1 means left, 1 means right
        return np.array([cur_x+choice, cur_y])
    else:
        delta = end_y - cur_y
        probs = softmax(np.array([0, delta/temp]))  # coord 0 means left, 1 means right
        # negative delta, high chance for left, else right
        choice = int(np.random.choice(2, 1, p=probs))
        choice = 2 * choice - 1  # -1 means left, 1 means right
        return np.array([cur_x, cur_y + choice])

def create_random_point_toward_end_with_obstacles(instance, cur_point,end, temp):
    map_graph = instance.map_graph

    # Get next available points from collision map
    next_points = np.array(list(list(n) for n in map_graph.neighbors(tuple(cur_point))))

    # Calculate deltas w.r.t origin point
    deltas = next_points - np.array(cur_point)

    # Calculate deltas from origin point to end point
    end_deltas = np.array(end)-np.array(cur_point)

    # Score points according to direction w.r.t end deltas
    point_scores = (deltas * end_deltas).sum(axis = 1)

    # Turn scores into probabilities
    probs = softmax(point_scores / temp)  # coord 0 means left, 1 means right

    choice_idx = int(np.random.choice(len(next_points), 1, p=probs))

    return np.array(next_points[choice_idx])


def create_random_point_toward_end_with_obstacles_old(instance,cur_point,end, temp):
    end_x = end[0]
    end_y = end[1]
    cur_x = cur_point[0]
    cur_y = cur_point[1]
    delta_x = end_x - cur_x
    delta_y = end_y - cur_y
    x_scores = [0,delta_x/temp] if delta_x > 0 else [-delta_x/temp, 0]
    y_scores = [0,delta_y/temp] if delta_y > 0 else [-delta_y/temp, 0]


    scores = np.array(x_scores + y_scores) # coord 0 means left, 1 means right

    x_shift_list = [-1, 1, 0, 0]
    y_shift_list = [0, 0, -1, 1]

    for idx, x_shift, y_shift in zip(range(4), x_shift_list, y_shift_list):
        if not instance.is_valid_move((cur_x, cur_y), (cur_x+x_shift, cur_y+y_shift)):
            scores[idx] = -np.inf
    probs = softmax(scores)  # coord 0 means left, 1 means right
    choice = int(np.random.choice(4, 1, p=probs))

    if choice < 2:
        choice = 2 * choice -1 # -1 means left, 1 means right
        return np.array([cur_x+choice, cur_y])
    else:
        choice -= 2
        choice = 2 * choice - 1  # -1 means left, 1 means right
        return np.array([cur_x, cur_y + choice])


def is_point_valid(point: np.array, num_of_rows, num_of_cols):
    return 0 <= point[0] < num_of_rows and 0 <= point[1] < num_of_cols


def create_random_step_path(instance, start, end, num_of_rows, num_of_cols, temp = 1):
    path = [start]

    cur_point = path[0]
    while not np.array_equal(cur_point,end) and aux.manhattan_dist(cur_point, end) > 1:
        new_point = create_random_point_toward_end(instance, cur_point,end, temp)
        if is_point_valid(new_point, num_of_rows, num_of_cols):# and instance.map[new_point[0], new_point[1]] != 1:
            cur_point = new_point
            path += [cur_point]
    path += [end]
    return np.array(path)


def k_shortest_paths(graph, source, target, k, weight=None):
    return list(
        islice(nx.shortest_simple_paths(graph, source, target, weight=weight), k)
    )
