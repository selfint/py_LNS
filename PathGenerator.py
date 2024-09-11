import numpy as np
def softmax(x):
    return np.exp(x) / sum(np.exp(x))

def create_random_point_toward_end(cur_point,end, temp):
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


def is_point_valid(point: np.array, num_of_rows, num_of_cols):
    return 0 <= point[0] <= num_of_rows and 0 <= point[1] <= num_of_cols


def create_random_step_path(start, end, num_of_rows, num_of_cols, temp = 1):
    path = [start]

    cur_point = path[0]
    while not np.array_equal(cur_point,end):
        new_point = create_random_point_toward_end(cur_point,end, temp)
        if is_point_valid(new_point, num_of_rows, num_of_cols):
            cur_point = new_point
            path += [cur_point]
    return np.array(path)