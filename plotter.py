import numpy as np
import matplotlib.pyplot as plt

'''def plot_path(map_size, path):
    x = np.arange(0, 1, 0.05)
    y = np.power(x, 2)

    fig = plt.figure()
    ax = fig.gca()
    ax.set_xticks(np.arange(1, map_size+1, 1))
    ax.set_yticks(np.arange(1, map_size+1, 1))
    plt.scatter(x, y)
    plt.grid()
    plt.show()'''

def plot_spiral():
    theta = np.radians(np.linspace(0, 360*2, 1000))
    max_r = int(np.ceil(max(theta)))
    r = theta
    x_2 = r * np.cos(theta)
    y_2 = r * np.sin(theta)
    plt.figure(figsize=[10, 10])
    plt.plot(x_2, y_2)
    plt.grid()
    xint = range(-max_r,max_r)
    plt.xticks(xint)
    plt.yticks(xint)
    plt.show()

def plot_path(path):
    '''

    Plots a path represented by a 2d numpy array

    Args:
        path : numpy array (nx2)
    '''

    x = path[:,0]
    y = path[:,1]
    plt.figure(figsize=[10, 10])
    plt.plot(x, y)
    plt.grid()

    min_x = int(min(x))
    max_x = int(max(x))
    min_y = int(min(y))
    max_y = int(max(y))

    xint = range(min_x,max_x+1)
    yint = range(min_y,max_y+1)

    plt.xticks(xint)
    plt.yticks(yint)
    plt.show()

def calculate_grid_edges(paths):
    paths = np.concatenate(paths)
    x = paths[:,0]
    y = paths[:,1]

    x_min = min(x)
    x_max = max(x)+1
    y_min = min(y)
    y_max = max(y) + 1

    x_range = range(x_min,x_max)
    y_range = range(y_min, y_max)
    return x_range,y_range
def plot_paths(paths, labels = []):
    '''

    Plots a path represented by a 2d numpy array

    Args:
        path : numpy array (nx2)
    '''
    plt.figure(figsize=[10, 10])
    plt.grid()

    for i, path in enumerate(paths):
        x = path[:,0]
        y = path[:,1]
        label = f'path {i+1}' if len(labels)==0 else labels[i]
        plt.plot(x, y, label = label,marker = 'D',markersize=20,lw = 6)

    (x_range, y_range) = calculate_grid_edges(paths)

    plt.xticks(x_range)
    plt.yticks(y_range)
    plt.legend()
    plt.show()


def plot_line_graph(x,y, title = "", label = ""):
    '''

    Plots a path represented by a 2d numpy array

    Args:
        path : numpy array (nx2)
    '''
    plt.figure(figsize=[10, 10])
    plt.grid()
    plt.title(title)

    plt.plot(x, y,label = label)

    #plt.legend()
    plt.show()