import colorsys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

import instance
from PathTable import PathTable


def visualize(
    inst: instance.instance,
    path_table: PathTable,
    max_paths: int = 10,
    filename: str = None,
    verbose: bool = False,
) -> None:
    """
    Visualize agent paths on the grid.

    Note:
        Will not visualize agents that do not have a path (path_id == -1).

    Args:
        inst: instance object
        path_table: PathTable object
        max_plots: maximum number of agents to plot
        filename: filename to save the plot
    """

    # check bounds
    paths = [x for agent in inst.agents.values() for y in agent.paths for x in y]
    assert 0 <= np.min(paths) and np.max(paths) < min(
        inst.num_of_rows, inst.num_of_cols
    )

    # fill grid, ignore time dimension
    grid_2d = np.zeros((inst.num_agents, inst.num_of_rows, inst.num_of_cols))
    for row, col in path_table.table:
        for timestamp in path_table.table[(row, col)]:
            for agent_id in timestamp:
                grid_2d[agent_id - 1][row][col] = 1

    # plot map
    plt.imshow(inst.map, cmap=ListedColormap([(1, 1, 1, 0), (0, 0, 0, 1)]))

    # get agent collisions
    no_collisions = []
    collisions = []
    for agent_id, agent in inst.agents.items():
        cmatrix = path_table.get_agent_collisions_for_paths(agent, inst.num_agents)
        does_collide = cmatrix[agent.path_id] > 0
        if does_collide:
            collisions.append((agent_id, agent))
        else:
            no_collisions.append((agent_id, agent))

    if verbose:
        print("No collisions:", len(no_collisions))
        print("Collisions:", len(collisions))

    # plot agent paths
    legend_patches = []
    did_plot = 0
    for agent_id, agent in collisions:
        if agent.path_id == -1:
            continue

        did_plot += 1
        if did_plot >= max_paths:
            break

        agent_grid = grid_2d[agent_id - 1]

        # generate random colors
        r, g, b = colorsys.hls_to_rgb(np.random.rand(), 0.5, 0.5)
        colors = [(1, 1, 1, 0), (r, g, b, 0.3)]
        plt.imshow(agent_grid, cmap=ListedColormap(colors), interpolation="nearest")

        # mark start and end
        y, x = agent.start
        plt.text(
            x, y, f"S{agent_id}", color="black", ha="center", va="center", fontsize=4
        )

        y, x = agent.end
        plt.text(
            x, y, f"E{agent_id}", color="black", ha="center", va="center", fontsize=4
        )

        # add legend entry
        legend_patches.append(
            mpatches.Patch(color=(r, g, b, 0.3), label=f"Agent {agent_id}")
        )

    # configure plot
    plt.title("Agent paths")
    ax = plt.gca()

    ax.legend(
        handles=legend_patches,
        loc="upper center",  # Position legend at the top center
        bbox_to_anchor=(0.5, -0.1),  # Move legend below the plot
        ncol=3,  # Adjust number of columns to fit your needs
        frameon=False,
    )

    # # uncomment to move x labels to top
    # ax.xaxis.set_label_position("top")
    # ax.xaxis.tick_top()

    # Major ticks
    ax.set_xticks(np.arange(inst.num_of_cols))
    ax.set_yticks(np.arange(inst.num_of_rows))

    # Labels for major ticks
    ax.set_xticklabels(np.arange(inst.num_of_cols))
    ax.set_yticklabels(np.arange(inst.num_of_rows))

    # Minor ticks
    ax.set_xticks(np.arange(-0.5, inst.num_of_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, inst.num_of_rows, 1), minor=True)

    # Gridlines based on minor ticks
    ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.1)

    # Remove minor ticks
    ax.tick_params(which="minor", bottom=False, left=False)

    # clear axis labels
    plt.xticks([])
    plt.yticks([])

    plt.tight_layout()

    if filename is not None:
        plt.savefig(filename)
    plt.show()
