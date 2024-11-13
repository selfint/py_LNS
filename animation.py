from typing import NamedTuple
import colorsys

import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from matplotlib.animation import FuncAnimation


import instance
from PathTable import PathTable


class FrameObjects(NamedTuple):
    bg: mpimg.AxesImage
    agent_img: dict[int, mpimg.AxesImage]
    agent_txt_start: dict[int, plt.Text]
    agent_txt_end: dict[int, plt.Text]
    collision_texts: list[plt.Text]


def setup(
    inst: instance.instance,
    path_table: PathTable,
    ax: plt.Axes,
    start_time: int = 0,
    max_paths: int = 10,
    verbose: bool = False,
) -> FrameObjects:
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

    max_timestamp = np.max([len(path_table.table[loc]) for loc in path_table.table])

    # fill grid at start_time
    grid_2d = np.zeros((inst.num_agents, inst.num_of_rows, inst.num_of_cols))
    for row, col in path_table.table:
        if len(path_table.table[(row, col)]) > start_time:
            for agent_id in path_table.table[(row, col)][start_time]:
                grid_2d[agent_id - 1][row][col] = 1

    bg = ax.imshow(inst.map, cmap=ListedColormap([(1, 1, 1, 0), (0, 0, 0, 1)]))

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

    collisions_2d = np.zeros((inst.num_of_rows, inst.num_of_cols))
    relevant_ids = set(agent_id for agent_id, _ in collisions[:max_paths])
    for agent_id, agent in collisions[:max_paths]:
        for timestamp, (row, col) in enumerate(agent.paths[agent.path_id]):
            others = set(path_table.table[(row, col)][timestamp])

            # check if there are other agents in the same cell
            if len(others & relevant_ids) > 1:
                collisions_2d[row][col] = timestamp

    collision_texts = [[] for _ in range(max_timestamp)]
    for x, y in zip(*np.where(collisions_2d)):
        timestamp = int(collisions_2d[x][y])
        collision_texts[timestamp].append(
            ax.text(y, x, " ", color="red", ha="center", va="center", fontsize=8)
        )

    if verbose:
        print("No collisions:", len(no_collisions))
        print("Collisions:", len(collisions))

    # plot agent paths
    legend_patches = []
    did_plot = 0
    agent_img = {}
    agent_txt_start = {}
    agent_txt_end = {}
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

        agent_img[agent_id] = ax.imshow(
            agent_grid, cmap=ListedColormap(colors), interpolation="nearest"
        )

        # mark start and end
        y, x = agent.start
        agent_txt_start[agent_id] = ax.text(
            x, y, f"S{agent_id}", color="black", ha="center", va="center", fontsize=4
        )

        y, x = agent.end
        agent_txt_end[agent_id] = ax.text(
            x, y, f"E{agent_id}", color="black", ha="center", va="center", fontsize=4
        )

        # add legend entry
        legend_patches.append(
            mpatches.Patch(color=(r, g, b, 0.3), label=f"Agent {agent_id}")
        )

    # configure plot
    ax.set_title("Agent paths")

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
    ax.set_xticks([])
    ax.set_yticks([])

    return FrameObjects(bg, agent_img, agent_txt_start, agent_txt_end, collision_texts)


def update_frame(
    inst: instance.instance,
    path_table: PathTable,
    timestamp: int,
    frame_objects: FrameObjects,
) -> list:
    """
    Update the frame objects with the current timestamp.

    Args:
        inst: instance object
        path_table: PathTable object
        timestamp: timestamp to visualize
        frame_objects: FrameObjects object
    """

    # fill grid, at specific timestamp
    grid_2d = np.zeros((inst.num_agents, inst.num_of_rows, inst.num_of_cols))
    for row, col in path_table.table:
        if len(path_table.table[(row, col)]) > timestamp:
            for agent_id in path_table.table[(row, col)][timestamp]:
                grid_2d[agent_id - 1][row][col] = 1

    frame_objects.bg.set_data(inst.map)

    # update agent paths
    for agent_id in frame_objects.agent_img:
        frame_objects.agent_img[agent_id].set_data(grid_2d[agent_id - 1])
        frame_objects.agent_txt_start[agent_id].set_text(f"S{agent_id}")
        frame_objects.agent_txt_end[agent_id].set_text(f"E{agent_id}")

    # update collision texts
    for texts in frame_objects.collision_texts[: timestamp + 1]:
        for txt in texts:
            txt.set_text("X")
    for texts in frame_objects.collision_texts[timestamp + 1 :]:
        for txt in texts:
            txt.set_text(" ")

    # unpack and return
    flat_collisions_texts = [
        txt for sublist in frame_objects.collision_texts for txt in sublist
    ]
    return [
        frame_objects.bg,
        *frame_objects.agent_img.values(),
        *frame_objects.agent_txt_start.values(),
        *frame_objects.agent_txt_end.values(),
        *flat_collisions_texts,
    ]


def animate(
    inst: instance.instance,
    path_table: PathTable,
    fig: plt.Figure,
    ax: plt.Axes,
    max_paths: int = 10,
    verbose: bool = False,
) -> FuncAnimation:
    """
    Animate agent paths on the grid.

    Note:
        Will not render if the returned FuncAnimation object is not saved.
        See example below.

    Example:
        >>> ani = animation.animate(inst, path_table, fig, ax, max_paths=100, verbose=verbose)
        >>> # save as gif
        >>> ani.save("animation.gif", writer=PillowWriter(fps=2))
        >>> # save as mp4 (requires ffmpeg)
        >>> ani.save("animation.mp4", writer="ffmpeg", fps=2)
    """

    # check bounds
    paths = [x for agent in inst.agents.values() for y in agent.paths for x in y]
    assert 0 <= np.min(paths) and np.max(paths) < min(
        inst.num_of_rows, inst.num_of_cols
    )

    # get max time
    max_time = np.max([len(path_table.table[loc]) for loc in path_table.table])

    frame_objects = setup(inst, path_table, ax, 0, max_paths, verbose)

    def update_frame_wrapper(timestamp):
        return update_frame(inst, path_table, timestamp, frame_objects)

    ani = FuncAnimation(
        fig, update_frame_wrapper, frames=max_time, interval=100, blit=True
    )

    return ani
