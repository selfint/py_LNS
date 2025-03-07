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

agent_alpha: float = 0.6
obstacle_alpha: float = 0.2


class FrameObjects(NamedTuple):
    agent_img: dict[int, mpimg.AxesImage]
    collision_texts: list[plt.Text]
    edge_collision_texts: list[plt.Text]
    arrows: dict[int, plt.Arrow]
    agent_colors: dict[int, tuple]


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

    max_timestamp = np.max([t for _, _, t in path_table.table]) + 1

    # fill grid at start_time
    grid_2d = np.zeros((inst.num_agents, inst.num_of_rows, inst.num_of_cols))
    for (x, y, t), agent_ids in path_table.table.items():
        if t == start_time:
            for agent_id in agent_ids:
                grid_2d[agent_id - 1][x][y] = 1

    ax.imshow(
        inst.map, cmap=ListedColormap([(1, 1, 1, 0), (0, 0, 0, obstacle_alpha)])
    )

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

    collisions = collisions[:max_paths]

    vertex_collisions = [set() for _ in range(max_timestamp)]
    edge_collisions = [set() for _ in range(max_timestamp)]

    relevant_ids = set(agent_id for agent_id, _ in collisions)
    for agent_id, agent in collisions:
        vertices, edges = path_table.get_agent_collisions_for_path(
            agent_id, agent.paths[agent.path_id]
        )

        for (x, y, t), others in vertices:
            if len((others & relevant_ids) - {agent_id}) > 0:
                vertex_collisions[t].add((x, y))

        for edge, others in edges:
            if len((others & relevant_ids) - {agent_id}) > 0:
                x, y, t, px, py, pt = edge
                x, y = ((x + px) / 2, (y + py) / 2)

                edge_collisions[t].add((x, y))
                edge_collisions[pt].add((x, y))

    collision_texts = [[] for _ in range(max_timestamp)]
    for timestamp, vertices in enumerate(vertex_collisions):
        for (x, y) in vertices:
            collision_texts[timestamp].append(
                ax.text(y, x, " ", color="red", ha="center", va="center", fontsize=16)
            )

    edge_collision_texts = [[] for _ in range(max_timestamp)]
    for timestamp, edges in enumerate(edge_collisions):
        for (x, y) in edges:
            edge_collision_texts[timestamp].append(
                ax.text(y, x, " ", color="blue", ha="center", va="center", fontsize=16)
            )

    if verbose:
        print("No collisions:", len(no_collisions))
        print("Collisions:", len(collisions))

    # plot agent paths
    legend_patches = []
    agent_img = {}
    arrows = {}
    agent_colors = {}
    for agent_id, agent in collisions:
        assert agent.path_id != -1, "got agent without active path"

        agent_grid = grid_2d[agent_id - 1]

        # generate random colors
        r, g, b = colorsys.hls_to_rgb(np.random.rand(), 0.5, 0.5)

        colors = [(1, 1, 1, 0), (r, g, b, agent_alpha)]
        agent_colors[agent_id] = (r, g, b, agent_alpha)
        agent_img[agent_id] = ax.imshow(agent_grid, cmap=ListedColormap(colors))

        # plot agent path
        path = agent.paths[agent.path_id]
        path_x = path[:, 1]
        path_y = path[:, 0]

        ax.plot(path_x, path_y, color=(r, g, b, agent_alpha), linewidth=1)
        ax.scatter(path_x[-1], path_y[-1], color=(r, g, b, agent_alpha), s=10)

        # add arrow
        y, x = agent.paths[agent.path_id][0]
        ny, nx = agent.paths[agent.path_id][1]
        dy = ny - y
        dx = nx - x
        arrow = mpatches.Arrow(
            x,
            y,
            dx,
            dy,
            color=(r, g, b, agent_alpha),
            width=0.5,
        )

        arrows[agent_id] = ax.add_patch(arrow)

        # add legend entry
        legend_patches.append(
            mpatches.Patch(color=(r, g, b, agent_alpha), label=f"Agent {agent_id}")
        )

    # configure plot
    ax.set_title("Agent paths")

    ax.legend(
        handles=legend_patches,
        loc="upper left",  # Position legend at the top center
        bbox_to_anchor=(1, 1),
        ncol=int(len(legend_patches) / 25) + 1,
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

    return FrameObjects(
        agent_img,
        collision_texts,
        edge_collision_texts,
        arrows,
        agent_colors,
    )


def update_frame(
    inst: instance.instance,
    path_table: PathTable,
    timestamp: int,
    frame_objects: FrameObjects,
    ax: plt.Axes,
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
    for (x, y, t), agent_ids in path_table.table.items():
        if t == timestamp:
            for agent_id in agent_ids:
                grid_2d[agent_id - 1][x][y] = 1

    # update agent paths
    for agent_id in frame_objects.agent_img:
        agent_grid = grid_2d[agent_id - 1]
        frame_objects.agent_img[agent_id].set_data(agent_grid)

    # update collision texts
    for t, texts in enumerate(frame_objects.collision_texts):
        for txt in texts:
            if timestamp >= t:
                txt.set_text("X")
            else:
                txt.set_text(" ")

    for t, texts in enumerate(frame_objects.edge_collision_texts):
        for txt in texts:
            if timestamp >= t:
                txt.set_text("X")
            else:
                txt.set_text(" ")

    # update arrows
    for agent_id in frame_objects.arrows:
        if frame_objects.arrows[agent_id] is not None:
            frame_objects.arrows[agent_id].remove()

        path = inst.agents[agent_id].paths[inst.agents[agent_id].path_id]
        if timestamp + 1 >= len(path):
            frame_objects.arrows[agent_id] = None
            continue

        y, x = path[timestamp]
        ny, nx = path[timestamp + 1]
        dy = ny - y
        dx = nx - x

        arrow = mpatches.Arrow(
            x,
            y,
            dx,
            dy,
            color=frame_objects.agent_colors[agent_id],
            width=0.5,
        )

        frame_objects.arrows[agent_id] = ax.add_patch(arrow)

    # unpack and return
    flat_collisions_texts = [
        txt for sublist in frame_objects.collision_texts for txt in sublist
    ]
    flat_edge_collision_texts = [
        txt for sublist in frame_objects.edge_collision_texts for txt in sublist
    ]
    return [
        *frame_objects.agent_img.values(),
        *flat_collisions_texts,
        *flat_edge_collision_texts,
        *[arrow for arrow in frame_objects.arrows.values() if arrow is not None],
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
    ), (
        0, np.min(paths), np.max(paths), (inst.num_of_rows, inst.num_of_cols)
    )

    fig.set_size_inches(16, 9)
    fig.subplots_adjust(left=0.05)  # Reduce the left margin

    # get max time
    max_time = np.max([t for x, y, t in path_table.table]) + 1

    frame_objects = setup(inst, path_table, ax, 0, max_paths, verbose)

    def update_frame_wrapper(timestamp):
        return update_frame(inst, path_table, timestamp, frame_objects, ax)

    ani = FuncAnimation(
        fig, update_frame_wrapper, frames=max_time, interval=100, blit=True
    )

    return ani
