import colorsys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

from Agent import Agent
import instance
from PathTable import PathTable
import networkx as nx


def visualize(
    inst: instance.instance,
    path_table: PathTable,
    ax: plt.Axes,
    max_paths: int = 10,
    verbose: bool = False,
    fontsize: int = 8,
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
    for row, col, timestamp in path_table.table:
        for agent_id in path_table.table[(row, col, timestamp)]:
            grid_2d[agent_id - 1][row][col] = 1

    ax.imshow(inst.map, cmap=ListedColormap([(1, 1, 1, 0), (0, 0, 0, 1)]))

    # get agent collisions
    no_collisions: list[tuple[int, Agent]] = []
    collisions: list[tuple[int, Agent]] = []
    for agent_id, agent in inst.agents.items():
        if agent.path_id == -1:
            continue

        cmatrix = path_table.get_agent_collisions_for_paths(agent, inst.num_agents)

        # sanity check
        vertices, edges = path_table.get_agent_collisions_for_path(agent.id, agent.paths[agent.path_id])
        vertices_agents = set() if len(vertices) == 0 else set.union(*(ids for _, ids in vertices))
        edge_agents = set() if len(edges) == 0 else set.union(*(ids for _, ids in edges))
        unique_agents = len(set.union(vertices_agents, edge_agents))
        assert unique_agents == cmatrix[agent.path_id]

        does_collide = cmatrix[agent.path_id] > 0
        if does_collide:
            collisions.append((agent_id, agent))
        else:
            no_collisions.append((agent_id, agent))

    collisions = collisions[:max_paths]

    vertex_collisions = set()
    edge_collisions = set()

    relevant_ids = set(agent_id for agent_id, _ in collisions)
    for agent_id, agent in collisions:
        vertices, edges = path_table.get_agent_collisions_for_path(
            agent_id, agent.paths[agent.path_id]
        )

        for (x, y, _), others in vertices:
            if len((others & relevant_ids) - {agent_id}) > 0:
                vertex_collisions.add((x, y))

        for edge, others in edges:
            if len((others & relevant_ids) - {agent_id}) > 0:
                x, y, _, px, py, _ = edge
                x, y = ((x + px) / 2, (y + py) / 2)

                edge_collisions.add((x, y))

    for x, y in vertex_collisions:
        ax.text(y, x, "X", color="red", ha="center", va="center", fontsize=fontsize)

    for x, y in edge_collisions:
        ax.text(
            y, x, "X", color="blue", ha="center", va="center", fontsize=fontsize
        )

    if verbose:
        print("No collisions:", len(no_collisions))
        print("Collisions:", len(collisions))

    # plot agent paths
    legend_patches = []
    for agent_id, agent in collisions[:max_paths]:
        agent_grid = grid_2d[agent_id - 1]

        # generate random colors
        r, g, b = colorsys.hls_to_rgb(np.random.rand(), 0.5, 0.5)
        colors = [(1, 1, 1, 0), (r, g, b, 0.3)]
        ax.imshow(agent_grid, cmap=ListedColormap(colors), interpolation="nearest")

        path = agent.paths[agent.path_id]
        path_x = path[:, 1]
        path_y = path[:, 0]

        ax.plot(path_x, path_y, color=(r, g, b, 0.3), linewidth=1)

        # mark start and end
        y, x = agent.start
        ax.text(
            x, y, f"S{agent_id}", color="black", ha="center", va="center", fontsize=fontsize // 2
        )

        y, x = agent.end
        ax.text(
            x, y, f"E{agent_id}", color="black", ha="center", va="center", fontsize=fontsize // 2
        )

        # add legend entry
        legend_patches.append(
            mpatches.Patch(color=(r, g, b, 0.3), label=f"Agent {agent_id}")
        )

    # configure plot
    ax.set_title("Agent paths")

    # Positioning the legend outside the plot
    ax.legend(
        handles=legend_patches,
        loc="upper left",
        bbox_to_anchor=(1, 1),
        ncol=int(len(legend_patches) / 25) + 1,
        fontsize="small",
    )

    # Ensure the layout accommodates the legend on the right
    # plt.subplots_adjust(right=1.6)

    # move x-axis to the top
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()

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

    # Remove axis labels and ticks
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(axis="both", which="both", bottom=False, top=False, left=False, right=False)



def draw_graph_highlight_paths(graph, paths):
    # Create graph figure
    plt.figure(figsize=(6, 6))
    # Shape as grid
    pos = {(x, y): (y, -x) for x, y in graph.nodes()}
    nx.draw(graph, pos=pos,
            node_color='lightgreen',
            node_size=60)
    colors = ['r', 'g', 'b']
    # draw path in red
    for idx, path in enumerate(paths):
        color = colors[idx%len(colors)]
        #path = nx.shortest_path(graph, source=(1,1), target=(5,5))
        path_edges = list(zip(path, path[1:]))
        nx.draw_networkx_nodes(graph, pos, nodelist=path, node_color=color, node_size=60)
        nx.draw_networkx_edges(graph, pos, edgelist=path_edges, edge_color=color, width=5)
    plt.axis('equal')
    plt.show()
