from typing import NamedTuple
from pathlib import Path

import networkx as nx
import numpy as np

import aux


class Point(NamedTuple):
    x: int
    y: int


class Agent(NamedTuple):
    start: Point
    end: Point


class Map(NamedTuple):
    graph: nx.Graph
    n_rows: int
    n_cols: int


def load_map(map_file: Path) -> Map:
    file_string = map_file.read_text()
    lines = file_string.splitlines()

    n_rows = aux.extract_digits(lines[1])
    n_cols = aux.extract_digits(lines[2])

    map_array = np.array([[c != "." for c in l] for l in lines[4:]], dtype=np.int8)
    assert map_array.shape == (n_rows, n_cols)

    map_graph = nx.grid_2d_graph(n_rows, n_cols)
    nodes = [tuple(loc) for loc in np.argwhere(map_array > 0)]
    map_graph.remove_nodes_from(nodes)

    return Map(map_graph, n_rows, n_cols)


def load_agents(agents_file: Path) -> list[Agent]:
    lines = agents_file.read_text().splitlines()

    agents = []
    for line in lines[1:]:
        sx, sy, ex, ey = line.split("\t")[4:8]

        agent = Agent(
            Point(int(sx), int(sy)),
            Point(int(ex), int(ey)),
        )
        agents.append(agent)

    return agents


def generate_scenario(map_graph: nx.Graph, n_agents: int) -> list[Agent]:
    """
    Generates a list of agents with start and end points on a given map graph.
    Parameters:
        map_graph (nx.Graph): The graph representing the map.
        n_agents (int): The number of agents to generate.

    Returns:
        list[Agent]: A list of Agent objects with start and end points.

    Raises:
        ValueError: If the generated graph is not fully connected.

    Notes:
        - The start and end points for each agent are randomly selected from the nodes of the graph.
        - The start and end points for each agent are guaranteed to be different.
        - Uses numpy random
    """

    points = np.array(map_graph.nodes)

    if not nx.is_connected(map_graph):
        raise ValueError("The generated graph is not fully connected.")

    starts = np.random.permutation(len(points))[:n_agents]

    ends = []
    for start in starts:
        # not start
        end = np.random.choice([i for i in range(len(points)) if i != start])
        ends.append(end)
    ends = np.array(ends)

    assert not any(starts[i] == ends[i] for i in range(n_agents))

    start_points = points[starts]
    end_points = points[ends]

    agents = []
    for (sx, sy), (ex, ey) in zip(start_points, end_points):
        agent = Agent(Point(int(sx), int(sy)), Point(int(ex), int(ey)))
        agents.append(agent)

    return agents


def main(map_file: Path, num_agents: int, output_file: Path) -> None:
    map_graph, n_rows, n_cols = load_map(map_file)
    points = np.array(map_graph.nodes)

    if not nx.is_connected(map_graph):
        raise ValueError("The generated graph is not fully connected.")

    starts = np.random.permutation(len(points))[:num_agents]

    ends = []
    for start in starts:
        # not start
        end = np.random.choice([i for i in range(len(points)) if i != start])
        ends.append(end)
    ends = np.array(ends)

    assert not any(starts[i] == ends[i] for i in range(num_agents))

    start_points = points[starts]
    end_points = points[ends]

    agent_lines = []
    for (sx, sy), (ex, ey) in zip(start_points, end_points):
        # assert start and end are points in graph
        assert (sx, sy) in map_graph.nodes
        assert (ex, ey) in map_graph.nodes

        line = [0, map_file.name, n_rows, n_cols, sx, sy, ex, ey, 1.0]
        agent_lines.append(line)

    output_text = "\n".join(
        ["version 1", *["\t".join(map(str, l)) for l in agent_lines]]
    )

    output_file.write_text(output_text)


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Generate a scenario file for a given map, with a set amount of agents."
    )
    parser.add_argument("map_file", type=str, help="Path to the map file")
    parser.add_argument("num_agents", type=int, help="Number of agents to generate")
    parser.add_argument(
        "output_file", type=str, help="Path to the output scenario file"
    )
    args = parser.parse_args()

    map_file = Path(args.map_file)
    num_agents = args.num_agents
    output_file = Path(args.output_file)

    # validate map file exists
    if not map_file.exists():
        print(f"Map file {map_file} does not exist.", file=sys.stderr)
        exit(1)

    # validate number of agents
    if num_agents < 1:
        print(
            f"Number of agents must be greater than 0. Got: {num_agents}",
            file=sys.stderr,
        )
        exit(1)

    # validate output file does not exist
    if output_file.exists():
        print(f"Output file {output_file} already exists.", file=sys.stderr)
        exit(1)

    main(map_file, num_agents, output_file)
