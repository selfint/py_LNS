from pathlib import Path

import networkx as nx
import numpy as np

import aux
from graphMethods import create_graph_from_map


def main(map_file: Path, num_agents: int, output_file: Path) -> None:
    file_string = map_file.read_text()
    lines = file_string.splitlines()

    num_of_rows = aux.extract_digits(lines[1])
    num_of_cols = aux.extract_digits(lines[2])

    map_array = np.array([[c != "." for c in l] for l in lines[4:]], dtype=np.int8)

    assert map_array.shape == (num_of_rows, num_of_cols)

    map_graph = create_graph_from_map(map_array, num_of_rows, num_of_cols)
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

    assert len(set(starts)) == num_agents
    assert len(set(ends)) == num_agents
    assert not any(starts[i] == ends[i] for i in range(num_agents))

    start_points = points[starts]
    end_points = points[ends]

    agent_lines = []
    for (sx, sy), (ex, ey) in zip(start_points, end_points):
        line = [0, map_file.name, num_of_rows, num_of_cols, sx, sy, ex, ey, 1.0]
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
