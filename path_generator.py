import json
from itertools import islice
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, Future

import networkx as nx
import tqdm

from scenario_generator import load_map, Agent, load_agents


def _generate_agent_paths(
    map_graph: nx.Graph, agent: Agent, n_paths: int, index: int
) -> list[list[tuple[int, int]]]:
    start, end = agent
    paths = []

    loc1 = agent.start
    loc2 = agent.end
    dist = abs(loc1[0] - loc2[0]) + abs(loc1[1] - loc2[1])

    # pbar = tqdm.tqdm(
    #     total=n_paths, desc=f"Agent {index} {dist=}", unit="path", position=1
    # )
    for path in islice(nx.shortest_simple_paths(map_graph, start, end), n_paths):
        paths.append([(x, y) for x, y in path])
        # pbar.update(1)

    return paths


def generate_paths(
    map_graph: nx.Graph, agents: list[Agent], n_paths: int
) -> list[list[tuple[int, int]]]:
    """
    Generate multiple paths for a list of agents on a given map graph in parallel.
    Paths are generated in ascending length order.

    Parameters:
        map_graph (nx.Graph): The graph representing the map.
        agents (list[Agent]): A list of agents, where each agent is represented by a tuple (start, end) indicating the start and end nodes.
        n_paths (int): The number of paths to generate for each agent.

    Returns:
        list[list[tuple[int, int]]]: A list of paths for each agent. Each path is represented as a list of tuples, where each tuple is a coordinate (x, y).

    Raises:
        ValueError: If the provided map graph is not fully connected.
    """
    if not nx.is_connected(map_graph):
        raise ValueError("The map graph is not fully connected.")

    with ProcessPoolExecutor() as executor:
        futures: list[Future] = []
        for i, agent in enumerate(agents):
            agent_paths = executor.submit(
                _generate_agent_paths,
                map_graph,
                agent,
                n_paths,
                i,
            )
            futures.append(agent_paths)

        pbar = tqdm.tqdm(
            total=len(agents),
            desc=f"Generating agent paths",
            unit="agent",
        )
        paths = []
        for future in futures:
            paths.append(future.result())
            pbar.update(1)

    return paths


def main(map_file: Path, agents_file: Path, n_paths: int, output_file: Path) -> None:
    map_graph, _, _ = load_map(map_file)
    agents = load_agents(agents_file)
    agent_paths = generate_paths(map_graph, agents, n_paths)
    output_file.write_text(json.dumps(agent_paths))


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Generate random paths for agents")
    parser.add_argument("map_file", type=str, help="Path to the map file")
    parser.add_argument("agent_file", type=str, help="Path to the agent file")
    parser.add_argument("n_paths", type=int, help="Amount of paths for each agent")
    parser.add_argument(
        "output_file", type=str, help="Path to the output agent paths file"
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Overwrite the output file if it exists",
    )
    args = parser.parse_args()

    map_file = Path(args.map_file)
    agent_file = Path(args.agent_file)
    n_paths = int(args.n_paths)
    output_file = Path(args.output_file)
    force = args.force

    # validate map file exists
    if not map_file.exists():
        print(f"Map file {map_file} does not exist.", file=sys.stderr)
        exit(1)

    # validate agent file exists
    if not agent_file.exists():
        print(f"Agent file {agent_file} does not exist.", file=sys.stderr)
        exit(1)

    # valid n_paths
    if n_paths < 1:
        print("n_paths must be at least 1", file=sys.stderr)
        exit(1)

    # validate output file does not exist
    if not force and output_file.exists():
        print(f"Output file {output_file} already exists.", file=sys.stderr)
        exit(1)

    main(map_file, agent_file, n_paths, output_file)
