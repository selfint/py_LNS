import copy
from heapq import heappush, heappop
import networkx as nx
from intervaltree import Interval
from adaptive_lns_neighborhood_picker import AdaptiveLNSNeighborhoodPicker
from node import Node
from safe_interval_table import SafeIntervalTable
from enum import Enum
import matplotlib.animation as animation

Strategy = Enum('strategy', 'PP AStar')

import matplotlib.pyplot as plt
import numpy as np


def animate_paths(grid_size, paths, index):
    """
    Animates the paths of agents on a grid.

    :param grid_size: Tuple of grid dimensions (rows, cols).
    :param paths: Dictionary where keys are agent IDs and values are lists of (x, y) tuples representing paths.
    """
    # Grid setup
    rows, cols = grid_size
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_xticks(np.arange(0, cols + 1, 1))
    ax.set_yticks(np.arange(0, rows + 1, 1))
    ax.grid(color='black', linestyle='--', linewidth=0.5)
    ax.set_aspect('equal', adjustable='box')
    ax.set_yticklabels(np.arange(rows, -1, -1))
    plt.gca().invert_yaxis()  # Match grid coordinates

    # Colors for agents
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta']
    max_steps = max(len(path) for path in paths.values())

    # Initialize plot elements for agents
    agent_lines = []
    agent_points = []
    for i, (agent, path) in enumerate(paths.items()):
        color = colors[i % len(colors)]
        line, = ax.plot([], [], color=color, lw=2, label=f'Agent {agent} ({color})')
        point, = ax.plot([], [], marker='o', color=color, markersize=8)
        agent_lines.append(line)
        agent_points.append(point)

    def init():
        """Initialize plot elements."""
        for line in agent_lines:
            line.set_data([], [])
        for point in agent_points:
            point.set_data([], [])
        return agent_lines + agent_points

    def update(frame):
        """Update plot elements at each frame."""
        for i, (agent, path) in enumerate(paths.items()):
            if frame < len(path):
                # Extract path up to the current frame
                x_coords = [p[1] for p in path[:frame + 1]]
                y_coords = [rows - p[0] for p in path[:frame + 1]]  # Flip y-coordinates for grid
                agent_lines[i].set_data(x_coords, y_coords)  # Update line
                agent_points[i].set_data(x_coords[-1:], y_coords[-1:])  # Update current position
        return agent_lines + agent_points

    ani = animation.FuncAnimation(
        fig, update, frames=max_steps, init_func=init, interval=500, blit=False, repeat=False
    )
    ani.save(f'agent_paths{index}.gif', writer='pillow')
    ax.legend(
        [line for line in agent_lines],  # Reference the line objects for each agent
        [f'Agent {i + 1} ({colors[i % len(colors)]})' for i in range(len(paths))],  # Create labels with color
        loc='upper left', fontsize='small', title="Agent Colors"
    )
    # Add legend and show plot
    ax.legend()
    plt.title("Agent Path Progression")
    plt.xlabel("Columns")
    plt.ylabel("Rows")
    plt.show()


class LNS2:
    MAX_ITERATIONS = 10000000  # TODO refactor for time limit of 1 minute
    NEIGHBORHOOD_SIZE = 8

    def __init__(self, graph: nx.Graph, agent_start_goal_list: list[(int, int, int)],
                 hard_obstacles: list[tuple[int, int]]):
        """
        Initialize the LNS solver.
        :param graph: The graph on which the agents are moving.
        :param agent_start_goal_list: A list of tuples, where each tuple contains the agent id, start location,
        and goal location.
        :param hard_obstacles: A list of tuples of hard obstacles in the format of (vertex, timestamp)
        meaning there is a hard obstacle in `vertex` at `timestamp`.
        :param soft_obstacles: A list of tuples of soft obstacles in the format of (vertex, timestamp)
        meaning there is a soft obstacle in `vertex` at `timestamp`.
        """
        # Init vars
        self.graph = graph
        self.agent_start_goal_dict = {agent_id: (start, goal) for agent_id, start, goal in agent_start_goal_list}
        self.hard_obstacles = hard_obstacles
        self.neighborhood_picker: AdaptiveLNSNeighborhoodPicker = AdaptiveLNSNeighborhoodPicker(self.NEIGHBORHOOD_SIZE, gamma=0.1, graph=graph)

        # Starting the algorithm
        solution = self.init_initial_solution()
        if self.num_of_colliding_pairs(solution) == 0:
            print("found solution with 0 collisions at initial solution (iteration 0)")
            animate_paths((32, 32), solution, 0)
            return
        for i in range(self.MAX_ITERATIONS):
            old_solution = copy.copy(solution)
            neighborhood = self.neighborhood_picker.pick(solution)
            print(f"picked {neighborhood}")
            for agent_id in neighborhood:
                del solution[agent_id]
            new_solution = self.construct_new_solution(neighborhood, solution)
            if new_solution is None:  # if we didn't find solution, try again
                solution = old_solution
                continue
            old_collisions = self.num_of_colliding_pairs(old_solution)
            new_collisions = self.num_of_colliding_pairs(new_solution)
            if new_collisions == 0:
                print(f"found solution with 0 collisions {new_collisions} after {i} iterations")
                animate_paths((32, 32), new_solution, i)
                return
            self.neighborhood_picker.update(old_collisions - new_collisions)
            if new_collisions < old_collisions:
                solution = new_solution
                print(f"found better solutions with num of colliding pairs {new_collisions} collsions in iterataion {i}\n\n{solution}")
                animate_paths((32, 32), solution, i)
            else:
                if i % 1000 == 0:
                    print(f"iteration {i}")
                solution = old_solution

    def find_non_colliding_solution_with_a_star(self):
        hard_obstacles = []
        edge_obstacles = []
        agents_paths = dict()
        for agent_id in self.agent_start_goal_dict.keys():
            start, goal = self.agent_start_goal_dict[agent_id]
            agents_paths[agent_id] = self.shortest_solution(start, goal, hard_obstacles, edge_obstacles)
            edge_obstacles.extend([((agents_paths[agent_id][t], t), (agents_paths[agent_id][t+1], t+1)) for t in range(len(agents_paths[agent_id]) - 1)])
            hard_obstacles.extend([(v, t) for t, v in enumerate(agents_paths[agent_id])])
        print(f"found solution with {self.num_of_colliding_pairs(agents_paths)} collisions")
        animate_paths((32, 32), agents_paths, 99999)


    def init_initial_solution(self, strategy=Strategy.PP):
        if strategy == Strategy.PP:
            return self.construct_new_solution(self.agent_start_goal_dict.keys(), dict())
        elif strategy == Strategy.AStar:
            solutions_dict = dict()
            for agent_id, start, goal in self.agent_start_goal_list:
                solutions_dict[agent_id] = self.shortest_solution(start, goal)
            return solutions_dict

    def shortest_solution(self, start, goal, hard_obstacles=None, edge_obstacles=None):
        """
        Find the shortest path from start to goal avoiding hard obstacles.
        This is basically an A* implementation.
        Unfortunately, we can't just run NetworkX shortest path algorithm
        because we need to avoid hard obstacles.
        :param start: The starting vertex.
        :param goal: The goal vertex.
        :return: A list of vertices representing the shortest path.
        """
        open_set = []
        heappush(open_set, (0, start, 0, []))  # (priority, current_vertex, time, path)

        visited = set()

        while open_set:
            _, current_vertex, current_time, path = heappop(open_set)
            # Add the current state to visited
            if (current_vertex, current_time) in visited:
                continue
            visited.add((current_vertex, current_time))
            # Add the current vertex to the path
            new_path = path + [current_vertex]
            # Check if the goal is reached
            if current_vertex == goal:
                return new_path
            # Explore neighbors
            for neighbor in self.graph.neighbors(current_vertex):
                next_time = current_time + 1
                if (neighbor, next_time) in hard_obstacles:
                    continue  # Skip hard obstacles
                if ((neighbor, current_time), (current_vertex, current_time+1)) in edge_obstacles:
                    continue  # Skip edge obstacles
                if (neighbor, next_time) not in visited:
                    cost = len(new_path) + self.heuristic(neighbor, goal)
                    heappush(open_set, (cost, neighbor, next_time, new_path))

        # Return an empty list if no path is found
        return []

    def construct_new_solution(self, neighborhood: list[int], solution: dict[int, list[int]]):
        """
        Constructs a new solution for the agents in the given neighborhood
        by replanning their paths using SIPP while avoiding collisions.
        :param neighborhood: List of agent IDs to be replanned.
        :param solution: A dictionary with the current solution, where the key is agent_id and value is their current path.
        :return: New solution dictionary with updated paths for agents in the neighborhood.
        """
        new_solution = dict(solution)  # Start with the existing solution
        # Existing paths are used as soft obstacles
        # For each agent in the neighborhood, we replan their path using sipp
        for agent_id in neighborhood:
            existing_paths = {agent_id: new_solution[agent_id] for agent_id in new_solution}
            soft_obstacles: list[tuple[int, int]] = [(v, t) for _agent_id, path in existing_paths.items() for t, v in enumerate(path)]
            start, goal = self.agent_start_goal_dict[agent_id]
            soft_edge_obstacles = [(t, (path[t], path[t+1])) for _agent_id, path in existing_paths.items() for t in range(len(path) - 1)]
            agent_solution = self.sipp_pathfinding(start, goal, soft_obstacles, [], [], soft_edge_obstacles)
            if agent_solution is None:
                return None
            new_solution[agent_id] = agent_solution

        return new_solution

    def sipp_pathfinding(self, start, goal, soft_obstacles: list[tuple[int, int]], hard_obstacles: list[tuple[int, int]], hard_edge_obstacles: list[tuple[int, tuple[int, int]]], soft_edge_obstacles: list[tuple[int, tuple[int, int]]]):
        """
        Perform SIPP for an agent from start to goal.
        This is a by-the-book implementation as shown in the paper MAPF-LNS2.
        :param start: Starting vertex for the agent.
        :param goal: Goal vertex for the agent.
        :param existing_paths: A dictionary of existing paths of all agents to avoid soft obstacles.
        :param soft_obstacles: A list of tuples of soft obstacles in the format of (vertex, timestamp)
        meaning there is a soft obstacle in `vertex` at `timestamp`.
        :param hard_obstacles: A list of tuples of hard obstacles in the format of (vertex, timestamp)
        meaning there is a hard obstacle in `vertex` at `timestamp`.
        :param soft_edge_obstacles: A list of tuples of soft edge obstacles in the format of (timestamp, (v1, v2))
        :param hard_edge_obstacles: A list of tuples of hard edge obstacles in the format of (timestamp, (v1, v2))
        :return: A list of vertices representing the shortest path found using SIPP.
        """
        safe_intervals: SafeIntervalTable = SafeIntervalTable(list(self.graph.nodes), soft_obstacles, self.hard_obstacles)
        root: Node = Node(vertex=start, safe_interval=safe_intervals[start][0], id=0, is_goal=False,
                    g=0,
                    h=self.heuristic(start, goal),
                    f=self.heuristic(start, goal) + 0,
                    c=0, path=[start])
        T: int = 0
        if goal in [v_t[0] for v_t in self.hard_obstacles]:
            T = max([v_t[1] for v_t in self.hard_obstacles if v_t[0] == goal])
        open_list: list[Node] = []
        closed_list: list[Node] = []
        heappush(open_list, root)
        while open_list:
            n: Node = heappop(open_list)
            if n.is_goal:
                return self.extract_path(n)
            if n.vertex == goal and n.safe_interval.begin >= T:
                c_future = len([(g, t) for g, t in soft_obstacles if t > n.safe_interval.begin and g == goal])
                if c_future == 0:
                    return self.extract_path(n)
                n_tag: Node = copy.deepcopy(n)
                n_tag.is_goal = True
                n_tag.c = n_tag.c + c_future
                self.insert_node(n_tag, open_list, closed_list)
            self.expand_node(n, open_list, closed_list, safe_intervals, soft_obstacles, hard_edge_obstacles, soft_edge_obstacles, goal)
            heappush(closed_list, n)
        return None

    def heuristic(self, v1, v2):
        """
        Computes the heuristic value between two vertices, which is the shortest path length.
        """
        return nx.shortest_path_length(self.graph, v1, v2)
    
    @staticmethod
    def num_of_colliding_pairs(solution: dict[int, list[int]]) -> int:
        """
        Count the number of collisions in new_solution.
        new_solution is a dict of agent, list of vertices representing the path.
        We return the number of colliding pairs of agents
        Args
            new_solution: dict[int, List[int]]: A dictionary of agent_id and their path.
        Returns:
        """
        collisions = 0
        for agent_id, path in solution.items():
            for agent_id2, path2 in solution.items():
                if agent_id == agent_id2:
                    continue
                for i in range(min(len(path), len(path2))):
                    if path[i] == path2[i]:
                        collisions += 1
                        break
                # detect edge collisions:
                for i in range(min(len(path), len(path2)) - 1):
                    if path[i] == path2[i+1] and path[i+1] == path2[i]:
                        collisions += 1
                        break
        return collisions / 2

    def insert_node(self, n: Node, open_list: list[Node], closed_list: list[Node]):
        # we assume c f g h values are computed for n
        open_and_closed = open_list + closed_list
        N = [q for q in open_and_closed if q.vertex == n.vertex and q.id == n.id and q.is_goal == n.is_goal]
        for q in N:
            if q.safe_interval.begin <= n.safe_interval.begin and q.c <= n.c:
                return
            elif n.safe_interval.begin <= q.safe_interval.begin and n.c <= q.c:
                try:
                    open_list.remove(q)
                    closed_list.remove(q)
                except ValueError:
                    pass  # q is in one of those lists
            elif n.safe_interval.begin < q.safe_interval.end and q.safe_interval.begin < n.safe_interval.end:
                if n.safe_interval.begin < q.safe_interval.begin:
                    n.safe_interval = Interval(n.safe_interval.begin, q.safe_interval.begin, "safe")
                else:
                    q.safe_interval = Interval(q.safe_interval.begin, n.safe_interval.begin, "safe")
        heappush(open_list, n)

    def expand_node(self, n: Node, open_list: list[Node], closed_list: list[Node], safe_intervals: SafeIntervalTable,
                    soft_obstacles: list[tuple[int, int]],
                    hard_edge_obstacles: list[tuple[int, tuple[int, int]]],
                    soft_edge_obstacles: list[tuple[int, tuple[int, int]]],
                    goal):
        """
        Expand a node n by generating its neighbors and inserting them into the open list
        """
        I: list[(int, int)] = []
        for neighbor in self.graph.neighbors(n.vertex):
            I.extend([(neighbor, id_interval) for id_interval in range(len(safe_intervals[neighbor]))
                      if safe_intervals[neighbor][id_interval].overlaps(Interval(n.safe_interval.begin + 1, n.safe_interval.end + 1))])

        k = 0  # Testing that the if statement runs at most once
        for id_interval in range(len(safe_intervals[n.vertex])):
            if safe_intervals[n.vertex][id_interval].begin == n.safe_interval.end:
                I.append((n.vertex, id_interval))
                k = 1
        assert k <= 1  # The if statement runs at most once

        for (v, id_interval) in I:
            original_low, high = safe_intervals[v][id_interval].begin, safe_intervals[v][id_interval].end
            low = max(n.safe_interval.begin + 1, self.earliest_arrival_time_at_v_within_range_without_colliding_with_edge_obstacles(original_low, high, (n, v), hard_edge_obstacles))
            if low is None:
                continue
            low_tag = max(n.safe_interval.begin + 1, self.earliest_arrival_time_at_v_within_range_without_colliding_with_edge_obstacles(original_low, high, (n, v), soft_edge_obstacles + hard_edge_obstacles))
            if low_tag > low:
                n1_g_val = low
                n1_c_val = n.c + (1 if self.safe_interval_contains_soft_obstacles(v, low, low_tag, soft_obstacles) else 0) + (1 if (low - 1, (v, n.vertex)) in soft_edge_obstacles else 0)
                n1 = Node(vertex=v, safe_interval=Interval(low, low_tag, "safe"), id=id_interval, is_goal=False,
                          g=n1_g_val, h=self.heuristic(v, goal), f=n.g + self.heuristic(v, goal),
                          c=n1_c_val, path=n.path + [v], who_expanded_me=n)
                self.insert_node(n1, open_list, closed_list)
                n2_g_val = low_tag
                n2_c_val = n.c + (1 if self.safe_interval_contains_soft_obstacles(v, low_tag, high, soft_obstacles) else 0) + (1 if (low_tag-1, (v, n.vertex)) in soft_edge_obstacles else 0)
                n2 = Node(vertex=v, safe_interval=Interval(low_tag, high, "safe"), id=id_interval, is_goal=False,
                          g=n2_g_val, h=self.heuristic(v, goal), f=n.g + self.heuristic(v, goal),
                          c=n2_c_val, path=n.path + [v], who_expanded_me=n)
                self.insert_node(n2, open_list, closed_list)
            else:
                n3_g_val = low
                n3_c_val = n.c + (1 if self.safe_interval_contains_soft_obstacles(v, low, high, soft_obstacles) else 0) + (1 if (low-1, (v, n.vertex)) in soft_edge_obstacles else 0)
                n3 = Node(vertex=v, safe_interval=Interval(low, high, "safe"), id=id_interval, is_goal=False,
                          g=n3_g_val, h=self.heuristic(v, goal), f=n.g + self.heuristic(v, goal),
                          c=n3_c_val, path=n.path + [v], who_expanded_me=n)

                self.insert_node(n3, open_list, closed_list)

    def earliest_arrival_time_at_v_within_range_without_colliding_with_edge_obstacles(self, low, high, edge,
                                                                                           edge_obstacles):
        """
        I know the name is long but this word by word from the paper
        Args:
            low: lower bound
            high: upper bound
            edge: tuple of (v1, v2) representing the edge
            edge_obstacles: list of hard edge obstacles in the format of (timestamp, (v1, v2))
            meaning there is a hard edge obstacle that goes from v1 to v2 starting from `timestamp` at v1
            and ending at `timestamp` + 1 at v2
        Returns: The earliest arrival time at v within the range [low, high] without colliding with edge obstacles
        """
        times_of_edge_being_used = [t for t, e in edge_obstacles if e == edge and low <= t < high]
        if not times_of_edge_being_used:
            return low
        return max(times_of_edge_being_used) + 1

    def safe_interval_contains_soft_obstacles(self, vertex, low, high, soft_obstacles):
        times_of_obstacles_in_t = [t for v, t in soft_obstacles if v == vertex]
        if not times_of_obstacles_in_t:
            return False
        max_timestamp = max(times_of_obstacles_in_t)
        for i in range(int(low), min(high-1, max_timestamp) + 1):
            if (vertex, i) in soft_obstacles:
                return True
        return False

    def extract_path(self, n):
        path = [n.vertex]
        while n.who_expanded_me is not None:
            for i in range(n.safe_interval.begin - n.who_expanded_me.safe_interval.begin):
                path.insert(0, n.who_expanded_me.vertex)
            n = n.who_expanded_me
        return path


# for scen-randon we got
# iteration 161
# found better solutions with self.num_of_colliding_pairs 1331.0 collsions