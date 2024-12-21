import copy
from heapq import heappush, heappop
import networkx as nx
from intervaltree import Interval
from adaptive_lns_neighborhood_picker import AdaptiveLNSNeighborhoodPicker
from node import Node
from safe_interval_table import SafeIntervalTable


class LNS2:
    MAX_ITERATIONS = 1000
    NEIGHBORHOOD_SIZE = 10

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
        self.agent_start_goal_list = agent_start_goal_list
        self.hard_obstacles = hard_obstacles
        self.neighborhood_picker: AdaptiveLNSNeighborhoodPicker = AdaptiveLNSNeighborhoodPicker(self.NEIGHBORHOOD_SIZE, gamma=0.1, graph=graph)

        # Starting the algorithm
        solution = self.init_initial_solution()  # TODO change to prioritize planning? where in the article
                                                #  they say what to do?
        for i in range(self.MAX_ITERATIONS):
            print(f"iteration {i}")
            old_solution = copy.deepcopy(solution)
            neighborhood = self.neighborhood_picker.pick(solution)
            for agent_id in neighborhood:
                del solution[agent_id]
            new_solution = self.construct_new_solution(neighborhood, solution)  # TODO what do we do if there is no solution
            if new_solution is None:  # if we didn't find solution, try again
                solution = old_solution
                continue
            self.neighborhood_picker.update(self.num_of_colliding_pairs(old_solution) - self.num_of_colliding_pairs(new_solution))
            if self.num_of_colliding_pairs(new_solution) < self.num_of_colliding_pairs(old_solution):
                solution = new_solution
            else:
                solution = old_solution

    def init_initial_solution(self):
        solutions_dict = dict()
        for agent_id, start, goal in self.agent_start_goal_list:
            solutions_dict[agent_id] = self.shortest_solution(start, goal)
        return solutions_dict

    def shortest_solution(self, start, goal):
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
                if (neighbor, next_time) in self.hard_obstacles:
                    continue  # Skip hard obstacles
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
            existing_paths = {agent_id: solution[agent_id] for agent_id in solution}
            soft_obstacles: list[tuple[int, int]] = [(v, t) for _agent_id, path in existing_paths.items() for t, v in enumerate(path)]
            _, start, goal = self.agent_start_goal_list[agent_id - 1]
            soft_edge_obstacles = [(t, (path[t], path[t+1])) for _agent_id, path in existing_paths.items() for t in range(len(path) - 1)]
            agent_solution = self.sipp_pathfinding(start, goal, existing_paths, soft_obstacles, [], [], soft_edge_obstacles)
            if agent_solution is None:
                return None
            new_solution[agent_id] = agent_solution

        return new_solution

    def sipp_pathfinding(self, start, goal, existing_paths, soft_obstacles: list[tuple[int, int]], hard_obstacles: list[tuple[int, int]], hard_edge_obstacles: list[tuple[int, tuple[int, int]]], soft_edge_obstacles: list[tuple[int, tuple[int, int]]]):
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
        safe_intervals: SafeIntervalTable = SafeIntervalTable(list(self.graph.nodes), existing_paths, self.hard_obstacles)
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
                return n.path
            if n.vertex == goal and n.safe_interval.begin >= T:
                c_future = len([(g, t) for g, t in soft_obstacles if t > T])
                if c_future == 0:
                    return n.path
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
            low, high = safe_intervals[v][id_interval].begin, safe_intervals[v][id_interval].end
            low = self.earliest_arrival_time_at_v_within_range_without_colliding_with_edge_obstacles(low, high, (n, v), hard_edge_obstacles)
            if low is None:
                continue
            low_tag = self.earliest_arrival_time_at_v_within_range_without_colliding_with_edge_obstacles(low, high, (n, v), soft_edge_obstacles + hard_edge_obstacles)
            if low_tag is not None and low_tag > low:
                n1_g_val = low
                n1_c_val = n.c + (1 if (v, low) in soft_obstacles else 0) + (1 if (low, (n.vertex, v)) in soft_edge_obstacles else 0)
                n1 = Node(vertex=v, safe_interval=Interval(low, low_tag, "safe"), id=id_interval, is_goal=False,
                          g=n1_g_val, h=self.heuristic(v, goal), f=n.g + self.heuristic(v, goal),
                          c=n1_c_val, path=n.path + [v])
                self.insert_node(n1, open_list, closed_list)
                n2_g_val = low
                n2_c_val = n.c + (1 if (v, low) in soft_obstacles else 0) + (1 if (low, (n.vertex, v)) in soft_edge_obstacles else 0)
                n2 = Node(vertex=v, safe_interval=Interval(low_tag, high, "safe"), id=id_interval, is_goal=False,
                          g=n2_g_val, h=self.heuristic(v, goal), f=n.g + self.heuristic(v, goal),
                          c=n2_c_val, path=n.path + [v])
                self.insert_node(n2, open_list, closed_list)
            else:
                n3_g_val = low
                n3_c_val = n.c + (1 if (v, low) in soft_obstacles else 0) + (1 if (low, (n.vertex, v)) in soft_edge_obstacles else 0)
                n3 = Node(vertex=v, safe_interval=Interval(low, high, "safe"), id=id_interval, is_goal=False,
                          g=n3_g_val, h=self.heuristic(v, goal), f=n.g + self.heuristic(v, goal),
                          c=n3_c_val, path=n.path + [v])

                self.insert_node(n3, open_list, closed_list)  # TODO currently there is a bug when running the program, try to compare to the version where you don't have edge collisions

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
        Returns: The earliest arrival time at v within the range [low, high] without colliding with hard edge obstacles
        """
        n, v = edge
        times_of_edge_being_used = sorted([t for t, (v1, v2) in edge_obstacles if v1 == n and v2 == v])
        if not times_of_edge_being_used:
            return low
        for t in times_of_edge_being_used:
            if low > high:
                return None
            if low == t:
                low += 1
            else:
                return low

