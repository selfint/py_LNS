from heapq import heappush, heappop
from typing import Tuple, List
import random
import networkx as nx
from lns2.random_neighborhood_picker import RandomNeighborhoodPicker


class LNS2:
    MAX_ITERATIONS = 1000
    NEIGHBORHOOD_SIZE = 10
    def __init__(self, graph: nx.Graph, agent_start_goal_list: Tuple[int, int, int],
                 hard_obstacles: Tuple[int, int],
                 soft_obstacles: Tuple[int, int]):
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
        self.soft_obstacles = soft_obstacles
        self.neighborhood_picker = RandomNeighborhoodPicker(self.NEIGHBORHOOD_SIZE)

        # Starting the algorithm
        solution = self.init_initial_solution()
        for _ in range(self.MAX_ITERATIONS):
            neighborhood = self.neighborhood_picker.pick(solution)
            for agent_id in neighborhood:
                del solution[agent_id]
            new_solution = self.construct_new_solution(neighborhood, solution)
            if new_solution < solution:
                solution = new_solution


    def init_initial_solution(self):
        solutions_dict = dict()
        for agent_id, start, goal in self.agent_start_goal_list:
            solutions_dict[agent_id] = self.shortest_solution(start, goal)
        return solutions_dict

    def shortest_solution(self, start, goal):
        """
        Find the shortest path from start to goal avoiding hard obstacles.
        :param start: The starting vertex.
        :param goal: The goal vertex.
        :return: A list of vertices representing the shortest path.
        """
        def heuristic(v1, v2):
            return nx.shortest_path_length(self.graph, v1, v2)

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
                    cost = len(new_path) + heuristic(neighbor, goal)
                    heappush(open_set, (cost, neighbor, next_time, new_path))

        # Return an empty list if no path is found
        return []

    def construct_new_solution(self, neighborhood: List[int], solution: dict[int, List[int]]):
        """
        Constructs a new solution for the agents in the given neighborhood
        by replanning their paths using SIPP while avoiding collisions.
        :param neighborhood: List of agent IDs to be replanned.
        :param solution: A dictionary with the current solution, where the key is agent_id and value is their current path.
        :return: New solution dictionary with updated paths for agents in the neighborhood.
        """
        new_solution = dict(solution)  # Start with the existing solution
        # Existing paths are used as soft obstacles
        existing_paths = {agent_id: solution[agent_id] for agent_id in solution}
        # For each agent in the neighborhood, we replan their path using sipp
        for agent_id in neighborhood:
            _, start, goal = self.agent_start_goal_list[agent_id]
            new_solution[agent_id] = self.sipp_pathfinding(start, goal, existing_paths)

        return new_solution

    def sipp_pathfinding(self, start, goal, existing_paths):
        """
        Perform SIPP for an agent from start to goal.
        :param start: Starting vertex for the agent.
        :param goal: Goal vertex for the agent.
        :param existing_paths: A dictionary of existing paths of all agents to avoid soft obstacles.
        :return: A list of vertices representing the shortest path found using SIPP.
        """
        open_list = []
        heappush(open_list, (0, start, 0, []))  # (cost, current_vertex, current_time, path)

        visited = set()
        safe_intervals = {v: [] for v in self.graph.nodes}  # Stores safe time intervals for each vertex

        # Generate safe intervals for all vertices based on existing agent paths
        for agent_id, path in existing_paths.items():
            for i, (vertex, time) in enumerate(path):
                if vertex not in safe_intervals:
                    safe_intervals[vertex] = []
                safe_intervals[vertex].append((time, time + 1))  # Occupied vertex at specific times

        # Create a function to get the safe time interval for a vertex
        def get_safe_interval(vertex, current_time):
            intervals = safe_intervals.get(vertex, [])
            for start_time, end_time in intervals:
                if current_time >= start_time and current_time < end_time:
                    return False  # The vertex is blocked at this time
            return True  # Vertex is safe at this time

        while open_list:
            cost, current_vertex, current_time, path = heappop(open_list)

            if (current_vertex, current_time) in visited:
                continue
            visited.add((current_vertex, current_time))

            new_path = path + [current_vertex]

            # If the goal is reached, return the path
            if current_vertex == goal:
                return new_path

            # Explore neighbors
            for neighbor in self.graph.neighbors(current_vertex):
                next_time = current_time + 1
                # Check for hard obstacles and avoid them
                if (neighbor, next_time) in self.hard_obstacles:
                    continue
                # Check for safe intervals (avoid soft obstacles if possible)
                if not get_safe_interval(neighbor, next_time):
                    continue
                # Heuristic cost based on shortest path length
                heuristic_cost = nx.shortest_path_length(self.graph, neighbor, goal)
                new_cost = cost + 1 + heuristic_cost  # The cost is updated by 1 step and heuristic

                heappush(open_list, (new_cost, neighbor, next_time, new_path))

        return []  # Return empty if no path is found





