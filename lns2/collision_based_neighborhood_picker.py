from random import random
from typing import List
from numpy.random import choice
import networkx as nx
from PathTable import PathTable
from neighborhood_picker import NeighborhoodPicker


class CollisionBasedNeighborhoodPicker(NeighborhoodPicker):
    def __init__(self, n_size: int, graph: nx.Graph):
        self.n_size: int = n_size
        self.graph: nx.Graph = graph

    def pick(self, paths: dict[int: List[int]]) -> list[int]:
        p_table = PathTable(None, None, len(paths))
        for agent_id, path in paths.items():
            p_table.insert_path(agent_id, path)

        collision_graph = p_table.get_collisions_graph()
        random_vertex = choice([v for v in collision_graph.nodes if collision_graph.degree[v] > 0])
        connected_component = self.find_connected_component(random_vertex, collision_graph)
        A_s: list[int] = []
        if connected_component.size() <= self.n_size:
            A_s.extend(connected_component.nodes)
            while len(A_s) < self.n_size:
                agent = choice(A_s)
                A_s.append(self.random_walk_until_colliding_with_another_agent(paths, agent, A_s))
        elif connected_component.size() > self.n_size:
            return self.random_walk(connected_component, random_vertex, self.n_size)

    @staticmethod
    def find_connected_component(vertex: object, collision_graph: nx.Graph) -> nx.Graph:
        if vertex not in collision_graph:
            raise ValueError(f"The vertex {vertex} is not in the graph.")

        if nx.is_connected(collision_graph):
            return collision_graph
        else:
            for component in nx.connected_components(collision_graph):
                if vertex in component:
                    return collision_graph.subgraph(component).copy()

        raise ValueError(f"The vertex {vertex} is not part of any connected component.")

    @staticmethod
    def random_walk(connected_component: nx.Graph, vertex: int, size: int) -> list[int]:
        """
        TODO not sure if this is the correct implementation of random walk
        Perform a random walk on the connected component from `random_vertex`
        until `n1` vertices are visited. Return the list of vertices visited
        :param connected_component: The connected component to walk on
        :param vertex: The starting vertex
        """
        if connected_component.size() < size:
            raise ValueError("The connected component is smaller than the required size")
        walk = set()
        walk.add(vertex)
        current_vertex = vertex
        while len(walk) < size:
            neighbors = list(connected_component.neighbors(current_vertex))
            if not neighbors:
                current_vertex = choice(walk)
                continue
            current_vertex = choice(neighbors)
            walk.add(current_vertex)
        return list(walk)

    def random_walk_until_colliding_with_another_agent(self, paths: dict[int, list], agent: int, A_s: list[int]) -> int:
        paths = paths[:]  # We are going to modify the dictionary and we don't want to ruin the original
        for a_i in A_s:  # Delete all agents` paths in A_s
            if agent != a_i:
                del paths[a_i]

        agent_path = [(vertex, i) for i, vertex in enumerate(paths[agent])]  # Our agent's path
        del paths[agent]  # Delete our agent's path so won't collide with itself
        random_v, random_timestamp = choice(agent_path)  # Start the random walk from a random vertex
        while True:
            neighbors = self.graph.neighbors(random_v)
            if len(neighbors) >= 1:
                random_v = choice(neighbors)
                random_timestamp += 1
            else:
                random_v, random_timestamp = choice(agent_path)  # If we got stuck we restart
            for agent_id, path in paths.items():  # For all agents not in A_s, check if collided with one and return it
                if path[random_timestamp] == random_v:
                    return agent_id