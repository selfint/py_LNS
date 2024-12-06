from random import random
from typing import List
from numpy.random import choice
import networkx as nx
from PathTable import PathTable
from lns2.neighborhood_picker import NeighborhoodPicker


class CollisionBasedNeighborhoodPicker(NeighborhoodPicker):
    def __init__(self, n_size: int, graph: nx.Graph):
        self.n_size: int = n_size
        self.graph: nx.Graph = graph

    def pick(self, paths: dict[int: List[int]]) -> list[int]:
        p_table = PathTable(None, None, len(paths))
        for agent_id, path in paths.items():
            p_table.insert_path(agent_id, path)

        collision_graph = p_table.get_collisions_graph()
        random_vertex = choice([v for v in collision_graph.nodes if collision_graph.degree[v] > 0], 1)[0]
        connected_component = self.find_connected_component(random_vertex, collision_graph)
        A_s = []
        if connected_component.size() <= self.n_size:
            A_s.extend(connected_component.nodes)
            while len(A_s) < self.n_size:
                agent = choice(A_s, 1)[0]
                A_s.append(self.random_walk_until_colliding_with_another_agent(paths[:], agent, A_s))
        elif connected_component.size() > self.n_size:
            return self.random_walk(connected_component, random_vertex, self.n_size)

    @staticmethod
    def find_connected_component(vertex: int, collision_graph: nx.Graph) -> nx.Graph:
        if nx.is_connected(collision_graph):
            component = collision_graph.nodes
        else:
            for component in nx.connected_components(collision_graph):
                if vertex in component:
                    break
        return component.subgraph(component).copy()

    @staticmethod
    def random_walk(connected_component, vertex, size):
        """
        TODO not sure if this is the correct implementation of random walk
        Perform a random walk on the connected component from `random_vertex`
        until `n1` vertices are visited. Return the list of vertices visited
        :param connected_component: The connected component to walk on
        :param vertex: The starting vertex
        """
        if connected_component.size() < size:
            raise ValueError("The connected component is smaller than the required size")
        walk = [vertex]
        current_vertex = vertex
        while len(walk) < size:
            neighbors = list(connected_component.neighbors(current_vertex))
            if not neighbors:
                current_vertex = choice(walk, 1)[0]
                continue
            current_vertex = random.choice(neighbors)
            walk.append(current_vertex)
        return walk

    def random_walk_until_colliding_with_another_agent(self, paths, agent, A_s):
        for a_i in A_s:
            if agent != a_i:
                del paths[a_i]
            agent_path = [(vertex, i) for i, vertex in enumerate(paths[agent])]
            del paths[agent]
            random_v, random_timestamp = choice(agent_path, 1)[0]
            while True:
                neighbors = self.graph.neighbors(random_v)
                if len(neighbors) > 1:
                    random_v = choice(neighbors, 1)[0]
                    random_timestamp += 1
                else:
                    random_v, random_timestamp = choice(agent_path, 1)[0]  # If we got stuck we restart
                for agent_id, path in paths.items():
                    if path[random_timestamp] == random_v:
                        return agent_id