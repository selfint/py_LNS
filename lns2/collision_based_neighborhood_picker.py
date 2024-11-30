from typing import List
from PathTable import PathTable
from lns2.neighborhood_picker import NeighborhoodPicker


class CollisionBasedNeighborhoodPicker(NeighborhoodPicker):
    def __init__(self, n_size):
        self.n_size = n_size

    def pick(self, paths: dict[int: List[int]]) -> List[int]:
        p_table = PathTable(0, 0, len(paths))
        for agent_id, path in paths.items():
            p_table.insert_path(agent_id, path)

        most_colliding_agents = []
        for agent_id, path in paths.items():
            most_colliding_agents.append((agent_id, p_table.count_collisions_points_along_existing_path(agent_id)))
        most_colliding_agents = most_colliding_agents.sort(key=lambda x: x[1], reverse=True)[self.n_size:]
        return [agent[0] for agent in most_colliding_agents]


