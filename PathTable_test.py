import numpy as np
from Agent import Agent
from PathTable import Edge, PathTable


__ = -1
"""Placeholder for grid index"""


def build_path_from_grid(grid: list[list[int]]) -> list[tuple[int, int]]:
    assert all(len(row) == len(grid) for row in grid), "invalid grid dimensions"

    flat_grid = [x for y in grid for x in y]
    path_len = max(flat_grid) + 1
    indices = list(sorted(x for x in flat_grid if x >= 0))

    assert indices == list(range(path_len)), ("invalid indices", indices)

    path = [None for _ in range(path_len)]

    for x, row in enumerate(grid):
        for y, index in enumerate(row):
            if index != __:
                path[index] = (x, y)

    assert path_len == len(path), ("invalid path", path)
    assert all(step is not None for step in path), ("missing path index", path)

    return np.array(path)


def format_cmatrix_for_diff(cm: list[list[int]]) -> str:
    return "\n".join(" ".join(f"{int(x):3}" for x in row) for row in cm)


def test_path_table_vertices():
    num_of_agents = 2
    table = PathTable(3, 3, num_of_agents)

    table.insert_path(1, build_path_from_grid([
        [ 0,  1,  2],
        [__, __,  3],
        [__, __, __],
    ]))
    table.insert_path(2, build_path_from_grid([
        [__, __, __],
        [ 1,  2,  3],
        [ 0, __, __],
    ]))

    assert table.table[0, 0, 0] == {1}
    assert table.table[0, 0, 0] == {1}
    assert table.table[0, 1, 1] == {1}
    assert table.table[0, 2, 2] == {1}

    assert table.table[2, 0, 0] == {2}
    assert table.table[1, 0, 1] == {2}
    assert table.table[1, 1, 2] == {2}

    assert table.table[1, 2, 3] == {1, 2}

def test_path_table_edges():
    num_of_agents = 2
    table = PathTable(3, 3, num_of_agents)
    table.insert_path(1, build_path_from_grid([
        [ 0,  1,  2],
        [__, __,  3],
        [__, __, __],
    ]))
    table.insert_path(2, build_path_from_grid([
        [__, __,  3],
        [ 0,  1,  2],
        [__, __, __],
    ]))

    assert table.edges[0, 0, 0, 0, 1, 1] == {1}
    assert table.edges[0, 1, 1, 0, 2, 2] == {1}
    assert table.edges[0, 2, 2, 1, 2, 3] == {1}
    assert table.edges[1, 0, 0, 1, 1, 1] == {2}
    assert table.edges[1, 1, 1, 1, 2, 2] == {2}
    assert table.edges[1, 2, 2, 0, 2, 3] == {2}


def test_path_table_vertex_collisions():
    num_of_agents = 2
    table = PathTable(3, 3, num_of_agents)

    table.insert_path(1, build_path_from_grid([
        [ 0,  1,  2],
        [__, __,  3],
        [__, __, __],
    ]))
    table.insert_path(2, build_path_from_grid([
        [__, __, __],
        [ 1,  2,  3],
        [ 0, __, __],
    ]))

    assert table.num_collisions(num_of_agents) == 1

    cmatrix = table.get_collisions_matrix(num_of_agents).tolist()
    expected = format_cmatrix_for_diff([
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0]
    ])
    assert format_cmatrix_for_diff(cmatrix) == expected

def test_path_table_unique_collisions():
    num_of_agents = 2
    table = PathTable(3, 3, num_of_agents)

    table.insert_path(1, build_path_from_grid([
        [ 0,  1,  2],
        [__, __,  3],
        [__, __, __],
    ]))
    table.insert_path(2, build_path_from_grid([
        [ 0,  1,  2],
        [__, __,  3],
        [__, __, __],
    ]))

    assert table.num_collisions(num_of_agents) == 1


def test_path_table_edge_collisions():
    num_of_agents = 2
    table = PathTable(3, 3, num_of_agents)
    table.insert_path(1, build_path_from_grid([
        [ 0,  1,  2],
        [__, __,  3],
        [__, __, __],
    ]))
    table.insert_path(2, build_path_from_grid([
        [__, __,  3],
        [ 0,  1,  2],
        [__, __, __],
    ]))

    assert table.num_collisions(num_of_agents) == 1

    cmatrix = table.get_collisions_matrix(num_of_agents).tolist()
    expected = format_cmatrix_for_diff([
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0]
    ])
    assert format_cmatrix_for_diff(cmatrix) == expected


def test_path_table_remove_edge():
    num_of_agents = 2
    table = PathTable(3, 3, num_of_agents)
    path_1 = build_path_from_grid([
        [ 0,  1,  2],
        [__, __,  3],
        [__, __, __],
    ])
    path_2 = build_path_from_grid([
        [__, __,  3],
        [ 0,  1,  2],
        [__, __, __],
    ])

    table.insert_path(1, path_1)
    table.insert_path(2, path_2)
    table.remove_path(1, path_1)

    assert table.num_collisions(num_of_agents) == 0

    cmatrix = table.get_collisions_matrix(num_of_agents).tolist()
    expected = format_cmatrix_for_diff([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ])
    assert format_cmatrix_for_diff(cmatrix) == expected


def test_collisions_along_path_vertices():
    num_of_agents = 2
    table = PathTable(3, 3, num_of_agents)
    path_1 = build_path_from_grid([
        [ 0,  1,  2],
        [__, __,  3],
        [__, __, __],
    ])
    path_2 = build_path_from_grid([
        [ 0,  1,  2],
        [__, __,  3],
        [__, __, __],
    ])

    table.insert_path(1, path_1)

    assert table.count_collisions_points_along_path(path_2) == 1

def test_collisions_along_path_edges():
    num_of_agents = 2
    table = PathTable(3, 3, num_of_agents)
    path_1 = build_path_from_grid([
        [ 0,  1,  2],
        [__, __,  3],
        [__, __, __],
    ])
    path_2 = build_path_from_grid([
        [__, __,  3],
        [ 0,  1,  2],
        [__, __, __],
    ])

    table.insert_path(1, path_1)

    assert table.count_collisions_points_along_path(path_2) == 1

def test_get_agent_collisions_for_paths():
    num_of_agents = 3
    table = PathTable(3, 3, num_of_agents)
    table.insert_path(1, build_path_from_grid([
        [__, __,  3],
        [__, __,  2],
        [__,  0,  1],
    ]))
    table.insert_path(2, build_path_from_grid([
        [__, __, __],
        [ 0,  1,  2],
        [__, __,  3],
    ]))

    agent = Agent(instance=None, agent_id=3, start_loc=None, end_loc=None, n_paths=3)

    agent.paths = [
        # no collision
        build_path_from_grid([
            [__, __, __],
            [ 3,  2, __],
            [ 0,  1, __],
        ]),
        # 2 vertex collision
        build_path_from_grid([
            [__,  2,  3],  # collision at 3 with path 1
            [__,  1, __],  # collision at 1 with path 2
            [__,  0, __],  # collision at 0 with path 1
        ]),
        # 1 edge collision
        build_path_from_grid([
            [ 0,  1,  2],  # V
            [__, __,  3],  # |- edge collision with path 1
            [__, __, __],
        ]),
    ]

    path_0_collisions = table.get_agent_collisions_for_path(agent.id, agent.paths[0])
    assert path_0_collisions == ([], [])

    path_1_collisions = table.get_agent_collisions_for_path(agent.id, agent.paths[1])
    assert len(path_1_collisions) == 2
    assert path_1_collisions[0] == [
        ((2, 1, 0), {1}),
        ((1, 1, 1), {2}),
        ((0, 2, 3), {1})
    ]
    assert path_1_collisions[1] == []

    path_2_collisions = table.get_agent_collisions_for_path(agent.id, agent.paths[2])
    assert path_2_collisions[0] == []
    assert path_2_collisions[1] == [((0, 2, 2, 1, 2, 3), {1})]

    assert table.get_agent_collisions_for_paths(agent, num_of_agents) == [0, 2, 1]


def test_is_path_available():
    num_of_agents = 2
    table = PathTable(3, 3, num_of_agents)
    path_1 = build_path_from_grid([
        [ 0,  1,  2],
        [__, __,  3],
        [__, __, __],
    ])
    path_2 = build_path_from_grid([
        [__, __,  3],
        [ 0,  1,  2],
        [__, __, __],
    ])
    path_3 = build_path_from_grid([
        [ 3,  2, __],
        [ 0,  1, __],
        [__, __, __],
    ])

    table.insert_path(1, path_1)

    assert not table.is_path_available(path_1)
    assert not table.is_path_available(path_2)
    assert table.is_path_available(path_3)
