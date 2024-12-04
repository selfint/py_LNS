import numpy as np
from Agent import Agent
from MatrixPathTable import MatrixPathTable


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


def build_agent_from_paths(agent_id: int, paths: list[list[tuple[int, int]]]) -> Agent:
    agent = Agent(
        instance=None,
        agent_id=agent_id,
        start_loc=None,
        end_loc=None,
        n_paths=len(paths),
    )

    agent.paths = paths

    return agent


def format_cmatrix_for_diff(cm: list[list[int]]) -> str:
    return "\n".join(" ".join(f"{int(x):3}" for x in row) for row in cm)


def test_path_table_vertex_collisions():
    path_1_0 = build_path_from_grid([
        [ 0,  1,  2],
        [__, __,  3],
        [__, __, __],
    ])
    path_2_0 = build_path_from_grid([
        [__, __, __],
        [ 1,  2,  3],
        [ 0, __, __],
    ])

    table = MatrixPathTable(agents=[
        build_agent_from_paths(1, [path_1_0]),
        build_agent_from_paths(2, [path_2_0]),
    ])

    table.insert_path(1, 0)
    table.insert_path(2, 0)

    assert table.num_collisions() == 1

    cmatrix = table.get_collisions_matrix().tolist()
    expected = format_cmatrix_for_diff([
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0]
    ])
    assert format_cmatrix_for_diff(cmatrix) == expected

def test_path_table_unique_collisions():
    path_1_0 = build_path_from_grid([
        [ 0,  1,  2],
        [__, __,  3],
        [__, __, __],
    ])
    path_2_0 = build_path_from_grid([
        [ 0,  1,  2],
        [__, __,  3],
        [__, __, __],
    ])

    table = MatrixPathTable(agents=[
        build_agent_from_paths(1, [path_1_0]),
        build_agent_from_paths(2, [path_2_0]),
    ])

    table.insert_path(1, 0)
    table.insert_path(2, 0)

    assert table.num_collisions() == 1


def test_path_table_edge_collisions():
    path_1_0 = build_path_from_grid([
        [ 0,  1,  2],
        [__, __,  3],
        [__, __, __],
    ])
    path_2_0 = build_path_from_grid([
        [__, __,  3],
        [ 0,  1,  2],
        [__, __, __],
    ])

    table = MatrixPathTable(agents=[
        build_agent_from_paths(1, [path_1_0]),
        build_agent_from_paths(2, [path_2_0]),
    ])

    table.insert_path(1, 0)
    table.insert_path(2, 0)

    assert table.num_collisions() == 1

    cmatrix = table.get_collisions_matrix().tolist()
    expected = format_cmatrix_for_diff([
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0]
    ])
    assert format_cmatrix_for_diff(cmatrix) == expected


def test_path_table_remove_edge():
    path_1_0 = build_path_from_grid([
        [ 0,  1,  2],
        [__, __,  3],
        [__, __, __],
    ])
    path_2_0 = build_path_from_grid([
        [__, __,  3],
        [ 0,  1,  2],
        [__, __, __],
    ])

    table = MatrixPathTable(agents=[
        build_agent_from_paths(1, [path_1_0]),
        build_agent_from_paths(2, [path_2_0]),
    ])

    table.insert_path(1, 0)
    table.insert_path(2, 0)
    table.remove_path(1, 0)

    assert table.num_collisions() == 0

    cmatrix = table.get_collisions_matrix().tolist()
    expected = format_cmatrix_for_diff([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ])
    assert format_cmatrix_for_diff(cmatrix) == expected


def test_collisions_along_path_vertices():
    path_1_0 = build_path_from_grid([
        [ 0,  1,  2],
        [__, __,  3],
        [__, __, __],
    ])
    path_2_0 = build_path_from_grid([
        [ 0,  1,  2],
        [__, __,  3],
        [__, __, __],
    ])

    table = MatrixPathTable(agents=[
        build_agent_from_paths(1, [path_1_0]),
        build_agent_from_paths(2, [path_2_0]),
    ])

    table.insert_path(1, 0)
    table.insert_path(2, 0)

    assert table.count_collisions_points_along_path(2, 0) == 1

def test_collisions_along_path_edges():
    path_1_0 = build_path_from_grid([
        [ 0,  1,  2],
        [__, __,  3],
        [__, __, __],
    ])
    path_2_0 = build_path_from_grid([
        [__, __,  3],
        [ 0,  1,  2],
        [__, __, __],
    ])

    table = MatrixPathTable(agents=[
        build_agent_from_paths(1, [path_1_0]),
        build_agent_from_paths(2, [path_2_0]),
    ])

    table.insert_path(1, 0)

    assert table.count_collisions_points_along_path(2, 0) == 1

def _test_get_agent_collisions_for_paths():
    path_1_0 = build_path_from_grid([
        [__, __,  3],
        [__, __,  2],
        [__,  0,  1],
    ])
    path_2_0 = build_path_from_grid([
        [__, __, __],
        [ 0,  1,  2],
        [__, __,  3],
    ])

    agents=[
        build_agent_from_paths(1, [path_1_0]),
        build_agent_from_paths(2, [path_2_0]),
    ]
    table = MatrixPathTable(agents)

    agent = build_agent_from_paths(3, paths=[
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
    ])

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

    assert table.get_agent_collisions_for_paths(agent) == [0, 2, 1]


def test_is_path_available():
    path_1_0 = build_path_from_grid([
        [ 0,  1,  2],
        [__, __,  3],
        [__, __, __],
    ])
    path_2_0 = build_path_from_grid([
        [__, __,  3],
        [ 0,  1,  2],
        [__, __, __],
    ])
    path_3_0 = build_path_from_grid([
        [ 3,  2, __],
        [ 0,  1, __],
        [__, __, __],
    ])

    table = MatrixPathTable(agents=[
        build_agent_from_paths(1, [path_1_0]),
        build_agent_from_paths(2, [path_2_0]),
        build_agent_from_paths(3, [path_3_0]),
    ])

    table.insert_path(1, 0)

    assert not table.is_path_available(1, 0)
    assert not table.is_path_available(2, 0)
    assert table.is_path_available(3, 0)
