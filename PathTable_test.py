from PathTable import PathTable


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

    return path


def format_cmatrix_for_diff(cm: list[list[int]]) -> str:
    return "\n".join(" ".join(f"{int(x):3}" for x in row) for row in cm)


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
