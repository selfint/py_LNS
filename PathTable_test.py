import numpy as np


from PathTable import PathTable


def build_path_from_grid(grid: list[list[int]]) -> list[tuple[int, int]]:
    assert all(len(row) == len(grid) for row in grid), "invalid grid dimensions"

    flat_grid = [x for y in grid for x in y]
    path_len = max(flat_grid) + 1

    # subtract 1 for '-1' place holder
    assert path_len == len(set(flat_grid)) - 1, (
        "invalid path indices",
        set(flat_grid),
    )

    path = [None for _ in range(path_len)]

    for x, row in enumerate(grid):
        for y, index in enumerate(row):
            if index >= 0:
                path[index] = (x, y)

    assert all(step is not None for step in path), ("missing path index", path)

    return path


def init_path_table(grids: list[tuple[int, list[int]]]) -> PathTable:
    num_of_agents = len(grids)
    grid_size = len(grids[0])

    table = PathTable(grid_size, grid_size, num_of_agents)
    for agent_id, grid in grids:
        table.insert_path(agent_id, build_path_from_grid(grid))

    return table


def format_cmatrix_for_diff(cmatrix: list[list[int]]) -> str:
    return "\n".join(" ".join(f"{int(x):3}" for x in row) for row in cmatrix)



def test_path_table_vertex_collisions():
    num_of_agents = 2
    table = init_path_table([
        (
            1,
            [
                [ 0,  1,  2],
                [-1, -1,  3],
                [-1, -1, -1],
            ]
        ),
        (
            2,
            [
                [-1, -1, -1],
                [ 1,  2,  3],
                [ 0, -1, -1],
            ]
        )
    ])

    assert table.num_collisions(num_of_agents) == 1

    cmatrix = table.get_collisions_matrix(num_of_agents).tolist()
    expected_cmatrix = format_cmatrix_for_diff([
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0]
    ])
    assert format_cmatrix_for_diff(cmatrix) == expected_cmatrix


def test_path_table_edge_collisions():
    num_of_agents = 2
    table = init_path_table([
        (
            1,
            [
                [ 0,  1,  2],
                [-1, -1,  3],
                [-1, -1, -1],
            ]
        ),
        (
            2,
            [
                [-1, -1,  3],
                [ 0,  1,  2],
                [ 0, -1, -1],
            ]
        )
    ])

    assert table.num_collisions(num_of_agents) == 1

    cmatrix = table.get_collisions_matrix(num_of_agents).tolist()
    expected_cmatrix = format_cmatrix_for_diff([
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0]
    ])
    assert format_cmatrix_for_diff(cmatrix) == expected_cmatrix