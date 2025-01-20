import random


def generate_random_map(map_type, map_height, map_width, density, obstacle_character="T", space_character="."):
    """
    Generates a random map with the given dimensions and obstacle density.
    :param map_type: The type of map
    :param map_height: Number of rows in the map
    :param map_width: Number of columns in the map
    :param density: The probability of a cell being an obstacle
    :param obstacle_character: The character representing an obstacle
    :param space_character: The character representing an empty space
    :return: A string representing the map
    """
    map_lines = [f"type {map_type}", f"height {map_height}", f"width {map_width}", "map",
                 "".join([obstacle_character] * map_width)]
    for row_index in range(map_height - 2):
        row = obstacle_character
        row += "".join(
            obstacle_character if random.random() < density else space_character
            for _ in range(map_width - 2)
        )
        row += obstacle_character
        map_lines.append(row)

    map_lines.append("".join([obstacle_character] * map_width))
    return "\n".join(map_lines)

