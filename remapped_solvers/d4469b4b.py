def get_transformation(grid):
    flattened_grid = [value for row in grid for value in row]
    max_value = max(flattened_grid)
    return max_value

def p(grid):
    transformations = {2: [[5, 5, 5], [0, 5, 0], [0, 5, 0]], 1: [[0, 5, 0], [5, 5, 5], [0, 5, 0]], 3: [[0, 0, 5], [0, 0, 5], [5, 5, 5]]}
    key = get_transformation(grid)
    return transformations[key]
