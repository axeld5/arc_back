def find_eights_coordinates(grid):
    return [(i, j) for i, row in enumerate(grid) for j, value in enumerate(row) if value == 8]

def is_within_eights_boundary(i, j, eights_coordinates):
    min_i = min((i for i, _ in eights_coordinates))
    max_i = max((i for i, _ in eights_coordinates))
    min_j = min((j for _, j in eights_coordinates))
    max_j = max((j for _, j in eights_coordinates))
    return min_i <= i <= max_i and min_j <= j <= max_j

def process_grid(grid):
    eights_coordinates = find_eights_coordinates(grid)
    return [[2 if is_within_eights_boundary(i, j, eights_coordinates) and grid[i][j] == 0 else grid[i][j] for j in range(len(grid[0]))] for i in range(len(grid))]

def p(grid):
    return process_grid(grid)
