def extract_subgrid(grid, start_row, start_col, size):
    return [row[start_col:start_col + size] for row in grid[start_row:start_row + size]]

def flatten_grid(grid):
    return [element for row in grid for element in row]

def p(grid):
    grid_size = len(grid)
    half_size = (grid_size - 1) // 2
    if half_size == 1:
        corner_elements = [grid[0][0], grid[0][2], grid[2][0], grid[2][2]]
        for element in corner_elements:
            if corner_elements.count(element) == 1:
                return [[element]]
    for start_row, start_col in [(0, 0), (0, half_size + 1), (half_size + 1, 0), (half_size + 1, half_size + 1)]:
        subgrid = extract_subgrid(grid, start_row, start_col, half_size)
        flattened_subgrid = flatten_grid(subgrid)
        if len(set(flattened_subgrid)) > 1:
            return subgrid
