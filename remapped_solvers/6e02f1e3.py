def create_empty_grid(size):
    return [[0] * size for _ in range(size)]

def unique_numbers_in_grid(grid):
    return len({cell for row in grid for cell in row})

def fill_diagonal(grid, primary=True, value=5):
    size = len(grid)
    range_iter = range(size)
    if primary:
        for i in range_iter:
            grid[i][i] = value
    else:
        for i in range_iter:
            grid[i][size - 1 - i] = value

def fill_first_row(grid, value=5):
    grid[0] = [value] * len(grid[0])

def solve(grid):
    grid_size = 3
    unique_number_count = unique_numbers_in_grid(grid)
    output_grid = create_empty_grid(grid_size)
    if unique_number_count == 2:
        fill_diagonal(output_grid, primary=True)
    elif unique_number_count == 3:
        fill_diagonal(output_grid, primary=False)
    else:
        fill_first_row(output_grid)
    return output_grid

def p(g):
    return solve(g)
