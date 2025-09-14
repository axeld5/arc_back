def expand_grid(grid, scale_factor):
    expanded_grid = [[element for element in row for _ in range(scale_factor)] for row in grid for _ in range(scale_factor)]
    return expanded_grid

def calculate_scale_factor(grid):
    unique_non_zero_elements = set(sum(grid, [])) - {0}
    return len(unique_non_zero_elements)

def p(grid):
    scale_factor = calculate_scale_factor(grid)
    expanded_grid = expand_grid(grid, scale_factor)
    return expanded_grid
