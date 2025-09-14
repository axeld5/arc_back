def find_max_value(grid):
    return max((value for row in grid for value in row))

def create_transformed_grid(height, width, max_value):
    return [[(row + col) % max_value + 1 for col in range(width)] for row in range(height)]

def p(grid):
    height = len(grid)
    width = len(grid[0])
    max_value = find_max_value(grid)
    return create_transformed_grid(height, width, max_value)
