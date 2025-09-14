def create_initial_output_grid(height, width, fill_value=8):
    return [[fill_value] * width for _ in range(height)]

def zigzag_fill_output_grid(output_grid):
    num_rows = len(output_grid)
    num_columns = len(output_grid[0])
    column_index = 0
    direction = 1
    for row_index in range(num_rows - 1, -1, -1):
        output_grid[row_index][column_index] = 1
        column_index += direction
        if column_index in (0, num_columns - 1):
            direction *= -1
    return output_grid

def p(grid):
    if not grid:
        return []
    height = len(grid)
    width = len(grid[0])
    output_grid = create_initial_output_grid(height, width)
    return zigzag_fill_output_grid(output_grid)
