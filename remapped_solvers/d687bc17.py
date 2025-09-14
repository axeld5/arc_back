def initialize_output_grid(height, width):
    return [[0] * width for _ in range(height)]

def fill_border(output_grid, top, right, bottom, left, height, width):
    for col in range(1, width - 1):
        output_grid[0][col] = top
        output_grid[height - 1][col] = bottom
    for row in range(1, height - 1):
        output_grid[row][0] = left
        output_grid[row][width - 1] = right

def fill_inner_area(output_grid, grid, top, right, bottom, left, height, width):
    for row in range(1, height - 1):
        for col in range(1, width - 1):
            value = grid[row][col]
            if value == top:
                output_grid[1][col] = value
            elif value == right:
                output_grid[row][width - 2] = value
            elif value == bottom:
                output_grid[height - 2][col] = value
            elif value == left:
                output_grid[row][1] = value

def p(grid):
    height = len(grid)
    width = len(grid[0])
    top_border_value = grid[0][1]
    right_border_value = grid[1][width - 1]
    bottom_border_value = grid[height - 1][1]
    left_border_value = grid[1][0]
    output_grid = initialize_output_grid(height, width)
    fill_border(output_grid, top_border_value, right_border_value, bottom_border_value, left_border_value, height, width)
    fill_inner_area(output_grid, grid, top_border_value, right_border_value, bottom_border_value, left_border_value, height, width)
    return output_grid
