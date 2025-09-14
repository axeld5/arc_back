def find_largest_zero_subgrid(grid):
    max_area = -1
    best_rectangle = None
    num_rows = len(grid)
    num_cols = len(grid[0])
    for width in range(2, 10):
        for height in range(2, 10):
            if height > num_rows or width > num_cols:
                continue
            for start_row in range(num_rows - height + 1):
                for start_col in range(num_cols - width + 1):
                    if is_zero_filled(grid, start_row, start_col, height, width):
                        area = height * width
                        if area > max_area:
                            max_area = area
                            best_rectangle = (start_row, start_col, height, width)
    return best_rectangle

def is_zero_filled(grid, start_row, start_col, height, width):
    for row in range(start_row, start_row + height):
        for col in range(start_col, start_col + width):
            if grid[row][col] != 0:
                return False
    return True

def mark_subgrid(grid, start_row, start_col, height, width, mark_value):
    new_grid = [row[:] for row in grid]
    for row in range(start_row, start_row + height):
        for col in range(start_col, start_col + width):
            new_grid[row][col] = mark_value
    return new_grid

def p(g):
    best_rectangle = find_largest_zero_subgrid(g)
    if best_rectangle is None:
        return g
    start_row, start_col, height, width = best_rectangle
    return mark_subgrid(g, start_row, start_col, height, width, 6)
