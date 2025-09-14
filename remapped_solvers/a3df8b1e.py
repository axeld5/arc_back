def fill_grid(grid):
    rows, cols = (len(grid), len(grid[0]))
    current_row = rows - 1
    current_col = 0
    direction = 1
    while current_row >= 0:
        grid[current_row][current_col] = 1
        new_row = current_row - 1
        new_col = current_col + direction
        if 0 <= new_col < cols:
            current_row = new_row
            current_col = new_col
        else:
            current_row = new_row
            direction = -direction
            current_col = current_col + direction
    return grid

def p(j):
    return fill_grid(j)
