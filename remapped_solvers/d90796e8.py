def is_valid_position(x, y, max_x, max_y):
    return 0 <= x < max_x and 0 <= y < max_y

def process_neighbors(grid, row_idx, col_idx):
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    num_rows = len(grid)
    num_cols = len(grid[0])
    for d_row, d_col in directions:
        new_row = row_idx + d_row
        new_col = col_idx + d_col
        if is_valid_position(new_row, new_col, num_rows, num_cols) and grid[new_row][new_col] == 3:
            grid[row_idx][col_idx] = 0
            grid[new_row][new_col] = 8

def p(grid, enumerator=enumerate):
    for row_idx, row in enumerator(grid):
        for col_idx, value in enumerator(row):
            if value == 2:
                process_neighbors(grid, row_idx, col_idx)
    return grid
