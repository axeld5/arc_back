def find_diagonal_start(grid, target):
    num_rows = len(grid)
    num_cols = len(grid[0])
    for row in range(num_rows - 1):
        for col in range(num_cols - 1):
            if grid[row][col] == grid[row + 1][col + 1] == target:
                return (row, col)
    return None

def propagate_diagonal_up(grid, start_row, start_col):
    while start_row >= 1 and start_col >= 1:
        start_row, start_col = (start_row - 1, start_col - 1)
        grid[start_row][start_col] = 1

def propagate_diagonal_down(grid, start_row, start_col):
    num_rows = len(grid)
    num_cols = len(grid[0])
    while start_row < num_rows - 1 and start_col < num_cols - 1:
        start_row, start_col = (start_row + 1, start_col + 1)
        grid[start_row][start_col] = 2

def p(grid):
    start_position = find_diagonal_start(grid, target=1)
    if start_position:
        propagate_diagonal_up(grid, *start_position)
    start_position = find_diagonal_start(grid, target=2)
    if start_position:
        propagate_diagonal_down(grid, *start_position)
    return grid
