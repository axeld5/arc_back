def get_adjacent_count(grid, row, col, value):
    count = 0
    num_rows = len(grid)
    num_cols = len(grid[row])
    if row > 0 and grid[row - 1][col] == value:
        count += 1
    if row + 1 < num_rows and grid[row + 1][col] == value:
        count += 1
    if col > 0 and grid[row][col - 1] == value:
        count += 1
    if col + 1 < num_cols and grid[row][col + 1] == value:
        count += 1
    return count

def should_keep_cell(grid, row, col, value):
    adjacent_count = get_adjacent_count(grid, row, col, value)
    return value if adjacent_count > 1 else 0

def solve(grid):
    return [[should_keep_cell(grid, row_index, col_index, value) for col_index, value in enumerate(row)] for row_index, row in enumerate(grid)]
p = solve
