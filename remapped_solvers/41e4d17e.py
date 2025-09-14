def count_ones_in_subgrid(grid, start_row, start_col):
    count = 0
    for row in range(start_row, start_row + 5):
        for col in range(start_col, start_col + 5):
            if grid[row][col] == 1:
                count += 1
    return count

def mark_non_ones(grid, critical_rows, critical_cols):
    num_rows = len(grid)
    num_cols = len(grid[0])
    for row in range(num_rows):
        for col in range(num_cols):
            if row in critical_rows or col in critical_cols:
                if grid[row][col] != 1:
                    grid[row][col] = 6
    return grid

def p(grid):
    critical_rows = []
    critical_cols = []
    num_rows = len(grid)
    num_cols = len(grid[0])
    for start_row in range(num_rows - 4):
        for start_col in range(num_cols - 4):
            ones_count = count_ones_in_subgrid(grid, start_row, start_col)
            if ones_count == 16:
                critical_rows.append(start_row + 2)
                critical_cols.append(start_col + 2)
    return mark_non_ones(grid, critical_rows, critical_cols)
