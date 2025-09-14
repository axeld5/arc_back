from collections import Counter

def count_and_update_subgrid(grid, updated_grid, row_start, col_start):
    subgrid_elements = [grid[row_start][col_start], grid[row_start][col_start + 1], grid[row_start + 1][col_start], grid[row_start + 1][col_start + 1]]
    common_element, frequency = Counter(subgrid_elements).most_common(1)[0]
    if frequency == 3 and common_element != 0:
        for i in range(row_start, row_start + 2):
            for j in range(col_start, col_start + 2):
                if updated_grid[i][j] == 0:
                    updated_grid[i][j] = 1

def p(grid):
    num_rows = len(grid)
    num_cols = len(grid[0])
    updated_grid = [[8 if val == 8 else 0 for val in row] for row in grid]
    for row in range(num_rows - 1):
        for col in range(num_cols - 1):
            count_and_update_subgrid(grid, updated_grid, row, col)
    return updated_grid
