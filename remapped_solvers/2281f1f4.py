def update_grid_based_on_conditions(grid):
    num_rows = len(grid) - 1
    num_columns = len(grid[0]) - 1
    for row_index, row in enumerate(grid):
        for col_index, value in enumerate(row):
            if row_index > 0 and col_index < num_rows:
                if grid[row_index][num_columns] == 5 and grid[0][col_index] == 5:
                    grid[row_index][col_index] = 2
    return grid

def p(grid, enumerator=enumerate):
    return update_grid_based_on_conditions(grid)
