def apply_influence_to_surrounding_cells(grid, row, col):
    for delta_row in range(-1, 2):
        for delta_col in range(-1, 2):
            if delta_row == 0 and delta_col == 0:
                continue
            new_row = row + delta_row
            new_col = col + delta_col
            if 0 <= new_row < len(grid) and 0 <= new_col < len(grid[0]):
                if grid[new_row][new_col] != 2:
                    grid[new_row][new_col] = 1

def p(grid, enumerator=enumerate):
    processed_grid = [[cell for _, cell in enumerator(row)] for row in grid]
    for row_index, row in enumerator(grid):
        for col_index, cell in enumerator(row):
            if cell == 2:
                apply_influence_to_surrounding_cells(processed_grid, row_index, col_index)
    return processed_grid
