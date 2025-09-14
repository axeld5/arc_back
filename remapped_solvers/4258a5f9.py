def mark_adjacent_cells_as_one(grid):
    for row_idx, row in enumerate(grid):
        for col_idx, value in enumerate(row):
            if value == 5:
                mark_adjacent_cells(grid, row_idx, col_idx)
    return grid

def mark_adjacent_cells(grid, target_row, target_col):
    adjacent_offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    for row_offset, col_offset in adjacent_offsets:
        new_row, new_col = (target_row + row_offset, target_col + col_offset)
        if 0 <= new_row < len(grid) and 0 <= new_col < len(grid[0]):
            grid[new_row][new_col] = 1

def p(grid, grid_enumerate=enumerate):
    return mark_adjacent_cells_as_one(grid)
