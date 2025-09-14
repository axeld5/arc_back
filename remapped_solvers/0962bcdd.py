def update_grid(grid):
    new_grid = [row[:] for row in grid]
    directions = range(-2, 3)
    for current_row_index, current_row in enumerate(grid):
        for column_index, value in enumerate(current_row):
            if value and current_row[column_index - 1] * current_row[column_index + 1]:
                for row_offset in directions:
                    for col_offset in directions:
                        if abs(row_offset) == abs(col_offset) or not row_offset * col_offset:
                            new_row_index = current_row_index + row_offset
                            new_col_index = column_index + col_offset
                            if 0 <= new_row_index < len(grid) and 0 <= new_col_index < len(current_row):
                                if abs(row_offset) == abs(col_offset):
                                    new_grid[new_row_index][new_col_index] = value
                                else:
                                    new_grid[new_row_index][new_col_index] = current_row[column_index - 1]
    return new_grid

def p(grid, ranges=range(-2, 3), enumerator=enumerate, absolute=abs):
    return update_grid(grid)
