def p(grid):

    def duplicate_grid(g):
        return [row[:] + row[:] for row in g] + [row[:] + row[:] for row in g]

    def mark_diagonals(extended_grid):
        rows, cols = (len(extended_grid), len(extended_grid[0]))
        for row in range(rows):
            for col in range(cols):
                cell_value = extended_grid[row][col]
                if cell_value > 0 and cell_value != 8:
                    for row_offset, col_offset in [[1, 1], [-1, -1], [-1, 1], [1, -1]]:
                        adjacent_row = row + row_offset
                        adjacent_col = col + col_offset
                        if 0 <= adjacent_row < rows and 0 <= adjacent_col < cols:
                            if extended_grid[adjacent_row][adjacent_col] == 0:
                                extended_grid[adjacent_row][adjacent_col] = 8
        return extended_grid
    extended_grid = duplicate_grid(grid)
    return mark_diagonals(extended_grid)
