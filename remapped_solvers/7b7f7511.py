def p(grid):

    def is_symmetric_row(row):
        mid_index = len(row) // 2
        return row[:mid_index] == row[mid_index:]

    def half_symmetric_rows(grid):
        return [row[:len(row) // 2] for row in grid]

    def is_symmetric_grid(grid):
        return all((is_symmetric_row(row) for row in grid))

    def half_grid(grid):
        return grid[:len(grid) // 2]
    if len(grid[0]) % 2 == 0 and is_symmetric_grid(grid):
        return half_symmetric_rows(grid)
    else:
        return half_grid(grid)
