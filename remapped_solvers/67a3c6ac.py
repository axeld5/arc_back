def reverse_row(row):
    return row[::-1]

def reverse_rows_in_grid(grid):
    return [reverse_row(row) for row in grid]

def p(grid):
    return reverse_rows_in_grid(grid)
