def reverse_row(row):
    return row[::-1]

def reverse_grid(grid):
    reversed_rows = [reverse_row(row) for row in grid]
    reversed_grid = reversed_rows[::-1]
    return reversed_grid

def p(grid):
    return reverse_grid(grid)
