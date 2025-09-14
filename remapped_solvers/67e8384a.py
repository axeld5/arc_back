def reflect_row(row):
    return row + row[::-1]

def create_symmetric_grid(grid):
    reflected_rows = [reflect_row(row) for row in grid]
    return reflected_rows + reflected_rows[::-1]

def p(grid):
    return create_symmetric_grid(grid)
