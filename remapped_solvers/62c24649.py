def reflect_row(row):
    return row + row[::-1]

def create_symmetric_grid(grid):
    horizontally_symmetric_rows = [reflect_row(row) for row in grid]
    return horizontally_symmetric_rows + horizontally_symmetric_rows[::-1]

def p(grid):
    return create_symmetric_grid(grid)
