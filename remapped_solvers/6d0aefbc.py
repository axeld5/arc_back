def mirror_row(row):
    return row + row[::-1]

def solve_grid(grid):
    mirrored_grid = [mirror_row(row) for row in grid]
    return mirrored_grid
p = solve_grid
