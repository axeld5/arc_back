def extract_subgrid_row(row, subgrid_length):
    return row[:subgrid_length]

def solve(grid):
    subgrid_length = int(len(grid[0]) / 3)
    return [extract_subgrid_row(row, subgrid_length) for row in grid]
p = solve
