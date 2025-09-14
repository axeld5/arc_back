def reverse_and_concatenate(row):
    reversed_row = row[::-1]
    return row + reversed_row

def solve_grid(grid):
    return [reverse_and_concatenate(row) for row in grid]
p = solve_grid
