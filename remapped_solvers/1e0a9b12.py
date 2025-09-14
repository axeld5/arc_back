def collapse_non_zeros(column):
    non_zeros = [element for element in column if element != 0]
    zero_count = len(column) - len(non_zeros)
    return [0] * zero_count + non_zeros

def transpose(grid):
    return list(map(list, zip(*grid)))

def solve_grid(grid):
    transposed_grid = transpose(grid)
    collapsed_grid = [collapse_non_zeros(column) for column in transposed_grid]
    return transpose(collapsed_grid)
p = solve_grid
