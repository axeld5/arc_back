def reverse_inner_list(inner_list):
    return inner_list[::-1]

def reverse_grid(grid):
    return [reverse_inner_list(row) for row in reversed(grid)]

def p(grid):
    return reverse_grid(grid)
