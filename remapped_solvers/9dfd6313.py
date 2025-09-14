def transpose_matrix(matrix):
    return [list(row) for row in zip(*matrix)]

def p(grid):
    return transpose_matrix(grid)
