def add_padding(input_grid, range_function=range):
    height, width = (len(input_grid), len(input_grid[0]))
    padded_grid = [[0] * (width + 2) for _ in range_function(height + 2)]
    for i in range_function(height):
        for j in range_function(width):
            padded_grid[i + 1][j + 1] = input_grid[i][j]
    for j in range_function(width):
        padded_grid[0][j + 1] = input_grid[0][j]
        padded_grid[height + 1][j + 1] = input_grid[-1][j]
    for i in range_function(height):
        padded_grid[i + 1][0] = input_grid[i][0]
        padded_grid[i + 1][width + 1] = input_grid[i][-1]
    return padded_grid

def p(I, K=range):
    return add_padding(I, K)
