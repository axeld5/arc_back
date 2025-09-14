def is_border_or_different(grid, x, y, dx, dy):
    height = len(grid)
    width = len(grid[0])
    neighbor_x, neighbor_y = (x + dx, y + dy)
    if 0 <= neighbor_x < height and 0 <= neighbor_y < width:
        return grid[x][y] != grid[neighbor_x][neighbor_y]
    return True

def has_no_similar_neighbors(grid, x, y):
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if (dx != 0 or dy != 0) and (not is_border_or_different(grid, x, y, dx, dy)):
                return False
    return True

def solve(grid, zero_value=0):
    output_grid = [row[:] for row in grid]
    height = len(output_grid)
    width = len(output_grid[0])
    for i in range(height):
        for j in range(width):
            if output_grid[i][j] != zero_value and has_no_similar_neighbors(output_grid, i, j):
                output_grid[i][j] = zero_value
    return output_grid

def p(g, R=range):
    return solve(g)
