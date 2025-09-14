def is_within_grid(x, y, size=10):
    return 0 <= x < size and 0 <= y < size

def can_place_number(given_grid, x, y):
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    for dx, dy in directions:
        nx, ny = (x + dx, y + dy)
        if is_within_grid(nx, ny) and given_grid[nx][ny] != 0:
            return False
    return True

def transfer_number(fixed_grid, given_grid, x, y):
    diagonals = [(-1, -1), (-1, 1), (1, 1), (1, -1)]
    for dx, dy in diagonals:
        diag_x, diag_y = (x + dx, y + dy)
        if is_within_grid(diag_x, diag_y) and given_grid[diag_x][diag_y] != 0:
            opposite_dx, opposite_dy = (-dx, -dy)
            for step in range(1, 10):
                new_x, new_y = (x + opposite_dx * step, y + opposite_dy * step)
                if is_within_grid(new_x, new_y):
                    fixed_grid[new_x][new_y] = given_grid[x][y]

def p(given_grid):
    grid_size = 10
    result_grid = [row[:] for row in given_grid]
    for i in range(grid_size):
        for j in range(grid_size):
            if given_grid[i][j] != 0 and can_place_number(given_grid, i, j):
                transfer_number(result_grid, given_grid, i, j)
    return result_grid
