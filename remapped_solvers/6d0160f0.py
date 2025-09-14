def extract_subgrid(grid, start_row, start_col):
    return [[grid[start_row + i][start_col + j] for j in range(3)] for i in range(3)]

def expand_subgrid_with_cross(subgrid, position):
    expanded_grid = [[0] * 11 for _ in range(11)]
    base_row, base_col = position
    for i in range(3):
        for j in range(3):
            expanded_grid[4 * base_row + i][4 * base_col + j] = subgrid[i][j]
    for k in range(11):
        expanded_grid[k][3] = 5
        expanded_grid[k][7] = 5
        expanded_grid[3][k] = 5
        expanded_grid[7][k] = 5
    return expanded_grid

def p(grid):
    for row in range(3):
        for col in range(3):
            subgrid = extract_subgrid(grid, 4 * row, 4 * col)
            for i in range(3):
                for j in range(3):
                    if subgrid[i][j] == 4:
                        return expand_subgrid_with_cross(subgrid, (i, j))
