def is_border(i, j, rows, cols):
    return i < 0 or i >= rows or j < 0 or (j >= cols)

def is_exposed_zero(grid, i, j, rows, cols):
    neighbors = [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]
    return any((is_border(x, y, rows, cols) or grid[x][y] == 0 for x, y in neighbors))

def p(grid):
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0
    result_grid = []
    for i in range(rows):
        result_row = []
        for j in range(cols):
            if grid[i][j] != 0 and is_exposed_zero(grid, i, j, rows, cols):
                result_row.append(grid[i][j])
            else:
                result_row.append(0)
        result_grid.append(result_row)
    return result_grid
