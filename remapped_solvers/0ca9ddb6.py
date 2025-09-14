def p(grid):

    def mark_diagonal_neighbors(grid, row, col, height, width):
        for d_row, d_col in [[1, 1], [-1, -1], [-1, 1], [1, -1]]:
            new_row, new_col = (row + d_row, col + d_col)
            if 0 <= new_row < height and 0 <= new_col < width:
                grid[new_row][new_col] = 4

    def mark_cardinal_neighbors(grid, row, col, height, width):
        for d_row, d_col in [[0, 1], [0, -1], [-1, 0], [1, 0]]:
            new_row, new_col = (row + d_row, col + d_col)
            if 0 <= new_row < height and 0 <= new_col < width:
                grid[new_row][new_col] = 7
    height, width = (len(grid), len(grid[0]))
    for r in range(height):
        for c in range(width):
            if grid[r][c] == 2:
                mark_diagonal_neighbors(grid, r, c, height, width)
            elif grid[r][c] == 1:
                mark_cardinal_neighbors(grid, r, c, height, width)
    return grid
