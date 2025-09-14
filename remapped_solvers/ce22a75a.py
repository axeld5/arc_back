def mark_surrounding_cells(grid, target_value=5):
    marked_grid = [[0 for _ in range(9)] for _ in range(9)]
    for row in range(9):
        for col in range(9):
            if grid[row][col] == target_value:
                mark_neighbours(marked_grid, row, col)
    return marked_grid

def mark_neighbours(marked_grid, row, col):
    for r in range(max(0, row - 1), min(9, row + 2)):
        for c in range(max(0, col - 1), min(9, col + 2)):
            marked_grid[r][c] = 1

def p(grid, range_function=range):
    return mark_surrounding_cells(grid)
