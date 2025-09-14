def update_adjacent_cells(grid, cells_with_ones):
    for row, col in cells_with_ones:
        if row > 0:
            grid[row - 1][col] = 2
        if row < 9:
            grid[row + 1][col] = 8
        if col > 0:
            grid[row][col - 1] = 7
        if col < 9:
            grid[row][col + 1] = 6

def find_cells_with_value_one(grid):
    cells_with_ones = []
    for i, row in enumerate(grid):
        for j, value in enumerate(row):
            if value == 1:
                cells_with_ones.append([i, j])
    return cells_with_ones

def p(grid, enumerator=enumerate):
    cells_with_ones = find_cells_with_value_one(grid)
    update_adjacent_cells(grid, cells_with_ones)
    return grid
