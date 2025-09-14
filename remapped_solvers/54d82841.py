def solve(grid):

    def should_mark_cell(grid, row, col):
        left_neighbor = grid[row][col - 1]
        right_neighbor = grid[row][col + 1]
        above_cell = grid[row - 1][col]
        return grid[row][col] == 0 and left_neighbor == right_neighbor and (left_neighbor != 0) and (above_cell == left_neighbor)
    marked_grid = [row[:] for row in grid]
    number_of_rows = len(grid)
    number_of_columns = len(grid[0])
    for row in range(1, number_of_rows):
        for col in range(1, number_of_columns - 1):
            if should_mark_cell(grid, row, col):
                marked_grid[number_of_rows - 1][col] = 4
    return marked_grid

def p(j):
    return solve(j)
