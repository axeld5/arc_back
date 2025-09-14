def p(grid):

    def find_position_with_all_ones(lst):
        return next((index for index, value in enumerate(map(all, lst)) if value))

    def update_cross_pattern(row, col):
        grid[row - 1][col - 1:col + 2] = [4] * 3
        grid[row + 1][col - 1:col + 2] = [4] * 3
        grid[row][col - 1] = 4
        grid[row][col + 1] = 4
    central_row = find_position_with_all_ones(grid)
    central_col = find_position_with_all_ones(zip(*grid))
    update_cross_pattern(central_row, central_col)
    return grid
