def find_green_row(grid, range_k):
    GREEN = 5
    for row_index in range_k:
        if all((grid[row_index][col_index] == GREEN for col_index in range_k)):
            return row_index
    return -1

def distribute_blocks(grid, green_row, result_grid, range_k):
    EMPTY = 0
    BLUE = 1
    RED = 2
    GREEN = 5
    for col_index in range_k:
        result_grid[green_row][col_index] = GREEN
    for row in range_k:
        for col in range_k:
            value = grid[row][col]
            if value in (BLUE, RED):
                direction = (-1 if value == BLUE else 1) * (1 if row < green_row else -1)
                current_row = row
                while 0 <= current_row < 10 and result_grid[current_row][col] == EMPTY:
                    result_grid[current_row][col] = value
                    current_row += direction
    return result_grid

def p(grid, index_range=range(10)):
    GRID_SIZE = 10
    result_grid = [[0] * GRID_SIZE for _ in index_range]
    green_row = find_green_row(grid, index_range)
    return distribute_blocks(grid, green_row, result_grid, index_range)
