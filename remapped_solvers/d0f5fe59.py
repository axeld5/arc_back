def p(grid, range_function=range):
    num_rows, num_cols = (len(grid), len(grid[0]))
    island_count = 0

    def sink_island(row, col):
        grid[row][col] = 0
        for delta_row, delta_col in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            new_row, new_col = (row + delta_row, col + delta_col)
            if 0 <= new_row < num_rows and 0 <= new_col < num_cols and grid[new_row][new_col]:
                sink_island(new_row, new_col)
    for row in range_function(num_rows):
        for col in range_function(num_cols):
            if grid[row][col]:
                island_count += 1
                sink_island(row, col)
    result_matrix = [[8 * (i == j) for j in range_function(island_count)] for i in range_function(island_count)]
    return result_matrix
