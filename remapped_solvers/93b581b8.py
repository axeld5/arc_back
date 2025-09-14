def solve_grid(grid, enumerate_function=enumerate):

    def get_filled_positions(grid):
        return [(row, col) for row, row_values in enumerate_function(grid) for col, value in enumerate_function(row_values) if value != 0]

    def get_corners(filled_positions):
        row_indices = [row for row, _ in filled_positions]
        col_indices = [col for _, col in filled_positions]
        return (min(row_indices), max(row_indices), min(col_indices), max(col_indices))

    def set_subgrid_value(grid, start_row, start_col, value):
        for delta_row in range(2):
            for delta_col in range(2):
                new_row, new_col = (start_row + delta_row, start_col + delta_col)
                if 0 <= new_row < len(grid) and 0 <= new_col < len(grid[0]):
                    grid[new_row][new_col] = value
    filled_positions = get_filled_positions(grid)
    min_row, max_row, min_col, max_col = get_corners(filled_positions)
    top_left = grid[min_row][min_col]
    top_right = grid[min_row][max_col]
    bottom_left = grid[max_row][min_col]
    bottom_right = grid[max_row][max_col]
    set_subgrid_value(grid, min_row + 2, min_col + 2, top_left)
    set_subgrid_value(grid, min_row + 2, min_col - 2, top_right)
    set_subgrid_value(grid, min_row - 2, min_col + 2, bottom_left)
    set_subgrid_value(grid, min_row - 2, min_col - 2, bottom_right)
    return grid
p = solve_grid
