def find_position(flat_list, value):
    return flat_list.index(value)

def calculate_new_position(current, target):
    if current < target - 1:
        return current + 1
    if current > target + 1:
        return current - 1
    return current

def p(grid):

    def flatten_grid(grid):
        return sum(grid, [])
    num_columns = len(grid[0])
    flat_grid = flatten_grid(grid)
    position_3 = find_position(flat_grid, 3)
    position_4 = find_position(flat_grid, 4)
    current_row_3, current_col_3 = divmod(position_3, num_columns)
    target_row_4, target_col_4 = divmod(position_4, num_columns)
    new_row_3 = calculate_new_position(current_row_3, target_row_4)
    new_col_3 = calculate_new_position(current_col_3, target_col_4)
    grid[current_row_3][current_col_3] = 0
    grid[new_row_3][new_col_3] = 3
    return grid
