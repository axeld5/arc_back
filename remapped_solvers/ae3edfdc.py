def propagate_protection(input_grid, direction, current_position, protected_value, grid_size):
    protected_positions = []
    row, col = current_position
    delta_row, delta_col = direction
    target_row, target_col = (row + 2 * delta_row, col + 2 * delta_col)
    while 0 <= target_row < grid_size and 0 <= target_col < grid_size:
        if input_grid[target_row][target_col] == protected_value:
            intermediate_row, intermediate_col = (row + delta_row, col + delta_col)
            if 0 <= intermediate_row < grid_size and 0 <= intermediate_col < grid_size:
                protected_positions.append((intermediate_row, intermediate_col))
            break
        target_row += delta_row
        target_col += delta_col
    return protected_positions

def initialize_empty_grid(size):
    return [[0] * size for _ in range(size)]

def p(input_grid, grid_size=15, range_func=range):
    output_grid = initialize_empty_grid(grid_size)
    directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    for row in range_func(grid_size):
        for col in range_func(grid_size):
            cell_value = input_grid[row][col]
            if cell_value in (1, 2):
                output_grid[row][col] = cell_value
                protection_value = 3 if cell_value == 2 else 7
                for direction in directions:
                    protected_positions = propagate_protection(input_grid, direction, (row, col), protection_value, grid_size)
                    for r, c in protected_positions:
                        output_grid[r][c] = protection_value
    return output_grid
