def flatten_matrix(matrix):
    return sum(matrix, [])

def find_positions_of_value(matrix, value):
    return [(row_index, col_index) for row_index, row in enumerate(matrix) for col_index, element in enumerate(row) if element == value]

def find_max_positions(positions):
    max_row = max((row for row, _ in positions))
    max_col = max((col for _, col in positions))
    return (max_row, max_col)

def move_values_in_grid(grid, positions, max_row, max_col):
    result_grid = [[0] * len(grid[0]) for _ in grid]
    for row, col in positions:
        if row < max_row and col < max_col:
            result_grid[row][col + 1] = grid[row][col]
        else:
            result_grid[row][col] = grid[row][col]
    return result_grid

def solve(grid):
    unique_values = set(flatten_matrix(grid)) - {0}
    result_grid = [[0] * len(grid[0]) for _ in grid]
    for value in unique_values:
        positions = find_positions_of_value(grid, value)
        max_row, max_col = find_max_positions(positions)
        moved_grid = move_values_in_grid(grid, positions, max_row, max_col)
        for row_index, row in enumerate(moved_grid):
            for col_index, element in enumerate(row):
                if element != 0:
                    result_grid[row_index][col_index] = element
    return result_grid
p = solve
