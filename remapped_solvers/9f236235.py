from collections import Counter

def find_consistent_rows_and_cols(transposed_grid, grid_size, consistent_value):
    consistent_rows = [r for r in range(grid_size) if 0 < r < grid_size - 1 and all((transposed_grid[r][c] == consistent_value for c in range(grid_size)))]
    consistent_cols = [c for c in range(grid_size) if 0 < c < grid_size - 1 and all((transposed_grid[r][c] == consistent_value for r in range(grid_size)))]
    return (sorted(consistent_rows), sorted(consistent_cols))

def get_subgrid_boundaries_for_division(points, grid_size):
    return [-1] + points + [grid_size - 1]

def calculate_subgrid_value(transposed_grid, boundary_rows, boundary_cols):
    num_subgrids_rows = len(boundary_rows) - 1
    num_subgrids_cols = len(boundary_cols) - 1
    subgrid_values = [[0] * num_subgrids_cols for _ in range(num_subgrids_rows)]
    for i in range(num_subgrids_rows):
        mid_row = (boundary_rows[i] + boundary_rows[i + 1]) // 2
        for j in range(num_subgrids_cols):
            mid_col = (boundary_cols[j] + boundary_cols[j + 1]) // 2
            subgrid_values[i][j] = transposed_grid[mid_row][mid_col]
    return subgrid_values

def p(input_grid):
    grid_size = len(input_grid)
    transposed_grid = [row[::-1] for row in input_grid]
    consistent_rows = [r for r in range(grid_size) if all((transposed_grid[r][c] == transposed_grid[r][0] for c in range(grid_size)))]
    consistent_cols = [c for c in range(grid_size) if all((transposed_grid[r][c] == transposed_grid[0][c] for r in range(grid_size)))]
    consistent_counter = Counter([transposed_grid[r][0] for r in consistent_rows] + [transposed_grid[0][c] for c in consistent_cols])
    consistent_value = consistent_counter.most_common(1)[0][0]
    rows_for_subdivision, cols_for_subdivision = find_consistent_rows_and_cols(transposed_grid, grid_size, consistent_value)
    row_boundaries = get_subgrid_boundaries_for_division(rows_for_subdivision, grid_size)
    col_boundaries = get_subgrid_boundaries_for_division(cols_for_subdivision, grid_size)
    return calculate_subgrid_value(transposed_grid, row_boundaries, col_boundaries)
