from collections import Counter

def p(grid, range_type=range):
    num_rows, num_cols = (len(grid), len(grid[0]))

    def most_common_value(matrix):
        flat_values = (val for row in matrix for val in row)
        counter = Counter(flat_values)
        return max(counter, key=counter.get)

    def least_common_value(matrix):
        flat_values = (val for row in matrix for val in row)
        counter = Counter(flat_values)
        return min(counter, key=lambda k: (counter[k], k))
    common_value = most_common_value(grid)
    differing_coordinates = [(i, j) for i in range_type(num_rows) for j in range_type(num_cols) if grid[i][j] != common_value]
    processed_grid = [[0 if val == 8 else val for val in row] for row in grid]
    least_value = least_common_value(processed_grid)
    adjusted_grid = [[0 if val == least_value else val for val in row] for row in processed_grid]
    new_boundary_value = least_common_value(adjusted_grid)
    min_row = min((coord[0] for coord in differing_coordinates))
    min_col = min((coord[1] for coord in differing_coordinates))
    max_row = max((coord[0] for coord in differing_coordinates))
    max_col = max((coord[1] for coord in differing_coordinates))
    for i in range_type(min_row, max_row + 1):
        for j in range_type(min_col, max_col + 1):
            grid[i][j] = least_value
    for j in range_type(min_col, max_col + 1):
        grid[min_row][j] = grid[max_row][j] = new_boundary_value
    for i in range_type(min_row, max_row + 1):
        grid[i][min_col] = grid[i][max_col] = new_boundary_value
    return grid
