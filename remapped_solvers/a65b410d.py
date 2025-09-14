def solve(grid, range_func=range):
    num_rows = len(grid)
    for row_index in range_func(num_rows):
        if grid[row_index][0] == 2:
            largest_segment_length = find_largest_segment_of_twos(grid[row_index], num_rows)
            apply_transformation(grid, row_index, largest_segment_length, range_func)
    return grid

def find_largest_segment_of_twos(row, num_rows):
    segment_length = 0
    while segment_length < num_rows and row[segment_length] == 2:
        segment_length += 1
    return segment_length

def apply_transformation(grid, current_row, largest_segment_length, range_func):
    num_rows = len(grid)
    for other_row in range_func(num_rows):
        range_limit = (largest_segment_length + current_row - other_row) * (other_row != current_row)
        for col_index in range_func(range_limit):
            grid[other_row][col_index] = 3 - 2 * (other_row > current_row)
p = solve
