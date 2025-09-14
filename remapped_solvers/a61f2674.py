def count_non_zero_elements(column_index, grid, range_function=range):
    count = 0
    for row_index in range_function(9):
        if grid[row_index][column_index] > 0:
            count += 1
    return count

def clear_column(column_index, grid, range_function=range):
    for row_index in range_function(9):
        grid[row_index][column_index] = 0

def fill_column(column_index, value, count, grid):
    for row_index in range(count):
        grid[-(row_index + 1)][column_index] = value

def p(grid, range_function=range):
    non_zero_counts = [0 for _ in range_function(9)]
    for column in range_function(9):
        non_zero_counts[column] = count_non_zero_elements(column, grid, range_function)
        clear_column(column, grid, range_function)
    min_count = min([count for count in non_zero_counts if count > 0])
    min_column_index = non_zero_counts.index(min_count)
    fill_column(min_column_index, 2, min_count, grid)
    max_column_index = non_zero_counts.index(max(non_zero_counts))
    fill_column(max_column_index, 1, non_zero_counts[max_column_index], grid)
    return grid
