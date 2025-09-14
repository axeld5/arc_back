def fill_sides_with_edges(row, length, center_index):
    for i in range(center_index):
        row[i] = row[0]
        row[length - i - 1] = row[length - 1]
    row[center_index] = 5

def p(grid):
    num_columns = len(grid[0])
    center_index = (num_columns - 1) // 2
    for row_index, row in enumerate(grid):
        if max(row) > 0:
            fill_sides_with_edges(row, num_columns, center_index)
    return grid
