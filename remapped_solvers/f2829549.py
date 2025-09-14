def apply_transformation(row):
    for col_index in range(3):
        row[col_index] += row[col_index + 4]
        if row[col_index] > 0:
            row[col_index] = 0
        else:
            row[col_index] = 3
    return row[:3]

def p(grid, index_range=range):
    transformed_grid = []
    for row_index in index_range(4):
        transformed_row = apply_transformation(grid[row_index])
        transformed_grid.append(transformed_row)
    return transformed_grid
