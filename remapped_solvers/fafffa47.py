def is_both_cells_empty(top_row_value, bottom_row_value):
    return top_row_value == 0 and bottom_row_value == 0

def calculate_output_cell(top_row_value, bottom_row_value):
    if is_both_cells_empty(top_row_value, bottom_row_value):
        return 2
    return 0

def p(grid):
    top_grid = grid[:3]
    bottom_grid = grid[3:]
    output_grid = []
    for row_index in range(3):
        output_row = []
        for col_index in range(3):
            top_value = top_grid[row_index][col_index]
            bottom_value = bottom_grid[row_index][col_index]
            output_row.append(calculate_output_cell(top_value, bottom_value))
        output_grid.append(output_row)
    return output_grid
