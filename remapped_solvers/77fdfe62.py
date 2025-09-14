def solve_subgrid(grid, subgrid_size, corner_values):
    solved_subgrid = []
    for row_index in range(2, len(grid) - 2):
        new_row = []
        for col_index in range(2, len(grid[row_index]) - 2):
            value = grid[row_index][col_index]
            if value == 8:
                row_section = (row_index - 2) // subgrid_size
                col_section = (col_index - 2) // subgrid_size
                corner_index = row_section * 2 + col_section
                value = corner_values[corner_index]
            new_row.append(value)
        solved_subgrid.append(new_row)
    return solved_subgrid

def p(grid):
    num_rows = len(grid)
    subgrid_size = num_rows // 2 - 2
    corner_values = [grid[0][0], grid[0][-1], grid[-1][0], grid[-1][-1]]
    return solve_subgrid(grid, subgrid_size, corner_values)
