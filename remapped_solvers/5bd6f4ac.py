def extract_subgrid_row_values(grid, row_index):
    subgrid_row_values = grid[row_index][6:9]
    return subgrid_row_values

def solve(grid):
    subgrid_rows = []
    for row_index in range(0, 3):
        subgrid_rows.append(extract_subgrid_row_values(grid, row_index))
    return subgrid_rows
p = solve
