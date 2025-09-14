def update_grid(grid, symmetry_point_x, symmetry_point_y):
    for row_index, row in enumerate(grid):
        for col_index, cell in enumerate(row):
            if cell == 2:
                for sym_x, sym_y in ((row_index, col_index), (symmetry_point_x - row_index, col_index), (row_index, symmetry_point_y - col_index), (symmetry_point_x - row_index, symmetry_point_y - col_index)):
                    grid[sym_x][sym_y] = 2

def calculate_symmetry_points(grid, target_value=3):
    total_x = total_y = 0
    for row_index, row in enumerate(grid):
        for col_index, cell in enumerate(row):
            if cell == target_value:
                total_x += row_index
                total_y += col_index
    symmetry_point_x = total_x // 2
    symmetry_point_y = total_y // 2
    return (symmetry_point_x, symmetry_point_y)

def p(grid, enumerate_fn=enumerate):
    symmetry_point_x, symmetry_point_y = calculate_symmetry_points(grid)
    update_grid(grid, symmetry_point_x, symmetry_point_y)
    return grid
