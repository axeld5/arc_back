def solve_grid_mirror(grid):

    def apply_mirror_logic(row):
        transformed_row = []
        for j in range(4):
            if row[j] * row[8 - j] == 0:
                transformed_row.append(row[j] or row[8 - j])
            else:
                transformed_row.append(row[j])
        return transformed_row
    transformed_grid = [apply_mirror_logic(row) for row in grid]
    return transformed_grid
p = solve_grid_mirror
