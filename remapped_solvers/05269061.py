def build_modulo_dict(grid):
    modulo_dict = {}
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            value = grid[i][j]
            if value:
                modulo_dict[(i + j) % 3] = value
    return modulo_dict

def transform_grid(grid, modulo_dict):
    num_rows = len(grid)
    num_cols = len(grid[0])
    transformed_grid = []
    for i in range(num_rows):
        row = []
        for j in range(num_cols):
            value = modulo_dict.get((i + j) % 3, 0)
            row.append(value)
        transformed_grid.append(row)
    return transformed_grid

def p(grid):
    modulo_dict = build_modulo_dict(grid)
    return transform_grid(grid, modulo_dict)
