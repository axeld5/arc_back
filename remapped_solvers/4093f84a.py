def p(grid):

    def transpose(matrix):
        return [list(row) for row in zip(*matrix)]
    flat_values = [value for row in grid for value in row]
    least_frequent = min(set(flat_values), key=lambda k: (flat_values.count(k), k))
    modified_grid = [[5 if value == least_frequent else value for value in row] for row in grid]
    five_positions = [(i, j) for i, row in enumerate(grid) for j, value in enumerate(row) if value == 5]
    if not five_positions:
        use_vertical = True
    else:
        min_row = min((i for i, j in five_positions))
        max_row = max((i for i, j in five_positions))
        min_col = min((j for i, j in five_positions))
        max_col = max((j for i, j in five_positions))
        use_vertical = max_row - min_row > max_col - min_col
    working_grid = modified_grid if use_vertical else transpose(modified_grid)
    midpoint = 7
    sorted_grid = []
    for row in working_grid:
        sorted_row = sorted(row[:midpoint]) + sorted(row[midpoint:], reverse=True)
        sorted_grid.append(sorted_row)
    return sorted_grid if use_vertical else transpose(sorted_grid)
