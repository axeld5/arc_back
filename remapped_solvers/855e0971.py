def p(matrix, enumerate_func=enumerate):

    def transpose(matrix):
        return [list(row) for row in zip(*matrix)]
    is_horizontal = any((len(set(row)) == 1 for row in matrix))
    working_matrix = matrix if is_horizontal else transpose(matrix)
    value_coordinates = {}
    zero_out_coords = set()
    for i, row in enumerate_func(working_matrix):
        for j, value in enumerate_func(row):
            if value:
                value_coordinates.setdefault(value, []).append((i, j))
    for coordinates in value_coordinates.values():
        x_coords = [i for i, _ in coordinates]
        y_coords = [j for _, j in coordinates]
        min_x, max_x = (min(x_coords), max(x_coords))
        min_y, max_y = (min(y_coords), max(y_coords))
        zero_coords_in_box = {j for i in range(min_x, max_x + 1) for j in range(min_y, max_y + 1) if working_matrix[i][j] == 0}
        for i, j in coordinates:
            if j in zero_coords_in_box:
                zero_out_coords.add((i, j))
    for i, j in zero_out_coords:
        working_matrix[i][j] = 0
    return working_matrix if is_horizontal else transpose(working_matrix)
