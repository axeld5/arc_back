def p(grid):
    height, width = (len(grid), len(grid[0]))
    positions_with_2 = []
    for i, row in enumerate(grid):
        for j, value in enumerate(row):
            if value == 2:
                positions_with_2.append((i, j))
    all_non_2_values = []
    for row in grid:
        for value in row:
            non_2_value = value if value != 2 else 0
            all_non_2_values.append(non_2_value)
    unique_values = set(all_non_2_values)
    least_frequent_color = min(unique_values, key=lambda x: sum((1 for v in all_non_2_values if v == x)))
    positions_with_least_color = []
    for i, row in enumerate(grid):
        for j, value in enumerate(row):
            non_2_value = value if value != 2 else 0
            if non_2_value == least_frequent_color:
                positions_with_least_color.append((i, j))
    base_positions = set(positions_with_2) | set(positions_with_least_color)
    upper_i = min((i for i, j in base_positions))
    upper_j = min((j for i, j in base_positions))
    displacement_vectors = []
    for i, j in positions_with_2:
        di = 2 * (i - upper_i) - 1
        dj = 2 * (j - upper_j) - 1
        displacement_vectors.append((di, dj))
    vector_set = set()
    for di, dj in displacement_vectors:
        for k in range(9):
            vector_set.add((k * di, k * dj))
    output = [list(row) for row in grid]
    for di, dj in vector_set:
        for i, j in base_positions:
            new_i, new_j = (i + di, j + dj)
            if 0 <= new_i < height and 0 <= new_j < width:
                output[new_i][new_j] = least_frequent_color
    return [list(row) for row in output]
