def p(grid, range_func=range):

    def get_zero_positions(matrix):
        zero_positions = [(row, col) for row in range_func(len(matrix)) for col in range_func(len(matrix[0])) if matrix[row][col] == 0]
        return zero_positions

    def compute_subgrid_dimensions(zero_positions):
        min_row = min((row for row, _ in zero_positions))
        max_row = max((row for row, _ in zero_positions))
        min_col = min((col for _, col in zero_positions))
        subgrid_size = max_row - min_row + 1
        return (min_row, min_col, subgrid_size)

    def find_min_pattern_size(matrix):

        def attempt_pattern_size(pattern_size):
            positions_to_values = {}
            for row in range_func(len(matrix)):
                for col in range_func(len(matrix[0])):
                    value = matrix[row][col]
                    if value == 0:
                        continue
                    pos_in_pattern = (row % pattern_size, col % pattern_size)
                    if pos_in_pattern in positions_to_values and positions_to_values[pos_in_pattern] != value:
                        return (None, None)
                    positions_to_values[pos_in_pattern] = value
            return (pattern_size, positions_to_values)
        pattern_size, mapping = attempt_pattern_size(2)
        if pattern_size is None:
            pattern_size, mapping = attempt_pattern_size(3)
        return (pattern_size, mapping)

    def reconstruct_subgrid(min_row, min_col, subgrid_size, pattern_size, pattern_mapping):
        subgrid = [[0] * subgrid_size for _ in range_func(subgrid_size)]
        for i in range_func(subgrid_size):
            for j in range_func(subgrid_size):
                original_row = (min_row + i) % pattern_size
                original_col = (min_col + j) % pattern_size
                subgrid[i][j] = pattern_mapping.get((original_row, original_col), 0)
        return subgrid
    zero_positions = get_zero_positions(grid)
    min_row, min_col, subgrid_size = compute_subgrid_dimensions(zero_positions)
    pattern_size, pattern_mapping = find_min_pattern_size(grid)
    return reconstruct_subgrid(min_row, min_col, subgrid_size, pattern_size, pattern_mapping)
