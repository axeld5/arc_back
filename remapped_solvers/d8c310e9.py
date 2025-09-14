def p(grid, row_count=5, col_count=15):

    def find_last_occupied_column():
        last_occupied = 0
        for col in range(col_count):
            if any((grid[row][col] for row in range(row_count))):
                last_occupied = col + 1
        return last_occupied

    def flatten_column_structure(column_lengths):
        flattened = []
        for col_idx, length in enumerate(column_lengths):
            for row_idx in range(length):
                flattened.append(grid[row_count - 1 - row_idx][col_idx])
        return flattened

    def transform_grid(num_cols):
        if last_occupied_column < num_cols:
            return None
        column_lengths = [sum((1 for row in range(row_count) if grid[row][col])) for col in range(num_cols)]
        flattened = flatten_column_structure(column_lengths)

        def calculate_mismatches(flip):
            new_grid = [[0] * row_count for _ in range(last_occupied_column)]
            num_blocks = (last_occupied_column + num_cols - 1) // num_cols
            for block in range(num_blocks):
                flipped = flip and block % 2
                index = 0
                for col_idx, length in enumerate(column_lengths):
                    adjusted_col = block * num_cols + (num_cols - 1 - col_idx if flipped else col_idx)
                    if adjusted_col >= last_occupied_column:
                        index += length
                        continue
                    for row_idx in range(length):
                        new_grid[adjusted_col][row_count - 1 - row_idx] = flattened[index]
                        index += 1
            mismatches = 0
            matches = 0
            non_zero_count = 0
            for col in range(last_occupied_column):
                for row in range(row_count):
                    original_value = grid[row][col]
                    new_value = new_grid[col][row]
                    if original_value != new_value:
                        mismatches += 1
                        non_zero_count += original_value or new_value
                    elif original_value:
                        matches += 1
            return (non_zero_count, mismatches, -matches)
        scores = [(calculate_mismatches(False), 0), (calculate_mismatches(True), 1)]
        return (min(scores)[0], column_lengths, flattened, min(scores)[1], num_cols)
    last_occupied_column = find_last_occupied_column()
    best_result = None
    for num_cols in (3, 4):
        current_result = transform_grid(num_cols)
        if current_result and (best_result is None or current_result[0] < best_result[0]):
            best_result = current_result
    _, column_lengths, flattened, flip, num_cols = best_result
    output_grid = [[0] * col_count for _ in range(row_count)]
    num_blocks = (col_count + num_cols - 1) // num_cols
    for block in range(num_blocks):
        is_flipped = flip and block % 2
        index = 0
        for col_idx, length in enumerate(column_lengths):
            adjusted_col = block * num_cols + (num_cols - 1 - col_idx if is_flipped else col_idx)
            for row_idx in range(length):
                if adjusted_col < col_count:
                    output_grid[row_count - 1 - row_idx][adjusted_col] = flattened[index]
                index += 1
    return output_grid
