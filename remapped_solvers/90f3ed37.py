from copy import deepcopy

def p(input_grid, range_func=range, enumerate_func=enumerate):
    H, W = (15, 10)
    grid_copy = deepcopy(input_grid)
    max_height_check = min(5, H)
    row_counts = [sum((1 for col in range_func(W) if input_grid[row][col] == 8)) for row in range_func(max_height_check)]
    max_sum = -1
    best_rows = None
    for height in (1, 2, 3):
        for start_row in range_func(0, max_height_check - height + 1):
            row_window = list(range_func(start_row, start_row + height))
            if height > 1 and any((row_counts[row] == 0 for row in row_window)):
                continue
            current_sum = sum((row_counts[row] for row in row_window))
            if current_sum > max_sum:
                max_sum, best_rows = (current_sum, row_window)
    num_best_rows = len(best_rows)
    pattern = [[1 if input_grid[row][col] == 8 else 0 for col in range_func(W)] for row in best_rows]
    pattern = [row + [0] for row in pattern]
    extended_width = W + 1

    def calculate_max_width(x, y, offset):
        if x + num_best_rows > H:
            return 0
        max_length = 0
        while y + max_length < W and offset + max_length < extended_width:
            can_extend = True
            for height_offset in range_func(num_best_rows):
                pattern_match = pattern[height_offset][offset + max_length] == 1
                grid_match = input_grid[x + height_offset][y + max_length] == 8
                if pattern_match != grid_match:
                    can_extend = False
                    break
            if not can_extend:
                break
            max_length += 1
        return max_length

    def find_best_extension(start_row, offset):
        extensions = [calculate_max_width(start_row, col, offset) for col in range_func(3)]
        return (extensions, max(extensions) if extensions else 0)

    def can_place_pattern(start_row):
        if start_row + num_best_rows > H:
            return False
        for height_offset in range_func(num_best_rows):
            has_eight = any((input_grid[start_row + height_offset][col] == 8 for col in range_func(W)))
            if not has_eight:
                return False
        return True

    def apply_pattern(start_row, start_col, offset):
        for height_offset in range_func(num_best_rows):
            for col_offset in range_func(offset, extended_width):
                column = start_col + (col_offset - offset)
                if column >= W:
                    break
                if pattern[height_offset][col_offset] == 1 and grid_copy[start_row + height_offset][column] != 8:
                    grid_copy[start_row + height_offset][column] = 1
    current_row = best_rows[-1] + 1
    max_appliable_height = H - num_best_rows
    while current_row <= max_appliable_height:
        first_offset, max_width = find_best_extension(current_row, 0)
        if max_width >= 2:
            best_row = current_row
            max_offset = first_offset
            max_width_at_offset = max_width
            best_offset = 0
            for delta in range_func(1, num_best_rows):
                next_row = current_row + delta
                if next_row > max_appliable_height:
                    break
                next_offset, next_max = find_best_extension(next_row, 0)
                if next_max > max_width_at_offset:
                    best_row = next_row
                    max_offset = next_offset
                    max_width_at_offset = next_max
                    best_offset = 0
            best_col = next((i for i, length in enumerate_func(max_offset) if length == max_width_at_offset))
            apply_pattern(best_row, best_col, best_offset)
            current_row = best_row + num_best_rows
            continue
        if can_place_pattern(current_row):
            best_row = None
            max_width_at_offset = 0
            max_offset = None
            for delta in range_func(0, num_best_rows):
                next_row = current_row + delta
                if next_row > max_appliable_height:
                    break
                if not can_place_pattern(next_row):
                    continue
                next_offset, next_max = find_best_extension(next_row, 1)
                if next_max > max_width_at_offset:
                    best_row = next_row
                    max_width_at_offset = next_max
                    max_offset = next_offset
            if max_width_at_offset >= 2:
                best_col = next((i for i, length in enumerate_func(max_offset) if length == max_width_at_offset))
                apply_pattern(best_row, best_col, 1)
                current_row = best_row + num_best_rows
                continue
        current_row += 1
    return grid_copy
