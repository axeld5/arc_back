def fill_square(grid, row, col, size_offset, color):
    for row_offset, col_offset in size_offset:
        grid_row = row + row_offset
        grid_col = col + col_offset
        if grid[grid_row][grid_col] == 0:
            grid[grid_row][grid_col] = color

def is_valid_pattern(values):
    non_zero_values = [value for value in values if value]
    return len(non_zero_values) in (1, 4) and len(set(non_zero_values)) == 1

def p(input_grid, valid_range=range(2, 8)):
    offset_patterns = {1: [(-1, -1), (-1, 1), (1, -1), (1, 1)], 2: [(-2, 0), (0, 2), (2, 0), (0, -2)], 3: [(-2, -2), (-2, 2), (2, -2), (2, 2)]}
    for row in valid_range:
        for col in valid_range:
            if input_grid[row][col] == 0:
                continue
            fill_count = {}
            fill_colors = {}
            can_fill = True
            for pattern_size, offsets in offset_patterns.items():
                surrounding_values = [input_grid[row + ro][col + co] for ro, co in offsets]
                if not is_valid_pattern(surrounding_values):
                    can_fill = False
                    break
                non_zero_values = [val for val in surrounding_values if val]
                fill_count[pattern_size] = len(non_zero_values)
                fill_colors[pattern_size] = non_zero_values[0]
            if not can_fill:
                continue
            single_value_patterns = [size for size, count in fill_count.items() if count == 1]
            if len(single_value_patterns) != 1:
                continue
            updated_grid = [row[:] for row in input_grid]
            chosen_pattern_size = single_value_patterns[0]
            fill_color = fill_colors[chosen_pattern_size]
            fill_square(updated_grid, row, col, offset_patterns[chosen_pattern_size], fill_color)
            return updated_grid
    return input_grid
