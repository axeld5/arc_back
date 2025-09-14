def p(input_grid, range_func=range):
    height = width = 10
    target_value = 3
    replacement_value = 8
    output_grid = [row[:] for row in input_grid]

    def is_block(r, c, size):
        if r < 0 or c < 0 or r + size > height or (c + size > width):
            return False
        for dr in range_func(size):
            for dc in range_func(size):
                if input_grid[r + dr][c + dc] != target_value:
                    return False
        return True

    def fill_block(r, c, size, fill_value):
        for dr in range_func(size):
            for dc in range_func(size):
                row, col = (r + dr, c + dc)
                if 0 <= row < height and 0 <= col < width:
                    output_grid[row][col] = fill_value
    for size in (3, 2, 1):
        for r in range_func(height):
            for c in range_func(width):
                if not is_block(r, c, size) or is_block(r - 1, c, size) or is_block(r, c - 1, size):
                    continue
                if size == 1 and is_block(r, c, 2):
                    continue
                if is_block(r + size, c + size, size):
                    offset_a, offset_b = (c + 2 * size, c - size)
                elif is_block(r + size, c - size, size):
                    offset_a, offset_b = (c - 2 * size, c + size)
                else:
                    continue
                fill_block(r - size, offset_a, size, replacement_value)
                fill_block(r + 2 * size, offset_b, size, replacement_value)
    return output_grid
