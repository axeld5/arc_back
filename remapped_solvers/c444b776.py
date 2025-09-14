def p(grid):
    Y = 4
    height = len(grid)
    width = len(grid[0])

    def find_row_boundaries():
        return [r for r in range(height) if all((grid[r][c] == Y for c in range(width)))]

    def find_col_boundaries():
        return [c for c in range(width) if all((grid[r][c] == Y for r in range(height)))]

    def calculate_unit_sizes(row_boundaries, col_boundaries):
        unit_row_size = row_boundaries[1] - row_boundaries[0] - 1 if len(row_boundaries) >= 2 else row_boundaries[0] if row_boundaries else height
        unit_col_size = col_boundaries[1] - col_boundaries[0] - 1 if len(col_boundaries) >= 2 else col_boundaries[0] if col_boundaries else width
        return (unit_row_size, unit_col_size)

    def find_top_left_of_block():
        for r in range(height):
            for c in range(width):
                if grid[r][c] not in (0, Y):
                    top_row = r // (unit_row_size + 1) * (unit_row_size + 1)
                    left_col = c // (unit_col_size + 1) * (unit_col_size + 1)
                    return (top_row, left_col)
        return None
    row_boundaries = find_row_boundaries()
    col_boundaries = find_col_boundaries()
    unit_row_size, unit_col_size = calculate_unit_sizes(row_boundaries, col_boundaries)
    block_height_sections = len(row_boundaries) + 1
    block_width_sections = len(col_boundaries) + 1
    top_left = find_top_left_of_block()
    pattern = [[grid[top_left[0] + dr][top_left[1] + dc] for dc in range(unit_col_size)] for dr in range(unit_row_size)]
    result_grid = [[0] * width for _ in range(height)]

    def map_boundaries_to_output():
        for r in range(height):
            for c in range(width):
                if r % (unit_row_size + 1) == unit_row_size or c % (unit_col_size + 1) == unit_col_size:
                    result_grid[r][c] = Y

    def fill_patterns_within_boundaries():
        for row_block in range(block_height_sections):
            start_row = row_block * (unit_row_size + 1)
            for col_block in range(block_width_sections):
                start_col = col_block * (unit_col_size + 1)
                for dr in range(unit_row_size):
                    for dc in range(unit_col_size):
                        value = pattern[dr][dc]
                        if value:
                            result_grid[start_row + dr][start_col + dc] = value
    map_boundaries_to_output()
    fill_patterns_within_boundaries()
    return result_grid
