def p(grid, K=range):
    grid_size = 23
    y_value = 4
    pattern_coords = [(r, c) for r in K(grid_size) for c in K(grid_size) if grid[r][c] == 1]
    top_edge = min((r for r, _ in pattern_coords)) + 1
    left_edge = min((c for _, c in pattern_coords)) + 1
    bottom_edge = max((r for r, _ in pattern_coords)) - 1
    right_edge = max((c for _, c in pattern_coords)) - 1
    pattern_height = bottom_edge - top_edge + 1
    pattern_width = right_edge - left_edge + 1
    primary_pattern = [[grid[top_edge + r][left_edge + c] == y_value for c in K(pattern_width)] for r in K(pattern_height)]

    def rotate_90(matrix):
        return [list(x) for x in zip(*matrix[::-1])]

    def transpose(matrix):
        return [list(x) for x in zip(*matrix)]
    all_patterns = []
    current_pattern = primary_pattern
    for _ in K(4):
        all_patterns.extend([current_pattern, transpose(current_pattern)])
        current_pattern = rotate_90(current_pattern)
    output_grid = [[0] * grid_size for _ in K(grid_size)]
    for pattern in all_patterns:
        pattern_height, pattern_width = (len(pattern), len(pattern[0]))
        for row_offset in K(grid_size - pattern_height + 1):
            for col_offset in K(grid_size - pattern_width + 1):
                if all(((grid[row_offset + r][col_offset + c] == y_value) == pattern[r][c] for r in K(pattern_height) for c in K(pattern_width))):
                    start_row = max(0, row_offset - 1)
                    end_row = min(grid_size - 1, row_offset + pattern_height)
                    start_col = max(0, col_offset - 1)
                    end_col = min(grid_size - 1, col_offset + pattern_width)
                    for r in K(start_row, end_row + 1):
                        for c in K(start_col, end_col + 1):
                            output_grid[r][c] = 1
    for r in K(grid_size):
        for c in K(grid_size):
            if grid[r][c] == y_value:
                output_grid[r][c] = y_value
    return output_grid
