def p(grid, index_range=range, length=len):
    height, width = (length(grid), length(grid[0]))

    def transpose(matrix):
        return [list(row) for row in zip(*matrix)]

    def draw_path(sub_grid):
        height, width = (length(sub_grid), length(sub_grid[0]))
        path_grid = [[0] * width for _ in index_range(height)]
        non_zero_points = [(r, c, sub_grid[r][c]) for r in index_range(height) for c in index_range(width) if sub_grid[r][c] != 0]
        (start_row, start_col, start_val), (end_row, end_col, end_val) = non_zero_points
        horizontal_row = start_row
        if start_col <= end_col:
            left_col, left_val, right_col, right_val = (start_col, start_val, end_col, end_val)
        else:
            left_col, left_val, right_col, right_val = (end_col, end_val, start_col, start_val)
        half_distance = (abs(right_col - left_col) + 1) // 2

        def draw_half_point_path(start, color, increment):
            for step in index_range(half_distance - 1):
                current_col = start + increment * step
                if 0 <= current_col < width:
                    path_grid[horizontal_row][current_col] = color
            end_col = start + increment * (half_distance - 2)
            for delta in index_range(-2, 3):
                current_row = horizontal_row + delta
                if 0 <= current_row < height:
                    path_grid[current_row][end_col] = color
            next_col = end_col + increment
            if 0 <= next_col < width:
                if 0 <= horizontal_row - 2 < height:
                    path_grid[horizontal_row - 2][next_col] = color
                if 0 <= horizontal_row + 2 < height:
                    path_grid[horizontal_row + 2][next_col] = color
        draw_half_point_path(left_col, left_val, 1)
        draw_half_point_path(right_col, right_val, -1)
        return path_grid
    non_zero_coords = [(r, c) for r in index_range(height) for c in index_range(width) if grid[r][c] != 0]
    (point1_row, point1_col), (point2_row, point2_col) = non_zero_coords
    if point1_row == point2_row:
        return draw_path(grid)
    else:
        transposed_grid = transpose(grid)
        modified_grid = draw_path(transposed_grid)
        return transpose(modified_grid)
