from collections import Counter

def p(input_grid, range_func=range):

    def is_valid_center(start_row, start_col, step_size):
        for row, col in primary_cells:
            found = False
            for i in range_func(0, max(height, width) + 2):
                layer_width = step_size * i
                left = start_col - layer_width + (step_size - 1)
                right = start_col + layer_width - 1
                top = start_row - layer_width + (step_size - 1)
                bottom = start_row + layer_width - 1
                if col in (left, right) and top <= row <= bottom or (row in (top, bottom) and left <= col <= right):
                    found = True
                    break
            if not found:
                return False
        return True

    def fill_output_grid(start_row, start_col, step_size, output_grid):
        for i in range_func(0, max(height, width) + 2):
            layer_width = step_size * i
            left = start_col - layer_width + (step_size - 1)
            right = start_col + layer_width - 1
            top = start_row - layer_width + (step_size - 1)
            bottom = start_row + layer_width - 1
            if 0 <= left < width:
                for row in range_func(max(0, top), min(height, bottom + 1)):
                    output_grid[row][left] = most_common_value
            if 0 <= right < width:
                for row in range_func(max(0, top), min(height, bottom + 1)):
                    output_grid[row][right] = most_common_value
            if 0 <= top < height:
                for col in range_func(max(0, left), min(width, right + 1)):
                    output_grid[top][col] = most_common_value
            if 0 <= bottom < height:
                for col in range_func(max(0, left), min(width, right + 1)):
                    output_grid[bottom][col] = most_common_value
            if left >= width and right < 0 and (top >= height) and (bottom < 0):
                break
    height = len(input_grid)
    width = len(input_grid[0]) if input_grid else 0
    most_common_value = Counter((value for row in input_grid for value in row if value)).most_common(1)[0][0]
    primary_cells = [(row_index, col_index) for row_index in range_func(height) for col_index in range_func(width) if input_grid[row_index][col_index] == most_common_value]
    found_center = None
    for step_size in (2, 3):
        for row_start in range_func(height):
            if is_valid_center(row_start, 0, step_size):
                found_center = (row_start, 0, step_size)
                break
        if not found_center:
            for col_start in range_func(width):
                if is_valid_center(0, col_start, step_size):
                    found_center = (0, col_start, step_size)
                    break
        if found_center:
            break
    start_row, start_col, step_size = found_center if found_center else (0, 0, 2)
    output_grid = [[5] * width for _ in range_func(height)]
    fill_output_grid(start_row, start_col, step_size, output_grid)
    return output_grid
