def rotate_90_clockwise(matrix):
    return [list(row) for row in zip(*matrix[::-1])]

def rotate_90_counterclockwise(matrix):
    return rotate_90_clockwise([row[::-1] for row in matrix[::-1]])

def max_with_transposed_and_rotated(grid):
    rotated = rotate_90_counterclockwise(grid)
    max_transposed = [[max(a, b) for a, b in zip(row, col)] for row, col in zip(grid, list(zip(*grid)))]
    max_result = [[max(a, b) for a, b in zip(row, col)] for row, col in zip(max_transposed, rotated)]
    return max_result

def p(grid):
    cleaned_grid = [[0 if value == 4 else value for value in row] for row in grid]
    transformed_grid = max_with_transposed_and_rotated(cleaned_grid)
    final_result = rotate_90_counterclockwise(transformed_grid)
    return final_result
