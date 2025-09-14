def solve(matrix, range_generator=range):

    def initialize_output_matrix(height, width):
        return [[0] * width for _ in range(height)]

    def extract_non_zero_positions(grid, height, width):
        return [(row, col, grid[row][col]) for row in range_generator(height) for col in range_generator(width) if grid[row][col] != 0]

    def spread_value(matrix, output_matrix, row, col, value):
        for col_index in range_generator(col, len(matrix[0])):
            output_matrix[row][col_index] = value
        for row_index in range_generator(row, len(matrix)):
            output_matrix[row_index][len(matrix[0]) - 1] = value
    height, width = (len(matrix), len(matrix[0]))
    output_matrix = initialize_output_matrix(height, width)
    non_zero_positions = extract_non_zero_positions(matrix, height, width)
    for row, col, value in non_zero_positions:
        spread_value(matrix, output_matrix, row, col, value)
    return output_matrix
p = solve
