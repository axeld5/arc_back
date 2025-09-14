def p(input_grid, range_function=range):

    def initialize_output_grid(height, width):
        return [[0] * width for _ in range_function(height)]

    def apply_transformation(input_grid, output_grid, center_value, height, width):
        for row in range_function(len(input_grid)):
            for col in range_function(len(input_grid)):
                value = input_grid[row][col]
                if value and value != center_value and (row < height) and (col < width):
                    output_grid[row][col] = center_value
                    output_grid[row][width - col - 1] = center_value
                    output_grid[height - row - 1][col] = center_value
                    output_grid[height - row - 1][width - col - 1] = center_value
    grid_size = len(input_grid)
    half_size = (grid_size - 1) // 2
    height = width = 2 * half_size
    center_value = input_grid[half_size][half_size]
    output_grid = initialize_output_grid(height, width)
    apply_transformation(input_grid, output_grid, center_value, height, width)
    return output_grid
