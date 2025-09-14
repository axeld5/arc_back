def p(input_grid, row_range=range, sum_func=sum):

    def find_region_coordinates():
        special_color = next((value for row in input_grid for value in row if value not in (BORDER_COLOR, RED_COLOR)), GREEN_COLOR)
        coordinates = [(row, col) for row in row_range(GRID_HEIGHT) for col in row_range(GRID_WIDTH) if input_grid[row][col] == special_color]
        return (special_color, coordinates)

    def calculate_red_borders():

        def left_borders():
            return sum_func((1 for row in row_range(min_row, max_row + 1) if min_col - 1 >= 0 and input_grid[row][min_col - 1] == RED_COLOR))

        def right_borders():
            return sum_func((1 for row in row_range(min_row, max_row + 1) if max_col + 1 < GRID_WIDTH and input_grid[row][max_col + 1] == RED_COLOR))

        def upper_borders():
            return sum_func((1 for col in row_range(min_col, max_col + 1) if min_row - 1 >= 0 and input_grid[min_row - 1][col] == RED_COLOR))

        def lower_borders():
            return sum_func((1 for col in row_range(min_col, max_col + 1) if max_row + 1 < GRID_HEIGHT and input_grid[max_row + 1][col] == RED_COLOR))
        return {'L': left_borders(), 'R': right_borders(), 'U': upper_borders(), 'D': lower_borders()}

    def reflect_region(reflection_direction):
        if reflection_direction == 'L':
            for row, col in region_coordinates:
                reflected_col = 2 * min_col - 1 - col
                output_grid[row][reflected_col] = special_color
        elif reflection_direction == 'R':
            for row, col in region_coordinates:
                reflected_col = 2 * max_col + 1 - col
                output_grid[row][reflected_col] = special_color
        elif reflection_direction == 'U':
            for row, col in region_coordinates:
                reflected_row = 2 * min_row - 1 - row
                output_grid[reflected_row][col] = special_color
        else:
            for row, col in region_coordinates:
                reflected_row = 2 * max_row + 1 - row
                output_grid[reflected_row][col] = special_color
    GRID_HEIGHT = 10
    GRID_WIDTH = 10
    BORDER_COLOR = 0
    RED_COLOR = 2
    GREEN_COLOR = 3
    special_color, region_coordinates = find_region_coordinates()
    min_row = min((row for row, _ in region_coordinates))
    max_row = max((row for row, _ in region_coordinates))
    min_col = min((col for _, col in region_coordinates))
    max_col = max((col for _, col in region_coordinates))
    red_borders = calculate_red_borders()
    optimal_reflection_direction = max(red_borders, key=red_borders.get)
    output_grid = [[GREEN_COLOR] * GRID_WIDTH for _ in row_range(GRID_HEIGHT)]
    for row, col in region_coordinates:
        output_grid[row][col] = special_color
    reflect_region(optimal_reflection_direction)
    return output_grid
