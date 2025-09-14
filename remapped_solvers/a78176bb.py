def place_number_in_grid(number, position_offset, grid, grid_size):
    for row in range(grid_size):
        column = row - position_offset
        if 0 <= column < grid_size:
            grid[row][column] = number

def find_unique_number_offsets(input_grid, grid_size):
    existing_numbers = {value for row in input_grid for value in row if value}
    offsets_dictionary = {value: {row - column for row in range(grid_size) for column in range(grid_size) if input_grid[row][column] == value} for value in existing_numbers}
    return (offsets_dictionary, existing_numbers)

def p(input_grid, grid_range=range(10)):
    grid_size = len(grid_range)
    offsets_dictionary, existing_numbers = find_unique_number_offsets(input_grid, grid_size)
    number_with_min_offsets = min(offsets_dictionary, key=lambda number: len(offsets_dictionary[number]))
    other_number = (existing_numbers - {number_with_min_offsets}).pop()
    base_offset = next(iter(offsets_dictionary[number_with_min_offsets]))
    max_offset = max((row - column for row in grid_range for column in grid_range if input_grid[row][column] == other_number and row - column > base_offset), default=None)
    min_offset = min((row - column for row in grid_range for column in grid_range if input_grid[row][column] == other_number and row - column < base_offset), default=None)
    output_grid = [[0] * grid_size for _ in input_grid]
    place_number_in_grid(number_with_min_offsets, base_offset, output_grid, grid_size)
    if max_offset is not None:
        place_number_in_grid(number_with_min_offsets, max_offset + 2, output_grid, grid_size)
    if min_offset is not None:
        place_number_in_grid(number_with_min_offsets, min_offset - 2, output_grid, grid_size)
    return output_grid
