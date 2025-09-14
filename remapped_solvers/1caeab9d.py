def p(grid):

    def initialize_structure_grid(height, width):
        return [[0] * width for _ in range(height)]

    def gather_positions_of_numbers(grid, numbers):
        positions = {number: [] for number in numbers}
        for row in range(len(grid)):
            for col in range(len(grid[0])):
                value = grid[row][col]
                if value in positions:
                    positions[value].append((row, col))
        return positions

    def get_topmost_row_for_number(positions, number):
        return min((row for row, _ in positions[number]))
    NUMBERS = [1, 2, 4]
    height, width = (len(grid), len(grid[0]))
    positions = gather_positions_of_numbers(grid, NUMBERS)
    top_row_for_1 = get_topmost_row_for_number(positions, 1)
    output_grid = initialize_structure_grid(height, width)
    for number in NUMBERS:
        top_row_for_number = get_topmost_row_for_number(positions, number)
        vertical_shift = top_row_for_1 - top_row_for_number
        for row, col in positions[number]:
            output_grid[row + vertical_shift][col] = number
    return output_grid
