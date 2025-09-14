from itertools import chain

def transpose(grid):
    return [list(row) for row in zip(*grid)]

def find_unique_elements_positions(grid, empty_value):
    element_positions = {}
    for i, row in enumerate(grid):
        for j, value in enumerate(row):
            if value != empty_value:
                element_positions.setdefault(value, []).append((i, j))
    return element_positions

def calculate_step_size(positions):
    if positions:
        max_col = max((j for _, j in positions))
        min_col = min((j for _, j in positions))
        return max(2 * (max_col - min_col), 1)
    return 1

def fill_grid(grid, element_positions, step_size):
    height = len(grid)
    width = len(grid[0])
    for element, positions in element_positions.items():
        start_col = min((j for _, j in positions))
        for col in range(start_col, width, step_size):
            for row in range(height):
                grid[row][col] = element

def p(input_grid):
    height = len(input_grid)
    width = len(input_grid[0])
    empty_value = 0
    transpose_flag = height > width
    if transpose_flag:
        input_grid = transpose(input_grid)
        height, width = (width, height)
    element_positions = find_unique_elements_positions(input_grid, empty_value)
    all_positions = list(chain.from_iterable(element_positions.values()))
    step_size = calculate_step_size(all_positions)
    solution_grid = [row[:] for row in input_grid]
    fill_grid(solution_grid, element_positions, step_size)
    if transpose_flag:
        solution_grid = transpose(solution_grid)
    return solution_grid
