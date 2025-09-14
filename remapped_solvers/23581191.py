def initialize_grid(size):
    return [[0] * size for _ in range(size)]

def extract_nonzero_elements(grid):
    nonzero_elements = []
    for row_index in range(len(grid)):
        for col_index in range(len(grid[row_index])):
            if grid[row_index][col_index] != 0:
                nonzero_elements.append((row_index, col_index, grid[row_index][col_index]))
    return nonzero_elements

def process_nonzero_elements(grid, size, nonzero_elements):
    for row, col, value in nonzero_elements:
        for i in range(size):
            grid[row][i] = value
            grid[i][col] = value
    return grid

def set_pairs_in_grid(grid, nonzero_elements):
    grid[nonzero_elements[0][0]][nonzero_elements[1][1]] = 2
    grid[nonzero_elements[1][0]][nonzero_elements[0][1]] = 2
    return grid

def p(input_grid):
    size = 9
    result_grid = initialize_grid(size)
    nonzero_elements = extract_nonzero_elements(input_grid)
    result_grid = process_nonzero_elements(result_grid, size, nonzero_elements)
    result_grid = set_pairs_in_grid(result_grid, nonzero_elements)
    return result_grid
