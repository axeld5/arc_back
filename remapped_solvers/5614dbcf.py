def most_frequent_element(subgrid):
    frequency_dict = {}
    for row in subgrid:
        for element in row:
            frequency_dict[element] = frequency_dict.get(element, 0) + 1
    return max(frequency_dict, key=frequency_dict.get)

def extract_subgrid(grid, start_row, start_col):
    return [row[start_col:start_col + 3] for row in grid[start_row:start_row + 3]]

def p(grid):
    grid_size = 3
    solution_grid = [[0] * grid_size for _ in range(grid_size)]
    for subgrid_row in range(grid_size):
        for subgrid_col in range(grid_size):
            start_row, start_col = (subgrid_row * grid_size, subgrid_col * grid_size)
            subgrid = extract_subgrid(grid, start_row, start_col)
            solution_grid[subgrid_row][subgrid_col] = most_frequent_element(subgrid)
    return solution_grid
