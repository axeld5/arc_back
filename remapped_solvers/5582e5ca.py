def find_flattened_mode(grid):
    flattened_list = sum(grid, [])
    most_frequent_element = max(flattened_list, key=flattened_list.count)
    return most_frequent_element

def construct_uniform_grid(element, size=3):
    return [[element] * size] * size

def solve(grid):
    mode_element = find_flattened_mode(grid)
    uniform_grid = construct_uniform_grid(mode_element)
    return uniform_grid
p = solve
