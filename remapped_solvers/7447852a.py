def update_value_based_on_cycle(value, mask):
    if mask:
        return 4
    return value

def update_grid_elements(grid, cycle_pattern):
    for index in range(len(grid)):
        cycle_index = index % len(cycle_pattern)
        mask = cycle_pattern[cycle_index]
        grid[index] = update_value_based_on_cycle(grid[index], mask)

def p(j):
    grid_A, grid_c, grid_E = j
    cycle_pattern = (6, 4, 0, 0, 0, 1, 3, 1, 0, 0, 0, 4)
    update_grid_elements(grid_A, [val & 1 for val in cycle_pattern])
    update_grid_elements(grid_c, [(val & 2) >> 1 for val in cycle_pattern])
    update_grid_elements(grid_E, [(val & 4) >> 2 for val in cycle_pattern])
    return j
