def propagate_water_flow(grid, column_index):
    water_indicator = 2
    obstacle_indicator = 5
    offset = 0
    for row in range(10):
        if grid[-(row + 1)][column_index + offset] == obstacle_indicator:
            offset += 1
            grid[-row][column_index + offset] = water_indicator
        grid[-(row + 1)][column_index + offset] = water_indicator

def p(grid, R=range):
    water_indicator = 2
    for column_index in R(10):
        if grid[-1][column_index] == water_indicator:
            propagate_water_flow(grid, column_index)
    return grid
