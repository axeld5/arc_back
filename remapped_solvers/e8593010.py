def p(grid):

    def is_in_bounds(x, y):
        return 0 <= x < 10 and 0 <= y < 10

    def explore_component(start_x, start_y):
        if (start_x, start_y) in visited or not is_in_bounds(start_x, start_y) or grid[start_x][start_y] != 0:
            return []
        visited.add((start_x, start_y))
        component = [(start_x, start_y)]
        for delta_x, delta_y in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            component += explore_component(start_x + delta_x, start_y + delta_y)
        return component
    visited = set()
    result_grid = [row[:] for row in grid]
    for row in range(10):
        for col in range(10):
            if grid[row][col] == 0 and (row, col) not in visited:
                component = explore_component(row, col)
                component_size = abs(len(component) - 4)
                for x, y in component:
                    result_grid[x][y] = component_size
    return result_grid
