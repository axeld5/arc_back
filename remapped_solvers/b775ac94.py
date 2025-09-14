def p(grid, range_function=range):
    grid_size = len(grid)
    output_grid = [row[:] for row in grid]
    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    def is_within_bounds(row, col):
        return 0 <= row < grid_size and 0 <= col < grid_size

    def flood_fill(row, col, value):
        points_set = {(row, col)}
        stack = [(row, col)]
        while stack:
            current_row, current_col = stack.pop()
            for d_row, d_col in directions:
                new_row, new_col = (current_row + d_row, current_col + d_col)
                if is_within_bounds(new_row, new_col) and (new_row, new_col) not in points_set and (grid[new_row][new_col] == value):
                    points_set.add((new_row, new_col))
                    stack.append((new_row, new_col))
        return points_set
    quadrants = {'U': (-1, -1), 'R': (-1, 1), 'L': (1, -1), 'V': (1, 1)}
    swaps = []
    for row in range_function(grid_size - 1):
        for col in range_function(grid_size - 1):
            square_values = [grid[row][col], grid[row][col + 1], grid[row + 1][col], grid[row + 1][col + 1]]
            if len(set(square_values)) == 4:
                swaps.append({'U': ((row, col), grid[row][col]), 'R': ((row, col + 1), grid[row][col + 1]), 'L': ((row + 1, col), grid[row + 1][col]), 'V': ((row + 1, col + 1), grid[row + 1][col + 1])})
    transformations = []
    for swap in swaps:
        candidates = [(key,) + swap[key] for key in ('U', 'R', 'L', 'V') if swap[key][1] != 0]
        main_candidate = None
        for direction, (start_row, start_col), value in candidates:
            connected_component = flood_fill(start_row, start_col, value)
            if not 3 <= len(connected_component) <= 10:
                continue
            rows = [r for r, c in connected_component]
            cols = [c for r, c in connected_component]
            height, width = (max(rows) - min(rows) + 1, max(cols) - min(cols) + 1)
            if height > 5 or width > 5:
                continue
            if not (start_row in (min(rows), max(rows)) or start_col in (min(cols), max(cols))):
                continue
            main_candidate = (direction, (start_row, start_col), value, [(r - start_row, c - start_col) for r, c in connected_component])
            break
        if not main_candidate:
            continue
        direction, (start_row, start_col), value, offsets = main_candidate
        base_d_row, base_d_col = quadrants[direction]
        for other_direction, (other_start_row, other_start_col), other_value in candidates:
            if other_direction == direction:
                continue
            d_row, d_col = quadrants[other_direction]
            for delta_row, delta_col in offsets:
                new_row = other_start_row + (-delta_row if base_d_row != d_row else delta_row)
                new_col = other_start_col + (-delta_col if base_d_col != d_col else delta_col)
                if is_within_bounds(new_row, new_col):
                    transformations.append((new_row, new_col, other_value))
    for row, col, value in transformations:
        output_grid[row][col] = value
    return output_grid
