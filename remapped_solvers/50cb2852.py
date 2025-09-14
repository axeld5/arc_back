def p(input_grid):

    def get_dimensions(grid):
        return (len(grid), len(grid[0]))

    def initialize_empty_grid(grid):
        return [row[:] for row in grid]

    def is_within_bounds(row, col, row_count, col_count):
        return 0 <= row < row_count and 0 <= col < col_count

    def explore_region(start_row, start_col, initial_value, visited, grid):
        region = [(start_row, start_col)]
        stack = [(start_row, start_col)]
        row_count, col_count = get_dimensions(grid)
        while stack:
            current_row, current_col = stack.pop()
            for row_offset, col_offset in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                neighbor_row, neighbor_col = (current_row + row_offset, current_col + col_offset)
                if is_within_bounds(neighbor_row, neighbor_col, row_count, col_count) and grid[neighbor_row][neighbor_col] == initial_value and ((neighbor_row, neighbor_col) not in visited):
                    visited.add((neighbor_row, neighbor_col))
                    stack.append((neighbor_row, neighbor_col))
                    region.append((neighbor_row, neighbor_col))
        return region

    def fill_internal_region(region, output_grid):
        min_row = min((r for r, c in region))
        max_row = max((r for r, c in region))
        min_col = min((c for r, c in region))
        max_col = max((c for r, c in region))
        for row in range(min_row + 1, max_row):
            for col in range(min_col + 1, max_col):
                output_grid[row][col] = 8
    row_count, col_count = get_dimensions(input_grid)
    output_grid = initialize_empty_grid(input_grid)
    visited = set()
    for row in range(row_count):
        for col in range(col_count):
            if input_grid[row][col] != 0 and (row, col) not in visited:
                visited.add((row, col))
                region_value = input_grid[row][col]
                region = explore_region(row, col, region_value, visited, input_grid)
                fill_internal_region(region, output_grid)
    return output_grid
