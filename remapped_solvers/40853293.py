def p(grid):

    def collect_coordinates(grid):
        coordinates = {}
        for i, row in enumerate(grid):
            for j, value in enumerate(row):
                coordinates.setdefault(value, []).append((i, j))
        return coordinates

    def find_continuous_segments(coordinates):
        horizontal_segments, vertical_segments = ([], [])
        for value, cells in coordinates.items():
            rows = [row for row, _ in cells]
            cols = [col for _, col in cells]
            min_row, max_row = (min(rows), max(rows))
            min_col, max_col = (min(cols), max(cols))
            if max_row == min_row:
                horizontal_segments.append((value, min_row, min_col, max_col))
            if max_col == min_col:
                vertical_segments.append((value, min_col, min_row, max_row))
        return (horizontal_segments, vertical_segments)

    def fill_segments(grid, horizontal_segments, vertical_segments):
        for value, row, start_col, end_col in horizontal_segments:
            for col in range(start_col, end_col + 1):
                grid[row][col] = value
        for value, col, start_row, end_row in vertical_segments:
            for row in range(start_row, end_row + 1):
                grid[row][col] = value
        return grid
    output_grid = [row[:] for row in grid]
    coordinates = collect_coordinates(grid)
    horizontal_segments, vertical_segments = find_continuous_segments(coordinates)
    return fill_segments(output_grid, horizontal_segments, vertical_segments)
