def p(grid):
    GRID_SIZE = 20

    def dfs_extract_region(start_row, start_col):
        stack = [(start_row, start_col)]
        region = []
        visited[start_row][start_col] = True
        while stack:
            curr_row, curr_col = stack.pop()
            region.append((curr_row, curr_col))
            for d_row, d_col in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                new_row, new_col = (curr_row + d_row, curr_col + d_col)
                if 0 <= new_row < GRID_SIZE and 0 <= new_col < GRID_SIZE and (not visited[new_row][new_col]) and (grid[new_row][new_col] != 0):
                    visited[new_row][new_col] = True
                    stack.append((new_row, new_col))
        return region
    visited = [[False] * GRID_SIZE for _ in range(GRID_SIZE)]
    regions = []
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            if grid[row][col] != 0 and (not visited[row][col]):
                region = dfs_extract_region(row, col)
                regions.append(region)

    def count_target_value(cell):
        cell_row, cell_col = cell
        return grid[cell_row][cell_col] == 2
    largest_region = max(regions, key=lambda region: sum((count_target_value(cell) for cell in region)))
    row_indices = [r for r, _ in largest_region]
    col_indices = [c for _, c in largest_region]
    min_row, max_row = (min(row_indices), max(row_indices))
    min_col, max_col = (min(col_indices), max(col_indices))
    output_height = max_row - min_row + 1
    output_width = max_col - min_col + 1
    output_grid = [[1] * output_width for _ in range(output_height)]
    for r, c in largest_region:
        if grid[r][c] == 2:
            output_grid[r - min_row][c - min_col] = 2
    return output_grid
