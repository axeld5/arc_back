from collections import deque, Counter

def p(grid, range_=range):

    def is_within_bounds(x, y):
        return 0 <= x < height and 0 <= y < width

    def bfs_collect_block(start_x, start_y, block_value):
        queue = deque([(start_x, start_y)])
        visited[start_x][start_y] = True
        block_cells = []
        while queue:
            x, y = queue.popleft()
            block_cells.append((x, y))
            for delta_x, delta_y in directions:
                adjacent_x, adjacent_y = (x + delta_x, y + delta_y)
                if is_within_bounds(adjacent_x, adjacent_y) and (not visited[adjacent_x][adjacent_y]) and (grid[adjacent_x][adjacent_y] == block_value):
                    visited[adjacent_x][adjacent_y] = True
                    queue.append((adjacent_x, adjacent_y))
        return block_cells

    def modify_grid_boundaries(block_cells):
        row_indices, col_indices = zip(*block_cells)
        min_row, max_row = (min(row_indices), max(row_indices))
        min_col, max_col = (min(col_indices), max(col_indices))
        boundary_cells = [(i, j) for i in range_(min_row, max_row + 1) for j in range_(min_col, max_col + 1) if (i, j) not in block_cells]
        if not boundary_cells:
            return
        counter = Counter((grid[i][j] for i, j in boundary_cells))
        most_common_diff_value = min(counter, key=counter.get)
        for i in range_(min_row, max_row + 1):
            for j in range_(min_col, max_col + 1):
                if grid[i - 1][j] == base_value:
                    modified_grid[i - 1][j] = most_common_diff_value
    height, width = (len(grid), len(grid[0]))
    base_value = 0
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    visited = [[False] * width for _ in range(height)]
    blocks_of_ones = []
    for i in range_(height):
        for j in range_(width):
            if not visited[i][j] and grid[i][j] != base_value:
                block_value = grid[i][j]
                block_cells = bfs_collect_block(i, j, block_value)
                if block_value == 1:
                    blocks_of_ones.append(block_cells)
    modified_grid = [row[:] for row in grid]
    for block_cells in blocks_of_ones:
        modify_grid_boundaries(block_cells)
    return modified_grid
