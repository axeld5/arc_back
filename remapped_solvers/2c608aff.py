from collections import Counter, deque

def p(grid, K=range, X=len, E=enumerate):
    height, width = (X(grid), X(grid[0]))
    element_count = Counter((value for row in grid for value in row))
    most_common_element = element_count.most_common(1)[0][0]
    least_common_element = min(element_count, key=element_count.get)
    visited = [[0] * width for _ in K(height)]
    largest_block = set()

    def bfs_find_block(start_row, start_col):
        value = grid[start_row][start_col]
        queue = deque([(start_row, start_col)])
        block = {(start_row, start_col)}
        visited[start_row][start_col] = 1
        while queue:
            i, j = queue.popleft()
            for x, y in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)):
                if 0 <= x < height and 0 <= y < width and (not visited[x][y]) and (grid[x][y] == value):
                    visited[x][y] = 1
                    block.add((x, y))
                    queue.append((x, y))
        return block
    for i in K(height):
        for j in K(width):
            if not visited[i][j] and grid[i][j] != most_common_element:
                current_block = bfs_find_block(i, j)
                if len(current_block) > len(largest_block):
                    largest_block = current_block
    least_common_coordinates = {(i, j) for i, row in E(grid) for j, value in E(row) if value == least_common_element}
    connection_paths = set()
    for a, b in largest_block:
        for c, d in least_common_coordinates:
            if a == c:
                connection_paths |= {(a, j) for j in K(min(b, d), max(b, d) + 1)}
            if b == d:
                connection_paths |= {(i, b) for i in K(min(a, c), max(a, c) + 1)}
    output_grid = [row[:] for row in grid]
    for i, j in connection_paths:
        if output_grid[i][j] == most_common_element:
            output_grid[i][j] = least_common_element
    return output_grid
