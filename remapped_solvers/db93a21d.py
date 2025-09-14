def p(grid):
    height, width = (len(grid), len(grid[0]))
    visited = [[0] * width for _ in range(height)]
    regions = []
    region_id = 0

    def flood_fill(row, col):
        stack = [(row, col)]
        visited[row][col] = 1
        min_row = max_row = row
        min_col = max_col = col
        while stack:
            r, c = stack.pop()
            min_row = min(min_row, r)
            max_row = max(max_row, r)
            min_col = min(min_col, c)
            max_col = max(max_col, c)
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = (r + dr, c + dc)
                if 0 <= nr < height and 0 <= nc < width and (not visited[nr][nc]) and (grid[nr][nc] == 9):
                    visited[nr][nc] = 1
                    stack.append((nr, nc))
        box_width = max_col - min_col + 1
        offset = box_width // 2
        return (min_row, min_col, max_row, max_col, offset)
    for row in range(height):
        for col in range(width):
            if grid[row][col] == 9 and (not visited[row][col]):
                bounding_box = flood_fill(row, col)
                regions.append((*bounding_box, region_id))
                region_id += 1
    regions.sort(key=lambda region: (region[2], region[5]))
    output_grid = [[0] * width for _ in range(height)]
    for min_row, min_col, max_row, max_col, offset, _ in regions:
        region_height = max_row - min_row + 1
        extend_bottom = 2 * offset - region_height if max_row == height - 1 else 0
        row_end = max_row + extend_bottom
        for i in range(row_end + 1, height):
            for j in range(min_col, min(min_col + 2 * offset, width)):
                output_grid[i][j] = 1
        for i in range(4 * offset):
            row_inner = row_end + offset - i
            if 0 <= row_inner < height:
                for j in range(4 * offset):
                    col_inner = min_col - offset + j
                    if 0 <= col_inner < width:
                        output_grid[row_inner][col_inner] = 3
        for i in range(min_row, max_row + 1):
            for j in range(min_col, max_col + 1):
                output_grid[i][j] = 9
    return output_grid
