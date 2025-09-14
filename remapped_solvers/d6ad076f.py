def p(grid, size=10):
    visited = [[False] * size for _ in range(size)]
    rectangles = []

    def flood_fill(start_row, start_col):
        value = grid[start_row][start_col]
        stack = [(start_row, start_col)]
        cells = []
        visited[start_row][start_col] = True
        min_row = max_row = start_row
        min_col = max_col = start_col
        while stack:
            row, col = stack.pop()
            cells.append((row, col))
            min_row = min(min_row, row)
            max_row = max(max_row, row)
            min_col = min(min_col, col)
            max_col = max(max_col, col)
            for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                new_row, new_col = (row + dr, col + dc)
                if 0 <= new_row < size and 0 <= new_col < size and (not visited[new_row][new_col]) and (grid[new_row][new_col] == value):
                    visited[new_row][new_col] = True
                    stack.append((new_row, new_col))
        return (value, cells, (min_row, min_col, max_row, max_col))
    for i in range(size):
        for j in range(size):
            if grid[i][j] and (not visited[i][j]):
                value, cells, (min_row, min_col, max_row, max_col) = flood_fill(i, j)
                expected_area = (max_row - min_row + 1) * (max_col - min_col + 1)
                if expected_area == len(cells):
                    rectangles.append((min_row, min_col, max_row, max_col, value))
    rectangles.sort(key=lambda x: (x[2] - x[0] + 1) * (x[3] - x[1] + 1), reverse=True)
    rect1, rect2 = (rectangles[0], rectangles[1])

    def get_dimensions(rect):
        min_row, min_col, max_row, max_col, _ = rect
        return (max_row - min_row + 1, max_col - min_col + 1)
    height1, width1 = get_dimensions(rect1)
    height2, width2 = get_dimensions(rect2)
    orient1 = 'horizontal' if width1 > height1 else 'vertical'
    orient2 = 'horizontal' if width2 > height2 else 'vertical'
    if orient1 == orient2:
        orientation = orient1
    else:
        orientation = 'horizontal' if width1 + height1 >= width2 + height2 else orient2
    size1 = width1 if orientation == 'horizontal' else height1
    size2 = width2 if orientation == 'horizontal' else height2
    smaller_rect, larger_rect = (rect1, rect2) if size1 <= size2 else (rect2, rect1)
    s_min_row, s_min_col, s_max_row, s_max_col, _ = smaller_rect
    l_min_row, l_min_col, l_max_row, l_max_col, _ = larger_rect
    if orientation == 'horizontal':
        line_col_start = s_min_col + 1
        line_col_end = s_max_col - 1
        if s_min_row < l_min_row:
            line_row_start = s_max_row + 1
            line_row_end = l_min_row - 1
        else:
            line_row_start = l_max_row + 1
            line_row_end = s_min_row - 1
        for row in range(max(line_row_start, 0), min(line_row_end, size - 1) + 1):
            for col in range(max(line_col_start, 0), min(line_col_end, size - 1) + 1):
                grid[row][col] = 8
    else:
        line_row_start = s_min_row + 1
        line_row_end = s_max_row - 1
        if s_min_col < l_min_col:
            line_col_start = s_max_col + 1
            line_col_end = l_min_col - 1
        else:
            line_col_start = l_max_col + 1
            line_col_end = s_min_col - 1
        for row in range(max(line_row_start, 0), min(line_row_end, size - 1) + 1):
            for col in range(max(line_col_start, 0), min(line_col_end, size - 1) + 1):
                grid[row][col] = 8
    return grid
