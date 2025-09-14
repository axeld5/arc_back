from collections import deque

def p(grid):
    height, width = (len(grid), len(grid[0]))
    visited = [[False] * width for _ in range(height)]
    components = []
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    for row in range(height):
        for col in range(width):
            if visited[row][col] or grid[row][col] == 0:
                continue
            value = grid[row][col]
            queue = deque([(row, col)])
            visited[row][col] = True
            cells = []
            while queue:
                r, c = queue.popleft()
                cells.append((r, c))
                for dr, dc in directions:
                    nr, nc = (r + dr, c + dc)
                    if 0 <= nr < height and 0 <= nc < width and (not visited[nr][nc]) and (grid[nr][nc] == value):
                        visited[nr][nc] = True
                        queue.append((nr, nc))
            rows = [r for r, c in cells]
            cols = [c for r, c in cells]
            min_row, max_row = (min(rows), max(rows))
            min_col, max_col = (min(cols), max(cols))
            top_edges = {}
            bottom_edges = {}
            left_edges = {}
            right_edges = {}
            for r, c in cells:
                if c not in top_edges or r < top_edges[c]:
                    top_edges[c] = r
                if c not in bottom_edges or r > bottom_edges[c]:
                    bottom_edges[c] = r
                if r not in left_edges or c < left_edges[r]:
                    left_edges[r] = c
                if r not in right_edges or c > right_edges[r]:
                    right_edges[r] = c
            center = ((min_row + max_row) // 2, (min_col + max_col) // 2)
            components.append({'value': value, 'center': center, 'top': top_edges, 'bottom': bottom_edges, 'left': left_edges, 'right': right_edges})
    components_2 = [comp for comp in components if comp['value'] == 2]
    components_8 = [comp for comp in components if comp['value'] == 8]
    connection_cells = set()
    target_cells = []
    for comp_2 in components_2:
        for comp_8 in components_8:
            center_2 = comp_2['center']
            center_8 = comp_8['center']
            best_vertical_gap = None
            best_vertical_col = None
            best_horizontal_gap = None
            best_horizontal_row = None
            vertical_up = center_2[0] < center_8[0]
            edges_2 = comp_2['bottom'] if vertical_up else comp_2['top']
            edges_8 = comp_8['top'] if vertical_up else comp_8['bottom']
            for col in set(edges_2.keys()) & set(edges_8.keys()):
                pos_2, pos_8 = (edges_2[col], edges_8[col])
                if abs(pos_8 - pos_2) >= 2 and all((grid[r][col] == 0 for r in range(min(pos_2, pos_8) + 1, max(pos_2, pos_8)))):
                    gap = abs(pos_8 - pos_2) - 1
                    if best_vertical_gap is None or gap < best_vertical_gap:
                        best_vertical_gap, best_vertical_col = (gap, col)
            horizontal_left = center_2[1] < center_8[1]
            edges_2 = comp_2['right'] if horizontal_left else comp_2['left']
            edges_8 = comp_8['left'] if horizontal_left else comp_8['right']
            for row in set(edges_2.keys()) & set(edges_8.keys()):
                pos_2, pos_8 = (edges_2[row], edges_8[row])
                if abs(pos_8 - pos_2) >= 2 and all((grid[row][c] == 0 for c in range(min(pos_2, pos_8) + 1, max(pos_2, pos_8)))):
                    gap = abs(pos_8 - pos_2) - 1
                    if best_horizontal_gap is None or gap < best_horizontal_gap:
                        best_horizontal_gap, best_horizontal_row = (gap, row)
            if best_vertical_gap is None and best_horizontal_gap is None:
                continue
            if best_horizontal_gap is None or (best_vertical_gap is not None and best_vertical_gap <= best_horizontal_gap):
                col = best_vertical_col
                if vertical_up:
                    target_pos = comp_8['top'][col]
                    start_pos = comp_2['bottom'][col] + 1
                    end_pos = target_pos - 1
                else:
                    target_pos = comp_8['bottom'][col]
                    start_pos = target_pos + 1
                    end_pos = comp_2['top'][col] - 1
                if end_pos >= start_pos:
                    connection_cells.update(((r, col) for r in range(start_pos, end_pos + 1)))
                target_cells.append((target_pos, col))
            else:
                row = best_horizontal_row
                if horizontal_left:
                    target_pos = comp_8['left'][row]
                    start_pos = comp_2['right'][row] + 1
                    end_pos = target_pos - 1
                else:
                    target_pos = comp_8['right'][row]
                    start_pos = target_pos + 1
                    end_pos = comp_2['left'][row] - 1
                if end_pos >= start_pos:
                    connection_cells.update(((row, c) for c in range(start_pos, end_pos + 1)))
                target_cells.append((row, target_pos))
    result = [row[:] for row in grid]
    for r, c in connection_cells:
        result[r][c] = 2
    for r, c in target_cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = (r + dr, c + dc)
                if 0 <= nr < height and 0 <= nc < width:
                    result[nr][nc] = 8
        result[r][c] = 2
    return result
