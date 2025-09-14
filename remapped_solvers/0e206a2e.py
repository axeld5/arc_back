from collections import *

def p(I):
    H, W = (len(I), len(I[0]))
    R = range

    def rotate_coords(r, c, height, width, rotation):
        rotations = [(width - c - 1, r), (c, height - r - 1), (height - r - 1, c), (c, r)]
        return rotations[rotation - 1]
    non_zero_positions = {(r, c): I[r][c] for r in R(H) for c in R(W) if I[r][c]}
    most_common_value = Counter(non_zero_positions.values()).most_common(1)[0][0]
    visited = [[0] * W for _ in R(H)]
    components = []
    for start_row in R(H):
        for start_col in R(W):
            if not I[start_row][start_col] or visited[start_row][start_col]:
                continue
            queue = deque([(start_row, start_col)])
            visited[start_row][start_col] = 1
            component = []
            while queue:
                r, c = queue.popleft()
                component.append((r, c))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = (r + dr, c + dc)
                    if 0 <= nr < H and 0 <= nc < W and I[nr][nc] and (not visited[nr][nc]):
                        visited[nr][nc] = 1
                        queue.append((nr, nc))
            components.append(component)
    patterns = []
    free_positions = set()
    shapes = []
    for component in components:
        has_common_value = any((non_zero_positions[r, c] == most_common_value for r, c in component))
        if has_common_value:
            patterns.append(component)
        else:
            free_positions.update(component)
    for pattern in patterns:
        rows = [r for r, _ in pattern]
        cols = [c for _, c in pattern]
        min_row, min_col = (min(rows), min(cols))
        normalized = [(r - min_row, c - min_col, non_zero_positions[r, c]) for r, c in pattern]
        height = 1 + max((r for r, _, _ in normalized))
        width = 1 + max((c for _, c, _ in normalized))
        pattern_positions = [pos for pos in normalized if pos[2] != most_common_value]
        shapes.append({'full': normalized, 'pattern': pattern_positions, 'height': height, 'width': width})
    free_by_value = defaultdict(list)
    for r, c in free_positions:
        free_by_value[non_zero_positions[r, c]].append((r, c))
    used_positions = set()
    placements = []
    for shape in shapes:
        height, width = (shape['height'], shape['width'])
        pattern_found = False
        for rotation in (1, 2, 3, 4):
            if pattern_found:
                break
            rotated_pattern = []
            for r, c, value in shape['pattern']:
                new_r, new_c = rotate_coords(r, c, height, width, rotation)
                rotated_pattern.append((new_r, new_c, value))
            for pattern_r, pattern_c, pattern_value in rotated_pattern:
                if pattern_found:
                    break
                for free_r, free_c in free_by_value.get(pattern_value, []):
                    if (free_r, free_c) in used_positions:
                        continue
                    offset_r = free_r - pattern_r
                    offset_c = free_c - pattern_c
                    matching_positions = []
                    valid_match = True
                    for pr, pc, pv in rotated_pattern:
                        target_r = offset_r + pr
                        target_c = offset_c + pc
                        if (target_r, target_c) not in non_zero_positions or non_zero_positions[target_r, target_c] != pv:
                            valid_match = False
                            break
                        matching_positions.append((target_r, target_c))
                    if valid_match:
                        used_positions.update(matching_positions)
                        placements.append((shape, rotation, offset_r, offset_c))
                        pattern_found = True
                        break
        if not pattern_found:
            placements.append((shape, 4, 0, 0))
    output = [[0] * W for _ in R(H)]
    for shape, rotation, offset_r, offset_c in placements:
        height, width = (shape['height'], shape['width'])
        for r, c, value in shape['full']:
            rotated_r, rotated_c = rotate_coords(r, c, height, width, rotation)
            final_r = offset_r + rotated_r
            final_c = offset_c + rotated_c
            if 0 <= final_r < H and 0 <= final_c < W:
                output[final_r][final_c] = value
    return output
