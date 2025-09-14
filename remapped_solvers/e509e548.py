from collections import deque

def p(grid, range_func=range):
    height, width = (len(grid), len(grid[0]))

    def find_regions():
        visited = set()
        regions = []
        for row in range_func(height):
            for col in range_func(width):
                if (row, col) in visited or grid[row][col] == 0:
                    continue
                region_value = grid[row][col]
                queue = deque([(row, col)])
                current_region = set()
                while queue:
                    r, c = queue.popleft()
                    if (r, c) in visited or grid[r][c] != region_value:
                        continue
                    visited.add((r, c))
                    current_region.add((r, c))
                    for rr, cc in [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]:
                        if 0 <= rr < height and 0 <= cc < width:
                            queue.append((rr, cc))
                regions.append(current_region)
        return regions

    def get_bounding_box(region):
        rows = [r for r, _ in region]
        cols = [c for _, c in region]
        return (min(rows), min(cols), max(rows), max(cols))

    def is_line(region):
        top, left, bottom, right = get_bounding_box(region)
        height = bottom - top + 1
        width = right - left + 1
        return len(region) == height + width - 1

    def has_internal_three(region):
        top, left, bottom, right = get_bounding_box(region)
        if bottom - top <= 1 or right - left <= 1:
            return False
        for row in range_func(top + 1, bottom):
            for col in range_func(left + 1, right):
                if grid[row][col] == 3:
                    return True
        return False
    all_regions = find_regions()
    regions_with_three = [region for region in all_regions if has_internal_three(region)]
    straight_line_regions = [region for region in all_regions if is_line(region)]
    for row in range_func(height):
        for col in range_func(width):
            if grid[row][col] == 3:
                grid[row][col] = 6
    for region in regions_with_three:
        for r, c in region:
            grid[r][c] = 2
    for region in straight_line_regions:
        for r, c in region:
            grid[r][c] = 1
    return grid
