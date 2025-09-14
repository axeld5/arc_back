def p(grid):

    def find_largest_rectangle_with_border(H, W):
        largest_section = ()
        max_area = 0
        for row in range(H):
            for col in range(W):
                border_value = grid[row][col]
                for height in range(6, 11):
                    bottom = row + height - 1
                    if bottom >= H:
                        break
                    for width in range(6, 11):
                        right = col + width - 1
                        if right >= W:
                            break
                        if check_border(row, col, bottom, right, border_value) and check_interior(row, col, bottom, right, border_value):
                            if height * width > max_area:
                                largest_section = (row, col, height, width, border_value)
                                max_area = height * width
        return largest_section

    def check_border(top, left, bottom, right, value):
        for x in range(left, right + 1):
            if grid[top][x] != value or grid[bottom][x] != value:
                return False
        for y in range(top + 1, bottom):
            if grid[y][left] != value or grid[y][right] != value:
                return False
        return True

    def check_interior(top, left, bottom, right, value):
        found_different_value = False
        for y in range(top + 1, bottom):
            for x in range(left + 1, right):
                if grid[y][x] != value:
                    found_different_value = True
                    break
            if found_different_value:
                break
        return found_different_value

    def replace_largest_section_with_value(top, left, height, width, border_value):
        section = [grid[r][left:left + width] for r in range(top, top + height)]
        fill_value = None
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                if section[y][x] != border_value:
                    fill_value = section[y][x]
                    break
            if fill_value is not None:
                break
        fill_mask_row = [0] * height
        fill_mask_col = [0] * width
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                if section[y][x] == fill_value:
                    fill_mask_row[y] = 1
                    fill_mask_col[x] = 1
        for y in range(height):
            for x in range(width):
                if fill_mask_row[y] or fill_mask_col[x]:
                    section[y][x] = fill_value
        return section
    H, W = (len(grid), len(grid[0]))
    largest_section = find_largest_rectangle_with_border(H, W)
    if not largest_section:
        return grid
    top, left, height, width, border_value = largest_section
    return replace_largest_section_with_value(top, left, height, width, border_value)
