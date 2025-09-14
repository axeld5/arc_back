def get_red_square_coordinates(grid, height, width, red_code=2):
    return [(r, c) for r in range(height) for c in range(width) if grid[r][c] == red_code]

def get_non_black_non_red_coordinates(grid, height, width, red_code=2, black_code=0):
    return [(r, c) for r in range(height) for c in range(width) if grid[r][c] not in (black_code, red_code)]

def get_unique_color_from_coordinates(grid, coordinates, exclude_colors):
    colors = set((grid[r][c] for r, c in coordinates))
    unique_colors = colors - exclude_colors
    assert len(unique_colors) == 1, 'There should be exactly one unique color.'
    return unique_colors.pop()

def create_empty_border(border_size, fill_color):
    return [[fill_color] * border_size for _ in range(border_size)]

def draw_border(grid, size, color):
    for i in range(size):
        grid[0][i] = grid[size - 1][i] = color
        grid[i][0] = grid[i][size - 1] = color

def fill_color_block(output_grid, start_r, start_c, size, fill_color):
    for dr in range(size):
        for dc in range(size):
            output_grid[start_r + dr][start_c + dc] = fill_color

def p(grid, range_func=range, min_func=min):
    RED_CODE, BLACK_CODE = (2, 0)
    grid_height, grid_width = (len(grid), len(grid[0]))
    red_square_coords = get_red_square_coordinates(grid, grid_height, grid_width, RED_CODE)
    top_row = min_func((r for r, _ in red_square_coords))
    bottom_row = max((r for r, _ in red_square_coords))
    square_side_length = bottom_row - top_row + 1
    internal_block_size = (square_side_length - 2) // 3
    non_black_non_red_coords = get_non_black_non_red_coordinates(grid, grid_height, grid_width, RED_CODE, BLACK_CODE)
    unique_color = get_unique_color_from_coordinates(grid, non_black_non_red_coords, {BLACK_CODE, RED_CODE})
    top_left_r = min_func((r for r, _ in non_black_non_red_coords))
    top_left_c = min_func((c for _, c in non_black_non_red_coords))
    output_grid = create_empty_border(square_side_length, BLACK_CODE)
    draw_border(output_grid, square_side_length, RED_CODE)
    for r, c in non_black_non_red_coords:
        rel_r, rel_c = (r - top_left_r, c - top_left_c)
        block_start_r = 1 + rel_r * internal_block_size
        block_start_c = 1 + rel_c * internal_block_size
        fill_color_block(output_grid, block_start_r, block_start_c, internal_block_size, unique_color)
    return output_grid
