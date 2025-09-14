def p(grid):
    MARKED_NUMBER = 8
    height = len(grid)
    width = len(grid[0])
    potential_squares = find_potential_squares(grid, height, width)
    potential_squares.sort(key=lambda x: x[2], reverse=True)
    _, _, square_size, top_color, right_color, bottom_color, left_color = potential_squares[0]
    inner_length = square_size - 1
    marked_positions = [(r, c) for r in range(height) for c in range(width) if grid[r][c] == MARKED_NUMBER]
    min_row = min((r for r, _ in marked_positions))
    min_col = min((c for _, c in marked_positions))
    inner_grid = create_inner_grid(inner_length, marked_positions, min_row, min_col)
    output_grid = fill_output_grid(inner_length, square_size, top_color, right_color, bottom_color, left_color, inner_grid, MARKED_NUMBER)
    return output_grid

def find_potential_squares(grid, height, width):
    potential_squares = []
    for row in range(height):
        for col in range(width):
            if grid[row][col]:
                continue
            for size in range(2, min(height - row, width - col)):
                bottom_row = row + size
                right_col = col + size
                if bottom_row >= height or right_col >= width:
                    break
                if grid[row][right_col] or grid[bottom_row][col] or grid[bottom_row][right_col]:
                    continue
                if is_valid_square(grid, row, col, bottom_row, right_col):
                    top_color = grid[row][col + 1]
                    right_color = grid[row + 1][right_col]
                    bottom_color = grid[bottom_row][col + 1]
                    left_color = grid[row + 1][col]
                    potential_squares.append((row, col, size, top_color, right_color, bottom_color, left_color))
    return potential_squares

def is_valid_square(grid, row, col, bottom_row, right_col):

    def unique_color(segment):
        return len(segment) > 0 and all((x == segment[0] and x for x in segment))
    top_border = [grid[row][i] for i in range(col + 1, right_col)]
    right_border = [grid[i][right_col] for i in range(row + 1, bottom_row)]
    bottom_border = [grid[bottom_row][i] for i in range(col + 1, right_col)]
    left_border = [grid[i][col] for i in range(row + 1, bottom_row)]
    return unique_color(top_border) and unique_color(right_border) and unique_color(bottom_border) and unique_color(left_border)

def create_inner_grid(inner_length, marked_positions, min_row, min_col):
    inner_grid = [[0] * inner_length for _ in range(inner_length)]
    for r, c in marked_positions:
        relative_row = r - min_row
        relative_col = c - min_col
        if 0 <= relative_row < inner_length and 0 <= relative_col < inner_length:
            inner_grid[relative_row][relative_col] = 1
    return inner_grid

def fill_output_grid(inner_length, square_size, top_color, right_color, bottom_color, left_color, inner_grid, marked_number):
    output_grid = [[0] * (inner_length + 2) for _ in range(inner_length + 2)]
    for i in range(1, square_size):
        output_grid[0][i] = top_color
        output_grid[i][square_size] = right_color
        output_grid[square_size][i] = bottom_color
        output_grid[i][0] = left_color
    for r in range(inner_length):
        for c in range(inner_length):
            if not inner_grid[r][c]:
                continue
            if r == c or r + c == inner_length - 1:
                color = marked_number
            else:
                row_diff = r - c
                col_diff = inner_length - 1 - r - c
                if row_diff < 0 and col_diff > 0:
                    color = top_color
                elif row_diff < 0 and col_diff < 0:
                    color = right_color
                elif row_diff > 0 and col_diff < 0:
                    color = bottom_color
                else:
                    color = left_color
            output_grid[r + 1][c + 1] = color
    return output_grid
