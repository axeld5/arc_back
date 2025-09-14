from collections import Counter, defaultdict

def get_background_and_positions(grid):
    height, width = (len(grid), len(grid[0]))
    background_char = Counter((value for row in grid for value in row)).most_common(1)[0][0]
    char_positions = defaultdict(list)
    for r in range(height):
        for c in range(width):
            if grid[r][c] != background_char:
                char_positions[grid[r][c]].append((r, c))
    return (background_char, char_positions)

def get_bounding_boxes(char_positions):
    bounding_boxes = {}
    for char, positions in char_positions.items():
        rows = [r for r, _ in positions]
        cols = [c for _, c in positions]
        min_row, max_row = (min(rows), max(rows))
        min_col, max_col = (min(cols), max(cols))
        bounding_boxes[char] = (min_row, max_row, min_col, max_col)
    return bounding_boxes

def calculate_minimal_square_side(bounding_boxes):
    max_side = 1
    for min_row, max_row, min_col, max_col in bounding_boxes.values():
        height = max_row - min_row + 1
        width = max_col - min_col + 1
        max_side = max(max_side, height, width)
    return max_side

def place_characters_on_grid(background_char, char_positions, bounding_boxes, square_side):
    new_grid = [[background_char] * square_side for _ in range(square_side)]
    for char in sorted(char_positions.keys()):
        min_row, max_row, min_col, max_col = bounding_boxes[char]
        char_height = max_row - min_row + 1
        char_width = max_col - min_col + 1
        start_row = (square_side - char_height) // 2
        start_col = (square_side - char_width) // 2
        for r, c in char_positions[char]:
            new_grid[r - min_row + start_row][c - min_col + start_col] = char
    return new_grid

def p(grid):
    background_char, char_positions = get_background_and_positions(grid)
    bounding_boxes = get_bounding_boxes(char_positions)
    square_side = calculate_minimal_square_side(bounding_boxes)
    new_grid = place_characters_on_grid(background_char, char_positions, bounding_boxes, square_side)
    return new_grid
