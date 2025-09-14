def find_positions(grid, value, enumerator=enumerate):
    return [(row_idx, col_idx) for row_idx, row in enumerator(grid) for col_idx, cell in enumerator(row) if cell == value]

def calculate_shift(coconut_positions, empty_positions):
    if not coconut_positions or not empty_positions:
        return (0, 0)
    min_coco_row = min((pos[0] for pos in coconut_positions))
    max_coco_row = max((pos[0] for pos in coconut_positions))
    min_coco_col = min((pos[1] for pos in coconut_positions))
    max_coco_col = max((pos[1] for pos in coconut_positions))
    min_empty_row = min((pos[0] for pos in empty_positions))
    max_empty_row = max((pos[0] for pos in empty_positions))
    min_empty_col = min((pos[1] for pos in empty_positions))
    max_empty_col = max((pos[1] for pos in empty_positions))
    row_shift = col_shift = 0
    if max_coco_col < min_empty_col:
        col_shift = min_empty_col - max_coco_col - 1
    elif max_empty_col < min_coco_col:
        col_shift = max_empty_col - min_coco_col + 1
    if max_coco_row < min_empty_row:
        row_shift = min_empty_row - max_coco_row - 1
    elif max_empty_row < min_coco_row:
        row_shift = max_empty_row - min_coco_row + 1
    return (row_shift, col_shift)

def p(grid, enumerator=enumerate):
    coconut_positions = find_positions(grid, 2, enumerator)
    empty_positions = find_positions(grid, 8, enumerator)
    row_shift, col_shift = calculate_shift(coconut_positions, empty_positions)
    coconut_set, empty_set = (set(coconut_positions), set(empty_positions))
    return [[8 if (row, col) in empty_set else 2 if (row - row_shift, col - col_shift) in coconut_set else 0 for col, _ in enumerator(grid[0])] for row, _ in enumerator(grid)]
