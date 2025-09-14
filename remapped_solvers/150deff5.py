def find_all_5_positions(grid):
    n, m = (len(grid), len(grid[0]))
    fives_positions = {(i, j) for i in range(n) for j in range(m) if grid[i][j] == 5}
    return fives_positions

def identify_shapes(grid, fives_positions):
    n, m = (len(grid), len(grid[0]))
    shapes = []
    for r in range(n - 1):
        for c in range(m - 1):
            square_cells = [(r, c), (r, c + 1), (r + 1, c), (r + 1, c + 1)]
            if all((pos in fives_positions for pos in square_cells)):
                shapes.append({'coordinates': square_cells, 'type': 'S'})
    for r in range(n):
        for c in range(m - 2):
            horizontal_line = [(r, c), (r, c + 1), (r, c + 2)]
            if all((pos in fives_positions for pos in horizontal_line)):
                shapes.append({'coordinates': horizontal_line, 'type': 'L'})
    for r in range(n - 2):
        for c in range(m):
            vertical_line = [(r, c), (r + 1, c), (r + 2, c)]
            if all((pos in fives_positions for pos in vertical_line)):
                shapes.append({'coordinates': vertical_line, 'type': 'L'})
    return shapes

def build_position_to_shape_map(fives_positions, shapes):
    position_to_shapes = {pos: set() for pos in fives_positions}
    for shape_index, shape in enumerate(shapes):
        for pos in shape['coordinates']:
            if pos in position_to_shapes:
                position_to_shapes[pos].add(shape_index)
    return position_to_shapes

def solve_shapes(remaining_positions, position_to_shapes, shapes):
    if not remaining_positions:
        return []

    def get_valid_shapes(pos):
        return [index for index in position_to_shapes[pos] if all((p in remaining_positions for p in shapes[index]['coordinates']))]
    position = min(remaining_positions, key=lambda x: len(get_valid_shapes(x)))
    possible_shapes = get_valid_shapes(position)
    for shape_index in possible_shapes:
        new_remaining = remaining_positions - set(shapes[shape_index]['coordinates'])
        conflicting_shapes = set()
        new_position_to_shapes = {pos: set(shapes_list) for pos, shapes_list in position_to_shapes.items()}
        for pos in shapes[shape_index]['coordinates']:
            conflicting_shapes |= position_to_shapes[pos]
        for conflict_index in conflicting_shapes:
            for pos in shapes[conflict_index]['coordinates']:
                if pos in new_position_to_shapes:
                    new_position_to_shapes[pos].discard(conflict_index)
        result = solve_shapes(new_remaining, new_position_to_shapes, shapes)
        if result is not None:
            return [shape_index] + result
    return None

def p(grid):
    fives_positions = find_all_5_positions(grid)
    shapes = identify_shapes(grid, fives_positions)
    position_to_shapes = build_position_to_shape_map(fives_positions, shapes)
    selected_shape_indices = solve_shapes(set(fives_positions), position_to_shapes, shapes)
    if selected_shape_indices is None:
        return grid
    solution_grid = [row[:] for row in grid]
    for index in selected_shape_indices:
        color = 8 if shapes[index]['type'] == 'S' else 2
        for r, c in shapes[index]['coordinates']:
            solution_grid[r][c] = color
    return solution_grid
