def analyze_subgrid(grid, subgrid_positions):
    subgrid = [[0] * 3 for _ in range(3)]
    number_counts = {}
    for i, j in subgrid_positions:
        cell_value = grid[i][j]
        subgrid[i - subgrid_positions[0][0]][j - subgrid_positions[0][1]] = cell_value
        if cell_value == 0:
            return 0
        if cell_value not in number_counts:
            number_counts[cell_value] = 0
        number_counts[cell_value] += 1
    if len(number_counts) == 1 and 2 in number_counts:
        return 0
    unique_count, subgrid_data = list(number_counts.items())[0]
    return ((unique_count, subgrid) if unique_count == 2 else list(number_counts.items())[1][1], subgrid)

def find_subgrids(grid):
    height, width = (len(grid), len(grid[0]))
    subgrids_data = []
    covered_positions = []
    for i in range(height - 2):
        for j in range(width - 2):
            subgrid_positions = [(x, y) for x in range(i, i + 3) for y in range(j, j + 3)]
            result = analyze_subgrid(grid, subgrid_positions)
            if result:
                subgrids_data.append([result[0], result[1]])
                covered_positions.extend(subgrid_positions)
    return (subgrids_data, covered_positions)

def extract_potential_grid(grid):
    subgrids_data, covered_positions = find_subgrids(grid)
    height, width = (len(grid), len(grid[0]))
    unvisited_cells = []
    for i in range(height):
        for j in range(width):
            if grid[i][j] == 2 and (i, j) not in covered_positions:
                unvisited_cells.append((i, j))
    min_x, min_y = (height, width)
    max_x, max_y = (0, 0)
    for x, y in unvisited_cells:
        min_x, min_y = (min(min_x, x), min(min_y, y))
        max_x, max_y = (max(max_x, x), max(max_y, y))
    potential_grid = [row[min_y:max_y + 1] for row in grid[min_x:max_x + 1]]
    return (subgrids_data, potential_grid)

def rotate_3x3_grid(grid, rotations=1):
    rotated_grid = [row[:] for row in grid]
    size = len(grid)
    for _ in range(rotations):
        new_grid = [row[:] for row in grid]
        for i in range(size):
            for j in range(size):
                new_grid[j][size - 1 - i] = rotated_grid[i][j]
        rotated_grid = new_grid
    return rotated_grid

def validate_placement(subgrid, pattern):
    for i in range(3):
        for j in range(3):
            if subgrid[i][j] == 0:
                if pattern[i][j] != 2:
                    return False
            if subgrid[i][j] == 2 and pattern[i][j] == 2:
                return False
    return True

def find_valid_placement(subgrid, subgrids_data, used_indices, rotation):
    for index, (count, pattern) in enumerate(subgrids_data):
        if index in used_indices:
            continue
        rotated_pattern = rotate_3x3_grid(pattern, rotation)
        if validate_placement(subgrid, rotated_pattern):
            return (index, rotated_pattern)
    return (-1, -1)

def p(grid):
    subgrids_data, potential_grid = extract_potential_grid(grid)
    height, width = (len(potential_grid), len(potential_grid[0]))
    completed_grid = [row[:] for row in potential_grid]
    used_indices = []
    visited_positions = []
    for rotation in range(4):
        for i in range(height - 2):
            for j in range(width - 2):
                if (i, j) in visited_positions:
                    continue
                subgrid_positions = [(x, y) for x in range(i, i + 3) for y in range(j, j + 3)]
                current_subgrid = [row[j:j + 3] for row in potential_grid[i:i + 3]]
                index, placed_pattern = find_valid_placement(current_subgrid, subgrids_data, used_indices, rotation)
                if index != -1:
                    used_indices.append(index)
                    visited_positions.extend(subgrid_positions)
                    for x in range(3):
                        for y in range(3):
                            completed_grid[i + x][j + y] = placed_pattern[x][y]
    return completed_grid
