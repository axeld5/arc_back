from collections import deque

def p(grid, range_function=range):
    num_rows, num_cols = (len(grid), len(grid[0]))
    visited, objects = (set(), [])
    for row in range_function(num_rows):
        for col in range_function(num_cols):
            if grid[row][col] != 6 or (row, col) in visited:
                continue
            block = find_block_of_sixes(grid, row, col, visited)
            objects.append(block)
    surrounding_positions, hollow_positions = (set(), set())
    for block in objects:
        mark_surrounding_and_hollow_positions(block, surrounding_positions, hollow_positions, num_rows, num_cols, range_function)
    output_grid = mark_positions_on_grid(grid, surrounding_positions, hollow_positions)
    return output_grid

def find_block_of_sixes(grid, row, col, visited):
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    block = set()
    queue = deque([(row, col)])
    visited.add((row, col))
    while queue:
        r, c = queue.popleft()
        block.add((r, c))
        for dr, dc in directions:
            new_r, new_c = (r + dr, c + dc)
            if 0 <= new_r < len(grid) and 0 <= new_c < len(grid[0]) and (grid[new_r][new_c] == 6) and ((new_r, new_c) not in visited):
                visited.add((new_r, new_c))
                queue.append((new_r, new_c))
    return block

def mark_surrounding_and_hollow_positions(block, surrounding_positions, hollow_positions, num_rows, num_cols, range_function):
    block_rows = [r for r, _ in block]
    block_cols = [c for _, c in block]
    min_row, max_row = (min(block_rows), max(block_rows))
    min_col, max_col = (min(block_cols), max(block_cols))
    for r in range_function(min_row, max_row + 1):
        for c in range_function(min_col, max_col + 1):
            if (r, c) not in block:
                hollow_positions.add((r, c))
    for r in range_function(min_row - 1, max_row + 2):
        if 0 <= r < num_rows:
            if min_col - 1 >= 0:
                surrounding_positions.add((r, min_col - 1))
            if max_col + 1 < num_cols:
                surrounding_positions.add((r, max_col + 1))
    for c in range_function(min_col - 1, max_col + 2):
        if 0 <= c < num_cols:
            if min_row - 1 >= 0:
                surrounding_positions.add((min_row - 1, c))
            if max_row + 1 < num_rows:
                surrounding_positions.add((max_row + 1, c))

def mark_positions_on_grid(grid, surrounding_positions, hollow_positions):
    output_grid = [row[:] for row in grid]
    for r, c in surrounding_positions:
        output_grid[r][c] = 3
    for r, c in hollow_positions:
        output_grid[r][c] = 4
    return output_grid
