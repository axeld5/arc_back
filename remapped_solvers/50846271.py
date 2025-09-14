def p(grid):
    from itertools import product
    RED, GREEN, COVERAGE = (2, 5, 8)
    HEIGHT, WIDTH = (len(grid), len(grid[0]))

    def get_red_positions():
        return {(i, j) for i, j in product(range(HEIGHT), range(WIDTH)) if grid[i][j] == RED}

    def get_horizontal_and_vertical_positions(i, j, length):
        horizontal = [(i, j + d) for d in range(-length, length + 1) if 0 <= j + d < WIDTH]
        vertical = [(i + d, j) for d in range(-length, length + 1) if 0 <= i + d < HEIGHT]
        return (horizontal, vertical)

    def evaluate_position(i, j, length):
        horizontal, vertical = get_horizontal_and_vertical_positions(i, j, length)
        combined = set(horizontal) | set(vertical)
        if all((grid[x][y] in (RED, GREEN) for x, y in combined)):
            red_positions = {(x, y) for x, y in combined if grid[x][y] == RED}
            return (len(red_positions) if len(red_positions) > 1 else 0, len(red_positions), combined)
        else:
            return (0, 0, set())

    def find_optimal_coverage(length):
        potential_positions = []
        for i in range(HEIGHT):
            for j in range(WIDTH):
                red_cover_count, _, affected_positions = evaluate_position(i, j, length)
                if red_cover_count:
                    potential_positions.append((red_cover_count, (i, j), affected_positions))
        final_positions, uncovered_reds = ([], set(red_positions))
        while True:
            best_choice = None
            for red_cover_count, position, affected in potential_positions:
                if any((max(abs(position[0] - x), abs(position[1] - y)) <= length for x, y in final_positions)):
                    continue
                uncover_count = len([p for p in affected if p in uncovered_reds and grid[p[0]][p[1]] == RED])
                if uncover_count > 0 and (best_choice is None or uncover_count > best_choice[0]):
                    best_choice = (uncover_count, position, affected)
            if best_choice is None:
                break
            _, position, affected = best_choice
            final_positions.append(position)
            uncovered_reds -= {p for p in affected if grid[p[0]][p[1]] == RED}
        return final_positions

    def get_coverage_count(positions, length):
        covered_cells = set()
        for i, j in positions:
            for d in range(-length, length + 1):
                if 0 <= j + d < WIDTH:
                    covered_cells.add((i, j + d))
                if 0 <= i + d < HEIGHT:
                    covered_cells.add((i + d, j))
        return sum((p in covered_cells for p in red_positions))
    red_positions = get_red_positions()
    best_coverage = None
    for length in (2, 3):
        optimal_positions = find_optimal_coverage(length)
        coverage_count = get_coverage_count(optimal_positions, length)
        if best_coverage is None or coverage_count > best_coverage[0]:
            best_coverage = (coverage_count, length, optimal_positions)
    _, optimal_length, optimal_positions = best_coverage
    modified_grid = [row[:] for row in grid]
    for i, j in optimal_positions:
        for d in range(-optimal_length, optimal_length + 1):
            if 0 <= j + d < WIDTH and modified_grid[i][j + d] == GREEN:
                modified_grid[i][j + d] = COVERAGE
            if 0 <= i + d < HEIGHT and modified_grid[i + d][j] == GREEN:
                modified_grid[i + d][j] = COVERAGE
    return modified_grid
