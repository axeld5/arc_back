def p(grid, enum=enumerate, rng=range):
    from collections import Counter
    GRID_SIZE = 10

    def count_value_five(grid):
        return sum((value == 5 for row in grid for value in row))

    def replace_fives(grid):
        return [0 if value == 5 else value for row in grid for value in row]

    def find_least_common_value(flat_grid):
        return min(Counter(flat_grid), key=Counter(flat_grid).get)

    def locate_least_common_value_coords(grid, uncommon_value):
        return [(i, j) for i, row in enum(grid) for j, val in enum(row) if val == uncommon_value]

    def initialize_output_grid():
        return [[0] * GRID_SIZE for _ in rng(GRID_SIZE)]

    def mark_lines(output_grid, row, col, uncommon_value):
        for i in rng(GRID_SIZE):
            output_grid[row][i] = uncommon_value
        for i in rng(GRID_SIZE):
            output_grid[i][col] = uncommon_value
    count_of_fives = count_value_five(grid)
    flat_grid_no_fives = replace_fives(grid)
    least_common_value = find_least_common_value(flat_grid_no_fives)
    least_common_value_coords = locate_least_common_value_coords(grid, least_common_value)
    output_grid = initialize_output_grid()
    for row, col in least_common_value_coords:
        if 0 < row < GRID_SIZE - 1 and 0 < col < GRID_SIZE - 1:
            if all((grid[x][y] == least_common_value for x, y in [(row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)])):
                mark_lines(output_grid, row + count_of_fives, col - count_of_fives, least_common_value)
                break
    return output_grid
