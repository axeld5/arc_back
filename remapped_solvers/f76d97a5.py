def solve(grid):

    def flatten_grid(grid):
        return [num for row in grid for num in row]

    def find_smallest_non_zero_and_non_five(flattened_grid):
        unique_numbers = set(flattened_grid)
        candidates = [number for number in unique_numbers if number not in [0, 5]]
        return min(candidates)

    def replace_fives_with_smallest_number(grid, replacement_number):
        return [[replacement_number if number == 5 else 0 for number in row] for row in grid]
    flattened_grid = flatten_grid(grid)
    smallest_number = find_smallest_non_zero_and_non_five(flattened_grid)
    return replace_fives_with_smallest_number(grid, smallest_number)
p = solve
