from collections import Counter

def solve(grid, A=range(9), c=range(3)):
    most_common_element = get_most_common_element(grid)
    result_grid = initialize_empty_grid(A)
    fill_grid(grid, result_grid, A, c, most_common_element)
    return result_grid

def get_most_common_element(grid):
    return Counter(grid[0] + grid[1] + grid[2]).most_common(1)[0][0]

def initialize_empty_grid(A):
    return [[0 for _ in A] for _ in A]

def fill_grid(grid, result_grid, A, c, most_common_element):
    for W, l in [(W, l) for l in c for W in c if grid[W][l] == most_common_element]:
        for J in A:
            result_grid[3 * W + J % 3][3 * l + J // 3] = grid[J % 3][J // 3]

def p(j, A=range(9), c=range(3)):
    return solve(j, A, c)
