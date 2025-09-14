from collections import Counter

def extract_subgrid(grid, start_row, start_col, size=3):
    return [row[start_col:start_col + size] for row in grid[start_row:start_row + size]]

def flatten_subgrid(subgrid):
    return [element for row in subgrid for element in row]

def p(grid):
    subgrid_size = 3
    for row in range(len(grid) - subgrid_size + 1):
        for col in range(len(grid[0]) - subgrid_size + 1):
            subgrid = extract_subgrid(grid, row, col, subgrid_size)
            flat_list = flatten_subgrid(subgrid)
            number_count = Counter(flat_list).most_common(1)
            if min(flat_list) > 0 and number_count[0][1] == 8:
                return [[subgrid[1][1]]]
