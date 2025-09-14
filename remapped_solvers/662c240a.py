def extract_subgrid(grid, start_row, start_col, size=3):
    return [grid[start_row + i][start_col:start_col + size] for i in range(size)]

def subgrid_differs(grid, start_row):
    subgrid = extract_subgrid(grid, start_row, 0)
    transposed_subgrid = [[grid[start_row + j][i] for j in range(3)] for i in range(3)]
    return subgrid != transposed_subgrid

def find_differing_subgrid(grid):
    for start_row in range(0, 9, 3):
        if subgrid_differs(grid, start_row):
            return extract_subgrid(grid, start_row, 0)
    return None

def p(grid):
    return find_differing_subgrid(grid)
