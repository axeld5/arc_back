def solve(grid):
    size = min(len(grid), len(grid[0]))
    primary_section, complementary_section = extract_grid_sections(grid, size)
    primary_section, complementary_section = swap_sections_if_needed(primary_section, complementary_section)
    return generate_transformed_grid(primary_section, complementary_section, size)

def extract_grid_sections(grid, size):
    primary_section = [row[:size] for row in grid[:size]]
    complementary_section = [row[-size:] for row in grid[-size:]]
    return (primary_section, complementary_section)

def swap_sections_if_needed(primary, complementary):
    if any((max(row) == 8 for row in primary)):
        return (complementary, primary)
    return (primary, complementary)

def generate_transformed_grid(primary, complementary, size):
    transformed_grid = []
    for y in range(size * size):
        row = []
        for x in range(size * size):
            primary_value = primary[y // size][x // size]
            complementary_value = complementary[y % size][x % size]
            new_value = primary_value * complementary_value // 8
            row.append(new_value)
        transformed_grid.append(row)
    return transformed_grid
p = solve
