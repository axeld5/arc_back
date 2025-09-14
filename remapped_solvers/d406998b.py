def solve(grid):

    def transform_element(element, column_index):
        if element == 5 and column_index % 2 == 0:
            return 3
        return element

    def transform_row(row):
        transformed_row = [transform_element(row[j], len(grid[0]) - 1 - j) for j in range(len(row))]
        return transformed_row
    transformed_grid = [transform_row(grid[i]) for i in range(3)]
    return transformed_grid
p = solve
