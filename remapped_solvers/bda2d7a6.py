def p(grid):

    def extract_diagonal(grid):
        half_length = len(grid) // 2
        diagonal_elements = [grid[i][i] for i in range(half_length)]
        return diagonal_elements

    def create_mapping(diagonal_elements):
        return {diagonal_elements[i]: diagonal_elements[i - 1] for i in range(len(diagonal_elements))}

    def transform_grid(grid, mapping):
        return [[mapping[element] for element in row] for row in grid]
    diagonal_elements = extract_diagonal(grid)
    diagonal_mapping = create_mapping(diagonal_elements)
    transformed_grid = transform_grid(grid, diagonal_mapping)
    return transformed_grid
