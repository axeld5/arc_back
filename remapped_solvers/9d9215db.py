def p(input_grid, range_func=range):
    SIZE = 19

    def reflect(grid, horizontal=False, vertical=False):
        if horizontal:
            grid = [row[::-1] for row in grid]
        if vertical:
            grid = grid[::-1]
        return grid

    def generate_diagonals(grid):
        diagonal_grid = [[0] * SIZE for _ in range(SIZE)]
        for i in range_func(1, SIZE - 2, 2):
            if grid[i][i] != 0 != grid[i][i + 2]:
                value = grid[i][i + 2]
                top, bottom = (i, SIZE - 1 - i)
                left, right = (i, SIZE - 1 - i)
                for X in range_func(i + 2, SIZE - 1 - i, 2):
                    diagonal_grid[X][left] = diagonal_grid[X][right] = value
                for Y in range_func(i + 2, SIZE - 1 - i, 2):
                    diagonal_grid[top][Y] = diagonal_grid[bottom][Y] = value
        return diagonal_grid

    def apply_diagonal_markings(overall_grid, diagonal_grid):
        for row in range(SIZE):
            for col in range(SIZE):
                if diagonal_grid[row][col] != 0:
                    overall_grid[row][col] = diagonal_grid[row][col]
    mirrored_grid = [[0] * SIZE for _ in range(SIZE)]
    for row in range(SIZE):
        for col in range(SIZE):
            value = input_grid[row][col]
            if value != 0:
                mirrored_grid[row][col] = mirrored_grid[SIZE - 1 - row][col] = mirrored_grid[row][SIZE - 1 - col] = mirrored_grid[SIZE - 1 - row][SIZE - 1 - col] = value
    for horizontal in (0, 1):
        for vertical in (0, 1):
            transformed_input = reflect(input_grid, horizontal, vertical)
            diagonal_grid = generate_diagonals(transformed_input)
            apply_diagonal_markings(mirrored_grid, reflect(diagonal_grid, horizontal, vertical))
    return mirrored_grid
