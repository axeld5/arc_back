def p(I):

    def sum_of_twos_in_rows(matrix):
        return [sum((cell == 2 for cell in row)) for row in matrix]

    def sum_of_twos_in_columns(matrix):
        return [sum((matrix[row][col] == 2 for row in range(len(matrix)))) for col in range(len(matrix[0]))]

    def transpose(matrix):
        return [list(row) for row in zip(*matrix)]

    def shift_balls_downward(grid):
        num_rows, num_cols = (len(grid), len(grid[0]))
        result_grid = [row[:] for row in grid]
        twos_count_per_row = sum_of_twos_in_rows(grid)
        row_indices_sorted_by_twos = sorted(range(num_rows), key=lambda r: twos_count_per_row[r], reverse=True)
        row_with_most_twos, second_row_with_most_twos = sorted((row_indices_sorted_by_twos[0], row_indices_sorted_by_twos[1]))
        for r in range(num_rows):
            for c in range(num_cols):
                if grid[r][c] == 5:
                    target_row = row_with_most_twos if abs(r - row_with_most_twos) <= abs(r - second_row_with_most_twos) else second_row_with_most_twos
                    potential_row = 2 * target_row - r
                    step = -1 if potential_row < target_row else 1
                    while 0 <= potential_row < num_rows and result_grid[potential_row][c]:
                        potential_row += step
                    if 0 <= potential_row < num_rows:
                        result_grid[potential_row][c] = 5
                    result_grid[r][c] = 0
        return result_grid
    twos_in_rows = sum_of_twos_in_rows(I)
    twos_in_columns = sum_of_twos_in_columns(I)
    if max(twos_in_rows) >= max(twos_in_columns):
        return shift_balls_downward(I)
    else:
        transposed_result = shift_balls_downward(transpose(I))
        return transpose(transposed_result)
