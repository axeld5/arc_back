def p(grid):
    height, width = (len(grid), len(grid[0]))
    occupied_cells = get_occupied_cells(grid)
    min_row = min((row for row, _ in occupied_cells))
    max_row = max((row for row, _ in occupied_cells))
    min_col = min((col for _, col in occupied_cells))
    square_size = int((max_row - min_row + 1) ** 0.5)
    square_area = square_size * square_size
    reference_value = grid[occupied_cells[0][0]][occupied_cells[0][1]]
    subgrid_status = initialize_subgrid_status(grid, min_row, min_col, square_size)
    result_grid = [[0] * square_area for _ in range(square_area)]
    fill_result_grid(result_grid, subgrid_status, square_size, reference_value)
    return result_grid

def get_occupied_cells(grid):
    height, width = (len(grid), len(grid[0]))
    occupied = [(r, c) for r in range(height) for c in range(width) if grid[r][c]]
    return occupied

def initialize_subgrid_status(grid, start_row, start_col, subgrid_size):
    subgrid_status = [[any((grid[start_row + i * subgrid_size + u][start_col + j * subgrid_size + v] for u in range(subgrid_size) for v in range(subgrid_size))) for j in range(subgrid_size)] for i in range(subgrid_size)]
    return subgrid_status

def fill_result_grid(result, subgrid_status, subgrid_size, value):
    for i in range(subgrid_size):
        for j in range(subgrid_size):
            if subgrid_status[i][j]:
                for r in range(subgrid_size):
                    for c in range(subgrid_size):
                        if subgrid_status[r][c]:
                            result[i * subgrid_size + r][j * subgrid_size + c] = value
