def get_empty_row_indices(matrix):
    return [row_idx for row_idx in range(len(matrix)) if all((cell == 0 for cell in matrix[row_idx]))]

def get_empty_column_indices(matrix):
    return [col_idx for col_idx in range(len(matrix[0])) if all((matrix[row_idx][col_idx] == 0 for row_idx in range(len(matrix))))]

def extract_sublists(matrix, row_indices, col_indices):
    sublists = []
    for row_block_start_idx in range(len(row_indices) - 1):
        sublist_row = []
        for col_block_start_idx in range(len(col_indices) - 1):
            for row_idx in range(row_indices[row_block_start_idx] + 1, row_indices[row_block_start_idx + 1]):
                for col_idx in range(col_indices[col_block_start_idx] + 1, col_indices[col_block_start_idx + 1]):
                    if matrix[row_idx][col_idx] != 0:
                        sublist_row.append(matrix[row_idx][col_idx])
                        break
                else:
                    continue
                break
        if sublist_row:
            sublists.append(sublist_row)
    return sublists

def p(grid):
    row_count, col_count = (len(grid), len(grid[0]))
    empty_row_indices = get_empty_row_indices(grid)
    empty_column_indices = get_empty_column_indices(grid)
    bounded_empty_row_indices = [-1] + empty_row_indices + [row_count]
    bounded_empty_column_indices = [-1] + empty_column_indices + [col_count]
    return extract_sublists(grid, bounded_empty_row_indices, bounded_empty_column_indices)
