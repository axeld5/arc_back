from collections import defaultdict
from typing import List, Tuple

def fill_largest_group(grid: List[List[int]]) -> List[List[int]]:
    number_count = defaultdict(int)
    number_positions = defaultdict(list)
    for row_index, row in enumerate(grid):
        for col_index, value in enumerate(row):
            number_count[value] += 1
            number_positions[value].append((row_index, col_index))
    if not number_count:
        return grid[:]
    max_number = max(number_count, key=number_count.get)
    filled_grid = [row[:] for row in grid]
    for number in number_count:
        if number and number != max_number:
            fill_rows_and_cols(number, number_positions[number], filled_grid)
    for i, j in number_positions[max_number]:
        filled_grid[i][j] = max_number
    return filled_grid

def fill_rows_and_cols(number: int, positions: List[Tuple[int, int]], grid: List[List[int]]):
    number_rows = defaultdict(list)
    number_cols = defaultdict(list)
    for row_index, col_index in positions:
        number_rows[row_index].append(col_index)
        number_cols[col_index].append(row_index)
    for row_index, column_indices in number_rows.items():
        if len(column_indices) > 1:
            for col_index in range(min(column_indices), max(column_indices) + 1):
                grid[row_index][col_index] = number
    for col_index, row_indices in number_cols.items():
        if len(row_indices) > 1:
            for row_index in range(min(row_indices), max(row_indices) + 1):
                grid[row_index][col_index] = number

def p(grid: List[List[int]]) -> List[List[int]]:
    return fill_largest_group(grid)
