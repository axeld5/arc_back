from collections import Counter
from typing import List

def extract_top_elements(flat_grid: List[int], start_rank: int, end_rank: int) -> List[int]:
    counter = Counter(flat_grid)
    return [element for element, _ in counter.most_common()[start_rank:end_rank]]

def grid_to_flat_list(grid: List[List[int]]) -> List[int]:
    return [element for row in grid for element in row]

def p(grid: List[List[int]]) -> List[List[int]]:
    flat_grid = grid_to_flat_list(grid)
    top_elements = extract_top_elements(flat_grid, 2, 5)
    return [[element] for element in top_elements]
