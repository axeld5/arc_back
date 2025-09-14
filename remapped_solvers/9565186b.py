from collections import Counter
from typing import List

def find_most_common_element(grid: List[List[int]]) -> int:
    element_counts = Counter([element for row in grid for element in row])
    most_common_element, _ = element_counts.most_common(1)[0]
    return most_common_element

def replace_elements_in_grid(grid: List[List[int]], target: int) -> List[List[int]]:
    return [[element if element == target else 5 for element in row] for row in grid]

def p(j: List[List[int]]) -> List[List[int]]:
    most_common_element = find_most_common_element(j)
    transformed_grid = replace_elements_in_grid(j, most_common_element)
    return transformed_grid
