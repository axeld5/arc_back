from collections import Counter
from typing import List, Any

def _find_most_common_element(matrix: List[List[Any]]) -> Any:
    element_counts = Counter((element for row in matrix for element in row if element))
    least_common_element = element_counts.most_common()[-1][0]
    return least_common_element

def _find_bounding_box(matrix: List[List[Any]], target_element: Any) -> (int, int, int, int):
    top_row = bottom_row = -1
    left_col = right_col = -1
    for i, row in enumerate(matrix):
        if target_element in row:
            if top_row < 0:
                top_row = i
            bottom_row = i
    for j in range(len(matrix[0])):
        if any((matrix[i][j] == target_element for i in range(top_row, bottom_row + 1))):
            if left_col < 0:
                left_col = j
            right_col = j
    return (top_row, bottom_row, left_col, right_col)

def p(matrix: List[List[Any]]) -> List[List[Any]]:
    least_common_element = _find_most_common_element(matrix)
    top_row, bottom_row, left_col, right_col = _find_bounding_box(matrix, least_common_element)
    return [row[left_col:right_col + 1] for row in matrix[top_row:bottom_row + 1]]
