def split_grid(grid):
    middle_index = 9 // 2
    middle_index_plus_one = middle_index + 9 % 2
    top_left = [row[:middle_index] for row in grid[:middle_index]]
    top_right = [row[middle_index_plus_one:] for row in grid[:middle_index]]
    bottom_left = [row[:middle_index] for row in grid[middle_index_plus_one:]]
    bottom_right = [row[middle_index_plus_one:] for row in grid[middle_index_plus_one:]]
    return (top_left, top_right, bottom_left, bottom_right)

def update_bottom_right_with_condition(source, destination, condition_value):
    for i, row in enumerate(source):
        for j, value in enumerate(row):
            if value == condition_value:
                destination[i][j] = condition_value

def p(grid, enumerator=enumerate):
    top_left, top_right, bottom_left, bottom_right = split_grid(grid)
    update_bottom_right_with_condition(bottom_left, bottom_right, 8)
    update_bottom_right_with_condition(top_right, bottom_right, 4)
    update_bottom_right_with_condition(top_left, bottom_right, 7)
    return bottom_right
