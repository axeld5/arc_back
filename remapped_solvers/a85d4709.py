def solve(grid):

    def get_repeated_value(row):
        selection_list = [2, 4, 3]
        index_of_five = row.index(5)
        selected_value = selection_list[index_of_five]
        return [selected_value] * 3
    return [get_repeated_value(row) for row in grid]
p = solve
