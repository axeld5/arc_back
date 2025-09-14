def solve(grid):

    def process_row(row):
        return [2 if a and b else 0 for a, b in zip(row[:3], row[4:7])]
    return [process_row(row) for row in grid[:3]]
p = solve
