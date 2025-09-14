def p(G):

    def remove_duplicates(row):
        return [x for i, x in enumerate(row) if all((x != row[j] for j in range(i)))]

    def transpose(grid):
        return [list(row) for row in zip(*grid)]

    def reverse(grid):
        return grid[::-1]

    def last(collection):
        return collection[-1]

    def remove_value(value, rows):
        return [element for element in rows if element != value]

    def concat(a, b):
        return a + b

    def transform(x):
        return concat(x, reverse(remove_value(last(x), x)))
    processed = remove_duplicates(transpose(remove_duplicates(transpose(G))))
    transformed = transpose(transform(processed))
    return transform(transformed)
