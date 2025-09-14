def p(G):
    n = len(G)
    C = [x for x in G[0] if x]
    result = []
    for i in range(n * 2):
        row = []
        for j in range(n * 2):
            color_index = max(i, j) % len(C)
            row.append(C[color_index])
        result.append(row)
    return result
