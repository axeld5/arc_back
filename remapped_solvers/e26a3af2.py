def p(g):
    if not g:
        return []
    h, w = (len(g), len(g[0]))

    def m(s):
        return max(set(s), key=s.count)
    r = [m(row) for row in g]
    c = [m(col) for col in zip(*g)]
    return [list(c) for _ in range(h)] if len(set(c)) > len(set(r)) else [[r[i]] * w for i in range(h)]
