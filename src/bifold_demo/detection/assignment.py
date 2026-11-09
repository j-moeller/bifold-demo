"""The assignment problem in numpy: who goes with whom at the least total cost.

:func:`linear_sum_assignment` pairs the rows of a cost matrix with its columns, each at most once, so that the summed
cost of the pairs is the least possible (every row of a matrix with fewer rows than columns gets a column, and the
other way round). It is the shortest augmenting path form of the Hungarian algorithm (Jonker and Volgenant; as
Crouse, 2016, describes it and SciPy implements it): one row at a time is added, by the cheapest path of reduced
costs that ends in a free column, with the dual variables keeping every reduced cost non-negative. O(n^3), which for
the matrices here (a crowd of a few dozen people, a gallery of a few dozen names) is a fraction of a millisecond.

:func:`match` is what trackers and decisions need: a pair is only made when its cost is below ``limit``, and the
assignment is the cheapest *with* leaving rows and columns unpaired at ``limit / 2`` each (ByteTrack's
``lapjv(..., extend_cost=True, cost_limit=limit)``). That is not the same as solving first and dropping expensive
pairs afterwards: an expensive pair can push a cheap one aside.

Both are deterministic: ties go to the lowest index.
"""

from __future__ import annotations

import numpy as np

__all__ = ["linear_sum_assignment", "match"]


def linear_sum_assignment(cost) -> tuple[np.ndarray, np.ndarray]:
    """``(rows, columns)``: the pairs of the least summed cost of ``cost`` (an ``n x m`` matrix of finite numbers),
    ``min(n, m)`` of them, sorted by row. Raises ValueError for a matrix with values that are not finite."""
    cost = np.asarray(cost, dtype=np.float64)
    if cost.ndim != 2:
        raise ValueError(f"a cost matrix must have two dimensions, not {cost.ndim}")
    if not np.isfinite(cost).all():
        raise ValueError("a cost matrix must hold finite numbers (use a large cost for pairs that may not be)")
    transposed = cost.shape[0] > cost.shape[1]
    if transposed:
        cost = cost.T
    n, m = cost.shape
    if n == 0:
        return np.zeros(0, int), np.zeros(0, int)
    u, v = np.zeros(n), np.zeros(m)  # the dual variables of the rows and the columns
    row_of = np.full(m, -1)  # the row each column is paired with
    col_of = np.full(n, -1)  # the column each row is paired with
    for start in range(n):
        shortest = np.full(m, np.inf)  # the cheapest reduced path from ``start`` to each column so far
        path = np.full(m, -1)  # the row before each column on that path
        done_rows = np.zeros(n, bool)
        done_cols = np.zeros(m, bool)
        lowest, i, sink = 0.0, start, -1
        while sink < 0:
            done_rows[i] = True
            open_cols = ~done_cols
            reduced = lowest + cost[i] - u[i] - v
            better = open_cols & (reduced < shortest)
            shortest[better] = reduced[better]
            path[better] = i
            candidates = np.flatnonzero(open_cols)
            values = shortest[candidates]
            best = values.min()
            ties = candidates[values == best]
            free = ties[row_of[ties] < 0]  # a free column among the cheapest ends the path at once
            j = int(free[0]) if len(free) else int(ties[0])
            lowest = float(best)
            done_cols[j] = True
            if row_of[j] < 0:
                sink = j
            else:
                i = int(row_of[j])
        u[start] += lowest  # the duals, so that every reduced cost stays non-negative
        others = done_rows.copy()
        others[start] = False
        rows = np.flatnonzero(others)
        u[rows] += lowest - shortest[col_of[rows]]
        v[done_cols] -= lowest - shortest[done_cols]
        j = sink  # the path, augmented: every row on it moves to the column after it
        while True:
            i = int(path[j])
            row_of[j] = i
            col_of[i], j = j, col_of[i]
            if i == start:
                break
    rows = np.arange(n)
    if transposed:
        order = np.argsort(col_of, kind="stable")
        return col_of[order], rows[order]
    return rows, col_of.copy()


def match(cost, limit: float) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """``(pairs, unpaired rows, unpaired columns)`` of the cheapest assignment of ``cost`` (rows to columns) in which
    a pair costs its entry and leaving a row or a column unpaired costs ``limit / 2``: a pair is made only when its cost
    is below ``limit``. Entries that are not finite (``inf``, ``nan``) may not be paired."""
    cost = np.asarray(cost, dtype=np.float64)
    n, m = cost.shape
    if n == 0 or m == 0:
        return [], list(range(n)), list(range(m))
    limit = float(limit)
    allowed = np.isfinite(cost) & (cost < limit)
    largest = float(np.abs(cost[allowed]).max()) if allowed.any() else 0.0
    big = 2.0 * (n + m) * (abs(limit) + largest) + 1.0  # dearer than any assignment without it: never chosen
    # (n + m) x (n + m): the real pairs, each row's own "unpaired" column, each column's own "unpaired" row, and the
    # unpaired rows and columns paired with each other at no cost
    extended = np.full((n + m, n + m), big)
    extended[:n, :m] = np.where(allowed, cost, big)
    extended[:n, m:][np.diag_indices(n)] = limit / 2.0
    extended[n:, :m][np.diag_indices(m)] = limit / 2.0
    extended[n:, m:] = 0.0
    rows, cols = linear_sum_assignment(extended)
    pairs = [(int(r), int(c)) for r, c in zip(rows, cols) if r < n and c < m and allowed[r, c]]
    paired_rows = {r for r, _ in pairs}
    paired_cols = {c for _, c in pairs}
    return pairs, [r for r in range(n) if r not in paired_rows], [c for c in range(m) if c not in paired_cols]
