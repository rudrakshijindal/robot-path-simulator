import heapq
import time

import numpy as np

MOVES_4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MOVES_8 = MOVES_4 + [(-1, -1), (-1, 1), (1, -1), (1, 1)]
SQRT2 = 2 ** 0.5

HEURISTICS = ["Manhattan", "Euclidean", "Chebyshev", "Octile"]


def heuristic(a, b, kind):
    """h(n): estimated remaining distance from cell a to the goal b."""
    dr, dc = abs(a[0] - b[0]), abs(a[1] - b[1])
    if kind == "Manhattan":
        return dr + dc
    if kind == "Euclidean":
        return (dr ** 2 + dc ** 2) ** 0.5
    if kind == "Chebyshev":
        return max(dr, dc)
    if kind == "Octile":
        return (dr + dc) + (SQRT2 - 2) * min(dr, dc)
    raise ValueError(f"Unknown heuristic: {kind}")


def neighbors(grid, node, diagonal):
    rows, cols = grid.shape
    for dr, dc in (MOVES_8 if diagonal else MOVES_4):
        r, c = node[0] + dr, node[1] + dc
        if not (0 <= r < rows and 0 <= c < cols) or grid[r, c] == 1:
            continue
        # no cutting through the corner of a wall
        if dr and dc and (grid[node[0] + dr, node[1]] == 1 or grid[node[0], node[1] + dc] == 1):
            continue
        yield (r, c), (SQRT2 if dr and dc else 1.0)


def find_path(grid, start, goal, heuristic_name="Manhattan", diagonal=False):
    """A* search:  f(n) = g(n) + h(n)"""
    t0 = time.perf_counter()
    start, goal = tuple(start), tuple(goal)
    came_from = {start: None}
    g = {start: 0.0}
    visited, closed = [], set()
    found = False
    counter = 0
    heap = [(heuristic(start, goal, heuristic_name), 0, start)]

    while heap:
        _, _, current = heapq.heappop(heap)
        if current in closed:
            continue
        closed.add(current)
        visited.append(current)
        if current == goal:
            found = True
            break
        for nxt, cost in neighbors(grid, current, diagonal):
            new_g = g[current] + cost
            if nxt not in g or new_g < g[nxt]:
                g[nxt] = new_g
                came_from[nxt] = current
                f = new_g + heuristic(nxt, goal, heuristic_name)
                counter += 1
                heapq.heappush(heap, (f, counter, nxt))

    path = []
    if found:
        node = goal
        while node is not None:
            path.append(node)
            node = came_from[node]
        path.reverse()

    return {
        "path": path,
        "visited": visited,
        "found": found,
        "length": round(g[goal], 2) if found else None,
        "time_ms": round((time.perf_counter() - t0) * 1000, 3),
    }


if __name__ == "__main__":
    grid = np.zeros((10, 10), dtype=int)
    grid[2:8, 5] = 1
    for h in HEURISTICS:
        r = find_path(grid, (0, 0), (9, 9), h, diagonal=True)
        print(f"{h:10} found={r['found']}  length={r['length']}  explored={len(r['visited'])}")