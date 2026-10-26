import heapq
import time
from collections import deque

import numpy as np

MOVES_4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MOVES_8 = MOVES_4 + [(-1, -1), (-1, 1), (1, -1), (1, 1)]
SQRT2 = 2 ** 0.5

ALGORITHMS = ["A*", "Dijkstra", "BFS", "Greedy Best-First"]


def heuristic(a, b, diagonal):
    """Manhattan for 4-way moves, Octile for 8-way moves (both admissible)."""
    dr, dc = abs(a[0] - b[0]), abs(a[1] - b[1])
    if diagonal:
        return (dr + dc) + (SQRT2 - 2) * min(dr, dc)
    return dr + dc


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


def find_path(grid, start, goal, algorithm="A*", diagonal=False):
    """
    A*                 : f = g + h   (cost so far + estimate to goal)
    Dijkstra           : f = g       (cost so far only)
    BFS                : explores level by level, ignores cost
    Greedy Best-First  : f = h       (estimate to goal only)
    """
    t0 = time.perf_counter()
    start, goal = tuple(start), tuple(goal)
    came_from = {start: None}
    g = {start: 0.0}
    visited = []
    found = False

    if algorithm == "BFS":
        queue = deque([start])
        while queue:
            current = queue.popleft()
            visited.append(current)
            if current == goal:
                found = True
                break
            for nxt, cost in neighbors(grid, current, diagonal):
                if nxt not in came_from:
                    came_from[nxt] = current
                    g[nxt] = g[current] + cost
                    queue.append(nxt)
    else:
        def priority(new_g, node):
            h = heuristic(node, goal, diagonal)
            if algorithm == "A*":
                return new_g + h
            if algorithm == "Dijkstra":
                return new_g
            return h                      # Greedy Best-First

        counter = 0
        heap = [(priority(0.0, start), 0, start)]
        closed = set()
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
                if algorithm == "Greedy Best-First" and nxt in g:
                    continue              # greedy never re-routes
                new_g = g[current] + cost
                if nxt not in g or new_g < g[nxt]:
                    g[nxt] = new_g
                    came_from[nxt] = current
                    counter += 1
                    heapq.heappush(heap, (priority(new_g, nxt), counter, nxt))

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
    for algo in ALGORITHMS:
        r = find_path(grid, (0, 0), (9, 9), algo)
        print(f"{algo:18} found={r['found']}  length={r['length']}  explored={len(r['visited'])}")
