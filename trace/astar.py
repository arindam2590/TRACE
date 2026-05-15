from __future__ import annotations
from collections import deque
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import heapq
import math

Cell = Tuple[int, int]


def neighbors(cell: Cell, rows: int, cols: int) -> Iterable[Cell]:
    r, c = cell
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            yield (nr, nc)


def manhattan(a: Cell, b: Cell) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def shortest_path(start: Cell, goal: Cell, free: set[Cell], rows: int, cols: int) -> List[Cell]:
    if start == goal:
        return [start]
    if start not in free or goal not in free:
        return []
    open_heap: list[tuple[float, int, Cell]] = []
    heapq.heappush(open_heap, (manhattan(start, goal), 0, start))
    came_from: Dict[Cell, Optional[Cell]] = {start: None}
    g_score = {start: 0}

    while open_heap:
        _, g, current = heapq.heappop(open_heap)
        if current == goal:
            path = [current]
            while came_from[path[-1]] is not None:
                path.append(came_from[path[-1]])
            path.reverse()
            return path
        if g > g_score.get(current, math.inf):
            continue
        for nb in neighbors(current, rows, cols):
            if nb not in free:
                continue
            ng = g + 1
            if ng < g_score.get(nb, math.inf):
                g_score[nb] = ng
                came_from[nb] = current
                heapq.heappush(open_heap, (ng + manhattan(nb, goal), ng, nb))
    return []


def bfs_distances(start: Cell, free: set[Cell], rows: int, cols: int) -> Dict[Cell, int]:
    q = deque([start])
    dist = {start: 0}
    while q:
        cur = q.popleft()
        for nb in neighbors(cur, rows, cols):
            if nb in free and nb not in dist:
                dist[nb] = dist[cur] + 1
                q.append(nb)
    return dist


def expand_waypoint_route(waypoints: Sequence[Cell], free: set[Cell], rows: int, cols: int) -> List[Cell]:
    if not waypoints:
        return []
    full: List[Cell] = [waypoints[0]]
    for a, b in zip(waypoints, waypoints[1:]):
        seg = shortest_path(a, b, free, rows, cols)
        if not seg:
            continue
        full.extend(seg[1:])
    return full
