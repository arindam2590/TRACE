from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Set, Tuple
from .config import Cell

@dataclass
class TraceMetrics:
    coverage: float
    priority_serviced: float
    redundancy: float
    total_turns: int
    mission_time: int
    discovered_hidden_victims: int


def count_turns(path: Sequence[Cell]) -> int:
    if len(path) < 3:
        return 0
    turns = 0
    prev_dir = (path[1][0] - path[0][0], path[1][1] - path[0][1])
    for a, b in zip(path[1:-1], path[2:]):
        cur = (b[0] - a[0], b[1] - a[1])
        if cur != prev_dir:
            turns += 1
        prev_dir = cur
    return turns


def compute_metrics(paths: Sequence[Sequence[Cell]], free: Set[Cell], priority: Set[Cell], hidden: Dict[Cell, int]) -> TraceMetrics:
    visits: Dict[Cell, int] = {}
    for p in paths:
        for c in p:
            if c in free:
                visits[c] = visits.get(c, 0) + 1
    covered = set(visits)
    redundant_steps = sum(max(0, v - 1) for v in visits.values())
    return TraceMetrics(
        coverage=len(covered) / max(len(free), 1),
        priority_serviced=len(covered & priority) / max(len(priority), 1),
        redundancy=redundant_steps / max(len(free), 1),
        total_turns=sum(count_turns(p) for p in paths),
        mission_time=max((len(p) for p in paths), default=0),
        discovered_hidden_victims=sum(hidden.get(c, 0) for c in covered),
    )
