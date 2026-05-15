from __future__ import annotations
import random
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple
import numpy as np
from .config import TraceConfig, Cell
from .astar import bfs_distances

@dataclass
class TraceEnvironment:
    cfg: TraceConfig
    obstacles: Set[Cell]
    free: Set[Cell]
    starts: List[Cell]
    prior: np.ndarray
    priority_cells: Set[Cell]
    hidden_victims: Dict[Cell, int]

    @classmethod
    def generate(cls, cfg: TraceConfig) -> "TraceEnvironment":
        rng = random.Random(cfg.seed)
        np_rng = np.random.default_rng(cfg.seed)
        starts = cfg.robot_starts
        all_cells = {(r, c) for r in range(cfg.rows) for c in range(cfg.cols)}
        protected = set(starts)

        # Structured random obstacles, keeping a single connected component around starts.
        max_obs = int(cfg.rows * cfg.cols * cfg.obstacle_ratio)
        obstacles: Set[Cell] = set()
        candidates = list(all_cells - protected)
        rng.shuffle(candidates)
        for cell in candidates:
            if len(obstacles) >= max_obs:
                break
            obstacles.add(cell)
            free_trial = all_cells - obstacles
            if any(s not in free_trial for s in starts):
                obstacles.remove(cell)
                continue
            reachable = bfs_distances(starts[0], free_trial, cfg.rows, cfg.cols)
            if any(s not in reachable for s in starts) or len(reachable) != len(free_trial):
                obstacles.remove(cell)

        free = all_cells - obstacles

        # Population prior: smooth clustered heat-map with high-value SAR pockets.
        prior = np.zeros((cfg.rows, cfg.cols), dtype=float)
        cluster_count = max(4, int(cfg.rows * cfg.cols * cfg.high_priority_ratio / 8))
        centers = rng.sample(list(free), min(cluster_count, len(free)))
        for center in centers:
            amp = rng.uniform(0.65, 1.0)
            sigma = rng.uniform(1.4, 3.0)
            for cell in free:
                d2 = (cell[0] - center[0]) ** 2 + (cell[1] - center[1]) ** 2
                prior[cell] += amp * np.exp(-d2 / (2 * sigma * sigma))
        if prior.max() > 0:
            prior = prior / prior.max()
        for cell in obstacles:
            prior[cell] = 0.0

        free_sorted = sorted(free, key=lambda x: prior[x], reverse=True)
        n_priority = max(1, int(len(free) * cfg.high_priority_ratio))
        priority_cells = set(free_sorted[:n_priority])

        exploration = list(free - priority_cells)
        rng.shuffle(exploration)
        n_hidden = int(len(exploration) * cfg.hidden_victim_ratio)
        hidden_victims = {cell: rng.randint(1, 10) for cell in exploration[:n_hidden]}
        return cls(cfg, obstacles, free, starts, prior, priority_cells, hidden_victims)
