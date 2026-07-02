from __future__ import annotations
import math
import numpy as np
from typing import List, Set, Tuple

from .config import TraceConfig, Cell
from .astar import bfs_distances
from .environment import TraceEnvironment
from .decomposition import DecompositionResult

class AStarDARP:
    def __init__(self, env: TraceEnvironment):
        self.env = env
        self.cfg = env.cfg
        self.n = self.cfg.n_robots
        self.starts = env.starts
        self.dist_maps = [bfs_distances(s, env.free, self.cfg.rows, self.cfg.cols) for s in self.starts]

    def run(self) -> DecompositionResult:
        cfg = self.cfg
        assignment = -np.ones((cfg.rows, cfg.cols), dtype=int)
        multipliers = np.ones(self.n, dtype=float)
        free_cells = list(self.env.free)
        target_cells = len(free_cells) / self.n
        
        for it in range(cfg.darp_iterations):
            regions: List[Set[Cell]] = [set() for _ in range(self.n)]
            for cell in free_cells:
                best_robot, best_score = 0, math.inf
                for r in range(self.n):
                    d = self.dist_maps[r].get(cell, cfg.rows * cfg.cols)
                    score = multipliers[r] * (d + cfg.distance_eps)
                    if score < best_score:
                        best_score = score
                        best_robot = r
                assignment[cell] = best_robot
                regions[best_robot].add(cell)

            # Keep robot start cells in their own regions.
            for r, s in enumerate(self.starts):
                old = int(assignment[s])
                if old != r:
                    regions[old].discard(s)
                    regions[r].add(s)
                    assignment[s] = r

            counts = np.array([len(x) for x in regions], dtype=float)
            cell_error = (counts - target_cells) / max(target_cells, 1.0)
            
            multipliers *= np.clip(1.0 + cfg.balance_lr * cell_error, 0.60, 1.65)
            multipliers = np.clip(multipliers, 0.25, 6.0)

        final_regions = [set() for _ in range(self.n)]
        for cell in free_cells:
            r = int(assignment[cell])
            if 0 <= r < self.n:
                final_regions[r].add(cell)
                
        # Calculate cell counts and priority loads
        cell_counts = [len(x) for x in final_regions]
        priority_loads = [sum(float(self.env.prior[c]) for c in reg & self.env.priority_cells) for reg in final_regions]
        
        return DecompositionResult(
            assignment=assignment,
            regions=final_regions,
            cell_counts=cell_counts,
            priority_loads=priority_loads,
            history=[]
        )
