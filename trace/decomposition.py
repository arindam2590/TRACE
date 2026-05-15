from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple
import math
import numpy as np
from .config import TraceConfig, Cell
from .astar import bfs_distances
from .environment import TraceEnvironment

@dataclass
class DecompositionResult:
    assignment: np.ndarray
    regions: List[Set[Cell]]
    cell_counts: List[int]
    priority_loads: List[float]
    history: List[dict]

class PriorityAwareADARP:
    """Priority-aware A*-DARP implementation used as TRACE Stage 1.

    This implementation uses obstacle-aware A* / BFS distances as the base
    evaluation matrix and updates per-robot multipliers so overloaded robots
    become less attractive. The update combines cell-count fairness and
    victim-priority fairness, with the beta term annealed over iterations.
    """

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
        beta = cfg.beta_start
        free_cells = list(self.env.free)
        target_cells = len(free_cells) / self.n
        total_priority = sum(float(self.env.prior[c]) for c in self.env.priority_cells)
        target_priority = total_priority / self.n if self.n else 0.0
        history: List[dict] = []

        for it in range(cfg.darp_iterations):
            regions: List[Set[Cell]] = [set() for _ in range(self.n)]
            for cell in free_cells:
                best_robot, best_score = 0, math.inf
                pc = float(self.env.prior[cell])
                for r in range(self.n):
                    d = self.dist_maps[r].get(cell, cfg.rows * cfg.cols)
                    # Distance is primary. The priority term helps nearby high-priority
                    # cells settle early, while load multipliers enforce fairness.
                    score = multipliers[r] * (d + cfg.distance_eps) - beta * pc
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
            ploads = np.array([sum(float(self.env.prior[c]) for c in reg & self.env.priority_cells) for reg in regions])
            cell_error = (counts - target_cells) / max(target_cells, 1.0)
            if target_priority > 0:
                priority_error = (ploads - target_priority) / max(target_priority, 1e-9)
            else:
                priority_error = np.zeros_like(cell_error)

            combined = (1.0 - beta) * cell_error + beta * priority_error
            multipliers *= np.clip(1.0 + cfg.balance_lr * combined, 0.60, 1.65)
            multipliers = np.clip(multipliers, 0.25, 6.0)

            history.append({
                "iteration": it,
                "beta": beta,
                "counts": counts.tolist(),
                "priority_loads": ploads.tolist(),
                "max_cell_error": float(np.max(np.abs(cell_error))),
                "max_priority_error": float(np.max(np.abs(priority_error))) if target_priority > 0 else 0.0,
            })
            beta = max(cfg.beta_min, beta * cfg.beta_decay)

        final_regions = [set() for _ in range(self.n)]
        for cell in free_cells:
            r = int(assignment[cell])
            if 0 <= r < self.n:
                final_regions[r].add(cell)
        return DecompositionResult(
            assignment=assignment,
            regions=final_regions,
            cell_counts=[len(x) for x in final_regions],
            priority_loads=[sum(float(self.env.prior[c]) for c in reg & self.env.priority_cells) for reg in final_regions],
            history=history,
        )
