from __future__ import annotations
from dataclasses import dataclass
from typing import List
import random
import numpy as np
from .config import TraceConfig, Cell
from .environment import TraceEnvironment
from .decomposition import PriorityAwareADARP, DecompositionResult
from .cvrp import build_instances, expand_route
from .delta_rl_memetic import DeltaRLMemeticSolver, SolverResult
from .metrics import compute_metrics, TraceMetrics

@dataclass
class TracePlan:
    env: TraceEnvironment
    decomposition: DecompositionResult
    solver_results: List[SolverResult]
    paths: List[List[Cell]]
    metrics: TraceMetrics
    global_cost_history: List[float]
    global_priority_history: List[float]


def build_trace_plan(cfg: TraceConfig) -> TracePlan:
    env = TraceEnvironment.generate(cfg)
    decomp = PriorityAwareADARP(env).run()
    instances = build_instances(env, decomp.regions)
    solver = DeltaRLMemeticSolver(cfg, random.Random(cfg.seed + 99))
    results: List[SolverResult] = []
    paths: List[List[Cell]] = []
    max_hist = 0
    for inst in instances:
        result = solver.solve(inst)
        results.append(result)
        paths.append(expand_route(inst, result.best_trips, env))
        max_hist = max(max_hist, len(result.cost_history))

    # Aggregated curves for right-panel graph.
    cost_hist, prio_hist = [], []
    for t in range(max_hist):
        csum = 0.0
        psum = 0.0
        count = 0
        for r in results:
            if r.cost_history:
                csum += r.cost_history[min(t, len(r.cost_history) - 1)]
                psum += r.priority_history[min(t, len(r.priority_history) - 1)]
                count += 1
        if count:
            cost_hist.append(csum)
            prio_hist.append(psum / count)

    metrics = compute_metrics(paths, env.free, env.priority_cells, env.hidden_victims)
    return TracePlan(env, decomp, results, paths, metrics, cost_hist, prio_hist)
