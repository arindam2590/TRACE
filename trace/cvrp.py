from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Sequence, Set, Tuple
import math
from .config import Cell, TraceConfig
from .astar import manhattan, shortest_path, expand_waypoint_route
from .environment import TraceEnvironment

@dataclass
class PWCVRPInstance:
    robot_id: int
    depot: Cell
    customers: List[Cell]
    priority_weights: Dict[Cell, float]
    capacity: int
    region: Set[Cell]


def build_instances(env: TraceEnvironment, regions: List[Set[Cell]]) -> List[PWCVRPInstance]:
    cfg = env.cfg
    instances: List[PWCVRPInstance] = []
    for rid, region in enumerate(regions):
        depot = env.starts[rid]
        customers = sorted([c for c in region if c != depot])
        weights: Dict[Cell, float] = {}
        for c in customers:
            if c in env.priority_cells:
                weights[c] = cfg.waypoint_weight * (1.0 + float(env.prior[c]))
            elif c in env.hidden_victims:
                weights[c] = 1.0 + 0.25 * env.hidden_victims[c]
            else:
                weights[c] = 1.0
        instances.append(PWCVRPInstance(rid, depot, customers, weights, cfg.energy_budget, region | {depot}))
    return instances


def route_cost(order: Sequence[Cell], inst: PWCVRPInstance, cfg: TraceConfig) -> tuple[float, List[List[Cell]], int]:
    """Evaluate a flattened order as a multi-trip priority-weighted CVRP route."""
    trips: List[List[Cell]] = []
    current: List[Cell] = []
    load = 0
    prev = inst.depot
    total_distance = 0.0
    turns = 0
    previous_direction = None
    early_limit = max(1, int(len(inst.customers) * cfg.early_service_fraction))
    early_priority_gain = 0.0
    serviced_so_far = 0

    def finish_trip() -> None:
        nonlocal total_distance, prev, current, load, previous_direction
        if current:
            total_distance += abs(prev[0] - inst.depot[0]) + abs(prev[1] - inst.depot[1])
            trips.append(current.copy())
        current = []
        load = 0
        prev = inst.depot
        previous_direction = None

    for cell in order:
        if cell == inst.depot:
            continue
        demand = 1
        step_distance = abs(prev[0] - cell[0]) + abs(prev[1] - cell[1])
        if current and load + demand > inst.capacity:
            finish_trip()
            step_distance = abs(prev[0] - cell[0]) + abs(prev[1] - cell[1])
        direction = (0 if cell[0] == prev[0] else (1 if cell[0] > prev[0] else -1),
                     0 if cell[1] == prev[1] else (1 if cell[1] > prev[1] else -1))
        if previous_direction is not None and direction != previous_direction:
            turns += 1
        previous_direction = direction
        total_distance += step_distance
        current.append(cell)
        load += demand
        prev = cell
        serviced_so_far += 1
        if serviced_so_far <= early_limit:
            early_priority_gain += inst.priority_weights.get(cell, 1.0)
    finish_trip()

    weighted = total_distance + cfg.turn_cost * turns - cfg.priority_reward_scale * early_priority_gain
    return weighted, trips, turns


def trips_to_waypoint_sequence(inst: PWCVRPInstance, trips: Sequence[Sequence[Cell]]) -> List[Cell]:
    seq: List[Cell] = [inst.depot]
    for trip in trips:
        if seq[-1] != inst.depot:
            seq.append(inst.depot)
        for c in trip:
            seq.append(c)
        seq.append(inst.depot)
    return seq


def expand_route(inst: PWCVRPInstance, trips: Sequence[Sequence[Cell]], env: TraceEnvironment) -> List[Cell]:
    waypoints = trips_to_waypoint_sequence(inst, trips)
    # Use the complete free map for safe depot returns; routes still only service regional cells.
    return expand_waypoint_route(waypoints, env.free, env.cfg.rows, env.cfg.cols)
