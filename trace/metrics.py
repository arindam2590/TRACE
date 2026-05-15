from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set

import csv
import re

from .config import Cell, TraceConfig


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


def compute_metrics(
    paths: Sequence[Sequence[Cell]],
    free: Set[Cell],
    priority: Set[Cell],
    hidden: Dict[Cell, int],
) -> TraceMetrics:
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


def build_metric_curves(
    paths: Sequence[Sequence[Cell]],
    free: Set[Cell],
    priority: Set[Cell],
    hidden: Dict[Cell, int],
) -> Dict[str, List[float]]:
    """
    Builds step-wise simulation metric curves.

    These curves represent how coverage, victim servicing,
    redundancy, turns, and hidden victim discovery evolve
    during the mission.
    """

    max_step = max((len(p) for p in paths), default=0)

    curves: Dict[str, List[float]] = {
        "step": [],
        "coverage": [],
        "priority_serviced": [],
        "redundancy": [],
        "total_turns": [],
        "mission_time": [],
        "hidden_victim_score": [],
    }

    for step in range(1, max_step + 1):
        partial_paths = [p[: min(step, len(p))] for p in paths]

        m = compute_metrics(
            partial_paths,
            free,
            priority,
            hidden,
        )

        curves["step"].append(step)
        curves["coverage"].append(m.coverage)
        curves["priority_serviced"].append(m.priority_serviced)
        curves["redundancy"].append(m.redundancy)
        curves["total_turns"].append(m.total_turns)
        curves["mission_time"].append(m.mission_time)
        curves["hidden_victim_score"].append(m.discovered_hidden_victims)

    return curves


def _safe_number(value: float | int) -> str:
    """
    Converts numbers into filename-safe strings.
    Example:
        0.10 -> 010
        2.25 -> 225
    """

    if isinstance(value, float):
        return f"{value:.3f}".replace(".", "")

    return str(value)


def metric_file_stem(cfg: TraceConfig) -> str:
    """
    Creates a filename stem based on simulation parameters.

    Example:
        TRACE_metrics_R20x20_UAV4_OBS010_HP010_HV020_E95_POP42_GEN280_SEED7
    """

    stem = (
        f"TRACE_metrics_"
        f"R{cfg.rows}x{cfg.cols}_"
        f"UAV{cfg.n_robots}_"
        f"OBS{_safe_number(cfg.obstacle_ratio)}_"
        f"HP{_safe_number(cfg.high_priority_ratio)}_"
        f"HV{_safe_number(cfg.hidden_victim_ratio)}_"
        f"E{cfg.energy_budget}_"
        f"POP{cfg.population_size}_"
        f"GEN{cfg.generations}_"
        f"SEED{cfg.seed}"
    )

    return re.sub(r"[^A-Za-z0-9_\\-]", "_", stem)


def metric_file_path(
    cfg: TraceConfig,
    out_dir: str | Path = "results",
    extension: str = ".csv",
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    return out_dir / f"{metric_file_stem(cfg)}{extension}"


def save_simulation_metrics(
    plan: Any,
    cfg: TraceConfig,
    out_dir: str | Path = "results",
) -> Path:
    """
    Saves all simulation metric data into a spreadsheet-compatible CSV file.

    The file contains:
        1. Simulation parameters
        2. Step-wise coverage metrics
        3. Step-wise priority servicing metrics
        4. Redundancy
        5. Turn count
        6. Hidden victim score
        7. Delta RL-MA cost convergence
        8. Delta RL-MA priority history
        9. Final summary metrics
    """

    file_path = metric_file_path(cfg, out_dir=out_dir, extension=".csv")

    curves = build_metric_curves(
        plan.paths,
        plan.env.free,
        plan.env.priority_cells,
        plan.env.hidden_victims,
    )

    max_rows = max(
        len(curves["step"]),
        len(plan.global_cost_history),
        len(plan.global_priority_history),
    )

    fieldnames = [
        # simulation parameters
        "rows",
        "cols",
        "n_robots",
        "seed",
        "obstacle_ratio",
        "high_priority_ratio",
        "hidden_victim_ratio",
        "energy_budget",
        "population_size",
        "generations",
        "local_search_depth",
        "turn_cost",

        # step-wise simulation metrics
        "step",
        "coverage",
        "priority_serviced",
        "redundancy",
        "total_turns",
        "mission_time",
        "hidden_victim_score",

        # solver history
        "generation",
        "global_cost",
        "global_priority",

        # final metrics
        "final_coverage",
        "final_priority_serviced",
        "final_redundancy",
        "final_total_turns",
        "final_mission_time",
        "final_hidden_victim_score",
    ]

    with file_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i in range(max_rows):
            row = {
                # simulation parameters
                "rows": cfg.rows,
                "cols": cfg.cols,
                "n_robots": cfg.n_robots,
                "seed": cfg.seed,
                "obstacle_ratio": cfg.obstacle_ratio,
                "high_priority_ratio": cfg.high_priority_ratio,
                "hidden_victim_ratio": cfg.hidden_victim_ratio,
                "energy_budget": cfg.energy_budget,
                "population_size": cfg.population_size,
                "generations": cfg.generations,
                "local_search_depth": cfg.local_search_depth,
                "turn_cost": cfg.turn_cost,

                # step-wise metrics
                "step": curves["step"][i] if i < len(curves["step"]) else "",
                "coverage": curves["coverage"][i] if i < len(curves["coverage"]) else "",
                "priority_serviced": curves["priority_serviced"][i] if i < len(curves["priority_serviced"]) else "",
                "redundancy": curves["redundancy"][i] if i < len(curves["redundancy"]) else "",
                "total_turns": curves["total_turns"][i] if i < len(curves["total_turns"]) else "",
                "mission_time": curves["mission_time"][i] if i < len(curves["mission_time"]) else "",
                "hidden_victim_score": curves["hidden_victim_score"][i] if i < len(curves["hidden_victim_score"]) else "",

                # solver history
                "generation": i + 1 if i < len(plan.global_cost_history) else "",
                "global_cost": plan.global_cost_history[i] if i < len(plan.global_cost_history) else "",
                "global_priority": plan.global_priority_history[i] if i < len(plan.global_priority_history) else "",

                # final metrics
                "final_coverage": plan.metrics.coverage,
                "final_priority_serviced": plan.metrics.priority_serviced,
                "final_redundancy": plan.metrics.redundancy,
                "final_total_turns": plan.metrics.total_turns,
                "final_mission_time": plan.metrics.mission_time,
                "final_hidden_victim_score": plan.metrics.discovered_hidden_victims,
            }

            writer.writerow(row)

    return file_path


def read_metric_file(
    cfg: TraceConfig,
    out_dir: str | Path = "results",
) -> List[Dict[str, str]]:
    """
    Reads the metrics spreadsheet using the same naming convention.
    """

    file_path = metric_file_path(cfg, out_dir=out_dir, extension=".csv")

    if not file_path.exists():
        raise FileNotFoundError(
            f"Metric file not found: {file_path}\\n"
            f"Run the simulation first to generate this file."
        )

    with file_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def to_float_list(rows: List[Dict[str, str]], column: str) -> List[float]:
    """
    Converts a numeric CSV column into a float list.
    Empty values are skipped.
    """

    values: List[float] = []

    for row in rows:
        value = row.get(column, "")

        if value is None or value == "":
            continue

        values.append(float(value))

    return values