from dataclasses import dataclass
from typing import List, Tuple

Cell = Tuple[int, int]

@dataclass
class TraceConfig:
    rows: int = 20
    cols: int = 20
    n_robots: int = 4
    obstacle_ratio: float = 0.10
    high_priority_ratio: float = 0.10
    hidden_victim_ratio: float = 0.20
    priority_threshold: float = 0.75
    seed: int = 7

    # TRACE Stage 1: priority-aware A*-DARP
    darp_iterations: int = 90
    beta_start: float = 0.60
    beta_min: float = 0.20
    beta_decay: float = 0.95
    balance_lr: float = 0.22
    distance_eps: float = 1e-6

    # TRACE Stage 2: priority-weighted CVRP
    energy_budget: int = 95
    turn_cost: float = 0.25
    waypoint_weight: float = 5.0
    priority_reward_scale: float = 2.25
    early_service_fraction: float = 0.35

    # TRACE Stage 3: Delta RL-MA
    population_size: int = 42
    generations: int = 280
    local_search_depth: int = 8
    rl_alpha: float = 0.10
    rl_gamma: float = 0.90
    epsilon_start: float = 1.00
    epsilon_decay: float = 0.992
    epsilon_floor: float = 0.03
    violation_penalty: float = 1000.0
    panic_cost_window: int = 60
    panic_priority_window: int = 45
    panic_threshold: float = 0.003
    panic_duration: int = 25

    # UI
    cell_size: int = 28
    left_panel_width: int = 310
    right_panel_width: int = 420
    top_margin: int = 20
    fps: int = 18
    animate_every_n_frames: int = 2

    @property
    def robot_starts(self) -> List[Cell]:
        base = [
            (1, 1),
            (1, self.cols - 2),
            (self.rows - 2, 1),
            (self.rows - 2, self.cols - 2),
            (self.rows // 2, 1),
            (self.rows // 2, self.cols - 2),
            (1, self.cols // 2),
            (self.rows - 2, self.cols // 2),
        ]
        return base[: self.n_robots]
