from __future__ import annotations

import argparse
import matplotlib.pyplot as plt

from trace import TraceConfig
from trace.metrics import read_metric_file, to_float_list, metric_file_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot TRACE metrics from saved spreadsheet file."
    )

    p.add_argument("--robots", type=int, default=4, choices=[2, 4, 8])
    p.add_argument("--rows", type=int, default=20)
    p.add_argument("--cols", type=int, default=20)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--obstacle-ratio", type=float, default=0.10)
    p.add_argument("--generations", type=int, default=280)
    p.add_argument("--population", type=int, default=42)
    p.add_argument("--energy", type=int, default=95)
    p.add_argument("--metrics-dir", type=str, default="results")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    cfg = TraceConfig(
        rows=args.rows,
        cols=args.cols,
        n_robots=args.robots,
        seed=args.seed,
        obstacle_ratio=args.obstacle_ratio,
        generations=args.generations,
        population_size=args.population,
        energy_budget=args.energy,
    )

    file_path = metric_file_path(cfg, out_dir=args.metrics_dir)

    print(f"Reading metric file: {file_path}")

    rows = read_metric_file(cfg, out_dir=args.metrics_dir)

    steps = to_float_list(rows, "step")
    coverage = to_float_list(rows, "coverage")
    priority = to_float_list(rows, "priority_serviced")
    redundancy = to_float_list(rows, "redundancy")
    turns = to_float_list(rows, "total_turns")
    hidden_score = to_float_list(rows, "hidden_victim_score")

    generations = to_float_list(rows, "generation")
    global_cost = to_float_list(rows, "global_cost")
    global_priority = to_float_list(rows, "global_priority")

    # Plot 1: coverage and priority servicing
    plt.figure()
    plt.plot(steps, coverage, label="Coverage")
    plt.plot(steps, priority, label="Priority Serviced")
    plt.xlabel("Simulation Step")
    plt.ylabel("Ratio")
    plt.title("Coverage and Priority Servicing")
    plt.ylim(0, 1.05)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot 2: redundancy
    plt.figure()
    plt.plot(steps, redundancy, label="Redundancy")
    plt.xlabel("Simulation Step")
    plt.ylabel("Redundancy Ratio")
    plt.title("Coverage Redundancy")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot 3: total turns
    plt.figure()
    plt.plot(steps, turns, label="Total Turns")
    plt.xlabel("Simulation Step")
    plt.ylabel("Number of Turns")
    plt.title("Total Turns Over Time")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot 4: hidden victim discovery
    plt.figure()
    plt.plot(steps, hidden_score, label="Hidden Victim Score")
    plt.xlabel("Simulation Step")
    plt.ylabel("Victim Score")
    plt.title("Hidden Victim Discovery")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot 5: Delta RL-MA cost convergence
    plt.figure()
    plt.plot(generations, global_cost, label="Global Cost")
    plt.xlabel("Generation")
    plt.ylabel("Cost")
    plt.title("Delta RL-MA Cost Convergence")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot 6: Delta RL-MA priority convergence
    plt.figure()
    plt.plot(generations, global_priority, label="Priority Coverage")
    plt.xlabel("Generation")
    plt.ylabel("Priority Coverage Ratio")
    plt.title("Delta RL-MA Priority Coverage History")
    plt.ylim(0, 1.05)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()