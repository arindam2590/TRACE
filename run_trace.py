from __future__ import annotations
import argparse
from trace import TraceConfig, build_trace_plan
from trace.ui import run_ui
from trace.metrics import save_simulation_metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TRACE: target-aware multi-robot SAR coverage simulation")
    p.add_argument("--robots", type=int, default=4, choices=[2, 4, 8])
    p.add_argument("--rows", type=int, default=20)
    p.add_argument("--cols", type=int, default=20)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--obstacle-ratio", type=float, default=0.10)
    p.add_argument("--generations", type=int, default=280)
    p.add_argument("--population", type=int, default=42)
    p.add_argument("--energy", type=int, default=95)
    p.add_argument("--no-ui", action="store_true", help="Only build the plan and print final metrics.")
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

    print("Building TRACE plan. This may take a few seconds...")
    plan = build_trace_plan(cfg)
    print("TRACE plan built.")

    metrics_file = save_simulation_metrics(plan, cfg, out_dir="results")
    print(f"Simulation metrics saved to: {metrics_file}")

    print(f"Coverage: {100 * plan.metrics.coverage:.2f}%")
    print(f"Priority servicing: {100 * plan.metrics.priority_serviced:.2f}%")
    print(f"Redundancy: {plan.metrics.redundancy:.3f}")
    print(f"Mission time: {plan.metrics.mission_time}")
    print(f"Total turns: {plan.metrics.total_turns}")
    print(f"Hidden victim score: {plan.metrics.discovered_hidden_victims}")

    if not args.no_ui:
        run_ui(plan)


if __name__ == "__main__":
    main()
