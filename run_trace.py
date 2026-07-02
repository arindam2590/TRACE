from __future__ import annotations
import argparse
from trace import TraceConfig, build_trace_plan
from trace.ui import run_ui
from trace.metrics import save_simulation_metrics
from trace.astar_darp_baseline import AStarDARP
from trace.decomposition import PriorityAwareADARP
from trace.environment import TraceEnvironment
from plot_metrics import plot_aggregate_results, export_aggregate_tables
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TRACE: target-aware multi-robot SAR coverage simulation")
    p.add_argument("--robots", type=int, default=4, choices=[2, 4, 8])
    p.add_argument("--rows", type=int, default=20)
    p.add_argument("--cols", type=int, default=20)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--obstacle-ratio", type=float, default=0.10)
    p.add_argument("--generations", type=int, default=300)
    p.add_argument("--population", type=int, default=50)
    p.add_argument("--energy", type=int, default=95)
    p.add_argument("--no-ui", action="store_true", help="Only build the plan and print final metrics.")
    p.add_argument("--experiment", type=str, default="false", choices=["true", "false"], help="Run aggregate performance experiments.")
    p.add_argument("--mission-time-limit", type=int, default=None)
    p.add_argument("--energy-budget", type=int, default=None)
    p.add_argument("--victim-energy-cost", type=float, default=2.0)
    p.add_argument("--priority-victim-energy-cost", type=float, default=2.0)
    p.add_argument("--hidden-victim-energy-cost", type=float, default=3.0)
    return p.parse_args()
def main() -> None:
    args = parse_args()

    if args.experiment == "true":
        import csv
        import time
        import os
        import random
        import numpy as np
        
        from trace.cvrp import PWCVRPInstance, expand_route, build_instances
        from trace.delta_rl_memetic import DeltaRLMemeticSolver
        from trace.metrics import compute_metrics
        
        def truncate_path_by_mission_constraints(
            path,
            env,
            mission_time_limit,
            energy_budget,
            victim_energy_cost,
            priority_victim_energy_cost,
            hidden_victim_energy_cost,
        ):
            executed_path = []
            used_energy = 0.0
            for cell in path:
                if len(executed_path) >= mission_time_limit:
                    break
                
                base_cost = 1.0
                extra_cost = 0.0
                if cell in env.priority_cells:
                    extra_cost = max(extra_cost, priority_victim_energy_cost)
                if cell in env.hidden_victims:
                    extra_cost = max(extra_cost, hidden_victim_energy_cost)
                
                step_energy = base_cost + extra_cost
                if used_energy + step_energy > energy_budget:
                    break
                
                used_energy += step_energy
                executed_path.append(cell)
            return executed_path, used_energy
        
        map_sizes = [(20, 20), (40, 40), (80, 80)]
        robot_counts = [4, 8, 16, 32, 64]
        seeds = [7, 11, 19, 23, 31]
        methods = ["A*-DARP", "Delta RL-MA", "DARP+RL-GA", "TRACE"]
        
        results_dir = "results"
        os.makedirs(results_dir, exist_ok=True)
        csv_path = os.path.join(results_dir, "aggregate_performance_all_seeds.csv")
        summary_path = os.path.join(results_dir, "aggregate_performance_summary.csv")
        plots_dir = os.path.join(results_dir, "plots")
        
        all_rows = []
        total_runs = len(map_sizes) * len(robot_counts) * len(seeds)
        current_run = 0
        
        for size in map_sizes:
            for n_robots in robot_counts:
                for seed in seeds:
                    current_run += 1
                    print(f"[{current_run}/{total_runs}] Map: {size[0]}x{size[1]}, Robots: {n_robots}, Seed: {seed}")
                    
                    t_lim = args.mission_time_limit if args.mission_time_limit is not None else int(0.45 * size[0] * size[1] / n_robots)
                    e_bud = args.energy_budget if args.energy_budget is not None else int(0.50 * size[0] * size[1] / n_robots)
                    
                    cfg = TraceConfig(
                        rows=size[0],
                        cols=size[1],
                        n_robots=n_robots,
                        seed=seed,
                        obstacle_ratio=0.10,
                        energy_budget=e_bud,
                        mission_time_limit=t_lim,
                        victim_energy_cost=args.victim_energy_cost,
                        priority_victim_energy_cost=args.priority_victim_energy_cost,
                        hidden_victim_energy_cost=args.hidden_victim_energy_cost,
                    )
                    
                    # Generate one environment for both methods under the same seed
                    env = TraceEnvironment.generate(cfg)
                    
                    for method in methods:
                        t0 = time.perf_counter()
                        paths = []
                        
                        if method == "A*-DARP":
                            # 1. Baseline decomposition
                            decomp = AStarDARP(env).run()
                            # 2. Simple Nearest Neighbor routing (no priority, no memetic search)
                            for rid, region in enumerate(decomp.regions):
                                depot = env.starts[rid]
                                customers = sorted([c for c in region if c != depot])
                                remaining = set(customers)
                                last = depot
                                ordered = []
                                while remaining:
                                    nxt = min(remaining, key=lambda c: abs(c[0]-last[0]) + abs(c[1]-last[1]))
                                    ordered.append(nxt)
                                    remaining.remove(nxt)
                                # split into trips based on capacity
                                trips = []
                                current = []
                                load = 0
                                for cell in ordered:
                                    if load + 1 > cfg.energy_budget:
                                        if current:
                                            trips.append(current)
                                        current = [cell]
                                        load = 1
                                    else:
                                        current.append(cell)
                                        load += 1
                                if current:
                                    trips.append(current)
                                inst = PWCVRPInstance(rid, depot, customers, {c: 1.0 for c in customers}, cfg.energy_budget, region | {depot})
                                paths.append(expand_route(inst, trips, env))
                                
                        elif method == "Delta RL-MA":
                            # 1. Baseline decomposition
                            decomp = AStarDARP(env).run()
                            # 2. Priority-weighted CVRP construction
                            instances = build_instances(env, decomp.regions)
                            # 3. Delta RL-MA routing solver
                            solver = DeltaRLMemeticSolver(cfg, random.Random(cfg.seed + 99))
                            for inst in instances:
                                result = solver.solve(inst)
                                paths.append(expand_route(inst, result.best_trips, env))
                                
                        elif method == "DARP+RL-GA":
                            # 1. Baseline decomposition
                            decomp = AStarDARP(env).run()
                            # 2. Non-priority weighted CVRP construction (all weights 1.0)
                            instances = []
                            for rid, region in enumerate(decomp.regions):
                                depot = env.starts[rid]
                                customers = sorted([c for c in region if c != depot])
                                weights = {c: 1.0 for c in customers}
                                instances.append(PWCVRPInstance(rid, depot, customers, weights, cfg.energy_budget, region | {depot}))
                            # 3. Delta RL-MA routing solver (no priority reward)
                            solver = DeltaRLMemeticSolver(cfg, random.Random(cfg.seed + 99))
                            old_scale = cfg.priority_reward_scale
                            cfg.priority_reward_scale = 0.0
                            for inst in instances:
                                result = solver.solve(inst)
                                paths.append(expand_route(inst, result.best_trips, env))
                            cfg.priority_reward_scale = old_scale
                            
                        elif method == "TRACE":
                            # 1. Priority-aware decomposition
                            decomp = PriorityAwareADARP(env).run()
                            # 2. Priority-weighted CVRP construction
                            instances = build_instances(env, decomp.regions)
                            # 3. Delta RL-MA routing solver
                            solver = DeltaRLMemeticSolver(cfg, random.Random(cfg.seed + 99))
                            for inst in instances:
                                result = solver.solve(inst)
                                paths.append(expand_route(inst, result.best_trips, env))
                                
                        elapsed = time.perf_counter() - t0
                        
                        # Constrain paths using mission time and energy budget
                        executed_paths = []
                        energies_used = []
                        for path in paths:
                            exec_p, e_used = truncate_path_by_mission_constraints(
                                path=path,
                                env=env,
                                mission_time_limit=t_lim,
                                energy_budget=e_bud,
                                victim_energy_cost=args.victim_energy_cost,
                                priority_victim_energy_cost=args.priority_victim_energy_cost,
                                hidden_victim_energy_cost=args.hidden_victim_energy_cost,
                            )
                            executed_paths.append(exec_p)
                            energies_used.append(e_used)
                        
                        # Calculate performance metrics using executed_paths
                        metrics_obj = compute_metrics(executed_paths, env.free, env.priority_cells, env.hidden_victims)
                        coverage_percent = metrics_obj.coverage * 100
                        priority_service_percent = metrics_obj.priority_serviced * 100
                        
                        # Hidden victim discovery percent cells (based on executed paths)
                        covered_cells = set()
                        for p in executed_paths:
                            for c in p:
                                if c in env.free:
                                    covered_cells.add(c)
                        hidden_victim_discovery_percent = (len(covered_cells & set(env.hidden_victims)) / max(len(env.hidden_victims), 1)) * 100
                        
                        avg_turns_per_robot = metrics_obj.total_turns / n_robots
                        redundancy_rate = metrics_obj.redundancy
                        makespan_cells = metrics_obj.mission_time
                        
                        mean_energy_used = (np.mean(energies_used) / e_bud) * 100
                        max_energy_used = (np.max(energies_used) / e_bud) * 100
                        mean_remaining_energy = (np.mean([e_bud - e for e in energies_used]) / e_bud) * 100
                        
                        row = {
                            "method": method,
                            "rows": size[0],
                            "cols": size[1],
                            "map_size": f"{size[0]}x{size[1]}",
                            "n_robots": n_robots,
                            "seed": seed,
                            "coverage_percent": coverage_percent,
                            "priority_service_percent": priority_service_percent,
                            "hidden_victim_discovery_percent": hidden_victim_discovery_percent,
                            "avg_turns_per_robot": avg_turns_per_robot,
                            "redundancy_rate": redundancy_rate,
                            "makespan_cells": makespan_cells,
                            "runtime_seconds": elapsed,
                            "mean_energy_used": mean_energy_used,
                            "max_energy_used": max_energy_used,
                            "mean_remaining_energy": mean_remaining_energy
                        }
                        all_rows.append(row)
                        
        # Save raw all seeds CSV
        raw_fields = [
            "method", "rows", "cols", "map_size", "n_robots", "seed",
            "coverage_percent", "priority_service_percent", "hidden_victim_discovery_percent",
            "avg_turns_per_robot", "redundancy_rate", "makespan_cells", "runtime_seconds",
            "mean_energy_used", "max_energy_used", "mean_remaining_energy"
        ]
        with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=raw_fields)
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"Saved raw performance data to: {csv_path}")
        
        # Calculate summary metrics (mean & std)
        numeric_metrics = raw_fields[6:]
        summary_rows = []
        
        groups = {}
        for r in all_rows:
            g_key = (r["method"], r["rows"], r["cols"], r["map_size"], r["n_robots"])
            groups.setdefault(g_key, []).append(r)
            
        for g_key, rows_list in groups.items():
            sum_row = {
                "method": g_key[0],
                "rows": g_key[1],
                "cols": g_key[2],
                "map_size": g_key[3],
                "n_robots": g_key[4]
            }
            for metric in numeric_metrics:
                vals = [float(r[metric]) for r in rows_list]
                sum_row[f"{metric}_mean"] = np.mean(vals)
                sum_row[f"{metric}_std"] = np.std(vals)
            summary_rows.append(sum_row)
            
        summary_fields = ["method", "rows", "cols", "map_size", "n_robots"]
        for m in numeric_metrics:
            summary_fields.append(f"{m}_mean")
            summary_fields.append(f"{m}_std")
            
        with open(summary_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=summary_fields)
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"Saved summary statistics to: {summary_path}")
        
        # Generate LaTeX and CSV tables
        export_aggregate_tables(summary_rows, results_dir)
        print("Exported CSV and LaTeX tables successfully.")
        
        # Call plotting code
        plot_aggregate_results(csv_path, plots_dir)
        print("Generated aggregate experiment plots successfully.")
        return

    cfg = TraceConfig(
        rows=args.rows,
        cols=args.cols,
        n_robots=args.robots,
        seed=args.seed,
        obstacle_ratio=args.obstacle_ratio,
        generations=args.generations,
        population_size=args.population,
        energy_budget=args.energy if args.energy_budget is None else args.energy_budget,
        mission_time_limit=args.mission_time_limit,
        victim_energy_cost=args.victim_energy_cost,
        priority_victim_energy_cost=args.priority_victim_energy_cost,
        hidden_victim_energy_cost=args.hidden_victim_energy_cost,
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
