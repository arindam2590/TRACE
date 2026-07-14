from __future__ import annotations
import argparse
from trace import TraceConfig, build_trace_plan
from trace.ui import run_ui
from trace.metrics import save_simulation_metrics
from trace.astar_darp_baseline import AStarDARP
from trace.decomposition import PriorityAwareADARP
from trace.environment import TraceEnvironment
from plot_metric import plot_aggregate_results, export_aggregate_tables
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
    p.add_argument("--study", type=str, default="performance", choices=["performance", "ablation", "beta_sensitivity", "stage3_sensitivity", "operator_ablation", "ablation_stats"], help="Study type for experiment")
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
        
        from trace.cvrp import PWCVRPInstance, expand_route, build_instances, route_cost, trips_to_waypoint_sequence
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
        
        def export_ablation_tables(summary_rows, tables_dir):
            groups = {}
            for r in summary_rows:
                key = (r["rows"], r["cols"], r["n_robots"])
                groups.setdefault(key, []).append(r)
                
            for key, rows_list in groups.items():
                rows, cols, n_robots = key
                a0_row = next((r for r in rows_list if r["variant"] == "A0: Full" and r["method"] == "TRACE"), None)
                if not a0_row:
                    continue
                    
                a0_prio = a0_row["priority_service_percent_mean"]
                a0_cov = a0_row["coverage_percent_mean"]
                
                for r in rows_list:
                    r["delta_priority_service"] = r["priority_service_percent_mean"] - a0_prio
                    r["delta_coverage"] = r["coverage_percent_mean"] - a0_cov
                    
                ablation_rows = [r for r in rows_list if r["method"] == "TRACE" and r["variant"] != "None"]
                
                a0_item = next((r for r in ablation_rows if r["variant"] == "A0: Full"), None)
                others = [r for r in ablation_rows if r["variant"] != "A0: Full"]
                
                others.sort(key=lambda r: r["delta_priority_service"])
                sorted_rows = [a0_item] + others
                
                csv_filename = os.path.join(tables_dir, f"ablation_{rows}x{cols}_robots_{n_robots}.csv")
                with open(csv_filename, mode="w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(["Variant", "Coverage", "Pri. Service", "Turns/robot", "dPri", "dCov"])
                    for r in sorted_rows:
                        var_name = r["variant"]
                        cov_str = f"{r['coverage_percent_mean']:.1f} \u00b1 {r['coverage_percent_std']:.1f}"
                        prio_str = f"{r['priority_service_percent_mean']:.1f} \u00b1 {r['priority_service_percent_std']:.1f}"
                        turns_str = f"{r['avg_turns_per_robot_mean']:.1f} \u00b1 {r['avg_turns_per_robot_std']:.1f}"
                        
                        if var_name == "A0: Full":
                            dpri_str = "\u2014"
                            dcov_str = "\u2014"
                        else:
                            dpri_str = f"{r['delta_priority_service']:.1f}"
                            dcov_str = f"{r['delta_coverage']:.1f}"
                            
                        writer.writerow([var_name, cov_str, prio_str, turns_str, dpri_str, dcov_str])
                        
                tex_filename = os.path.join(tables_dir, f"ablation_{rows}x{cols}_robots_{n_robots}.tex")
                with open(tex_filename, mode="w", encoding="utf-8") as f:
                    f.write("\\begin{tabular}{lccccc}\n")
                    f.write("\\hline\n")
                    f.write("Variant & Coverage & Pri. Service & Turns/robot & $\\Delta$ Pri. & $\\Delta$ Cov. \\\\\n")
                    f.write("\\hline\n")
                    for r in sorted_rows:
                        var_name = r["variant"].replace("_", "\\_")
                        cov_str = f"{r['coverage_percent_mean']:.1f} \\pm {r['coverage_percent_std']:.1f}"
                        prio_str = f"{r['priority_service_percent_mean']:.1f} \\pm {r['priority_service_percent_std']:.1f}"
                        turns_str = f"{r['avg_turns_per_robot_mean']:.1f} \\pm {r['avg_turns_per_robot_std']:.1f}"
                        
                        if var_name.startswith("A0:"):
                            cov_str = f"\\textbf{{{cov_str}}}"
                            prio_str = f"\\textbf{{{prio_str}}}"
                            turns_str = f"\\textbf{{{turns_str}}}"
                            dpri_str = "---"
                            dcov_str = "---"
                        else:
                            dpri_str = f"{r['delta_priority_service']:.1f}"
                            dcov_str = f"{r['delta_coverage']:.1f}"
                            
                        f.write(f"{var_name} & {cov_str} & {prio_str} & {turns_str} & {dpri_str} & {dcov_str} \\\\\n")
                    f.write("\\hline\n")
                    f.write("\\end{tabular}\n")
        
        def export_beta_sensitivity_tables(summary_rows, tables_dir):
            groups = {}
            for r in summary_rows:
                key = (r["rows"], r["cols"], r["n_robots"])
                groups.setdefault(key, []).append(r)
                
            for key, rows_list in groups.items():
                rows, cols, n_robots = key
                
                fixed_rows = [r for r in rows_list if r["beta_mode"] == "fixed"]
                fixed_rows.sort(key=lambda r: float(r["beta"]))
                
                annealed_row = next((r for r in rows_list if r["beta_mode"] == "annealed"), None)
                
                sorted_rows = fixed_rows.copy()
                if annealed_row:
                    sorted_rows.append(annealed_row)
                    
                best_prio_idx = -1
                best_prio_val = -1.0
                for idx, r in enumerate(sorted_rows):
                    if r["priority_service_percent_mean"] > best_prio_val:
                        best_prio_val = r["priority_service_percent_mean"]
                        best_prio_idx = idx
                        
                csv_filename = os.path.join(tables_dir, f"beta_sensitivity_{rows}x{cols}_robots_{n_robots}.csv")
                with open(csv_filename, mode="w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(["Beta", "Coverage", "Pri. Service", "Turns/robot", "Cell \u0394max"])
                    for idx, r in enumerate(sorted_rows):
                        beta_name = "Annealed" if r["beta_mode"] == "annealed" else r["beta"]
                        cov_str = f"{r['coverage_percent_mean']:.1f} \u00b1 {r['coverage_percent_std']:.1f}"
                        prio_str = f"{r['priority_service_percent_mean']:.1f} \u00b1 {r['priority_service_percent_std']:.1f}"
                        turns_str = f"{r['avg_turns_per_robot_mean']:.1f} \u00b1 {r['avg_turns_per_robot_std']:.1f}"
                        if r["beta_mode"] == "annealed":
                            cell_str = "\u2014"
                        else:
                            cell_str = f"{r['cell_delta_max_mean']:.1f} \u00b1 {r['cell_delta_max_std']:.1f}"
                        writer.writerow([beta_name, cov_str, prio_str, turns_str, cell_str])
                        
                tex_filename = os.path.join(tables_dir, f"beta_sensitivity_{rows}x{cols}_robots_{n_robots}.tex")
                with open(tex_filename, mode="w", encoding="utf-8") as f:
                    f.write("\\begin{tabular}{lccccc}\n")
                    f.write("\\hline\n")
                    f.write("$\\beta$ & Coverage & Pri. Service & Turns/robot & Cell $\\Delta_{\\max}$ \\\\\n")
                    f.write("\\hline\n")
                    for idx, r in enumerate(sorted_rows):
                        beta_name = "Annealed" if r["beta_mode"] == "annealed" else r["beta"]
                        cov_str = f"{r['coverage_percent_mean']:.1f} \\pm {r['coverage_percent_std']:.1f}"
                        prio_str = f"{r['priority_service_percent_mean']:.1f} \\pm {r['priority_service_percent_std']:.1f}"
                        turns_str = f"{r['avg_turns_per_robot_mean']:.1f} \\pm {r['avg_turns_per_robot_std']:.1f}"
                        if r["beta_mode"] == "annealed":
                            cell_str = "---"
                        else:
                            cell_str = f"{r['cell_delta_max_mean']:.1f} \\pm {r['cell_delta_max_std']:.1f}"
                            
                        if idx == best_prio_idx:
                            prio_str = f"\\textbf{{{prio_str}}}"
                            
                        f.write(f"{beta_name} & {cov_str} & {prio_str} & {turns_str} & {cell_str} \\\\\n")
                    f.write("\\hline\n")
                    f.write("\\end{tabular}\n")
        
        def export_stage3_sensitivity_tables(summary_rows, tables_dir):
            groups = {}
            for r in summary_rows:
                key = (r["rows"], r["cols"], r["n_robots"])
                groups.setdefault(key, []).append(r)
                
            omega_w_values = [1.0, 2.0, 3.0, 5.0, 8.0, 12.0]
            mu_values = [0, 10, 25, 50, 100, 200]
            
            for key, rows_list in groups.items():
                rows, cols, n_robots = key
                
                omega_lookup = {}
                mu_lookup = {}
                for r in rows_list:
                    if r["sensitivity_type"] == "omega_w":
                        omega_lookup[float(r["omega_w"])] = r
                    elif r["sensitivity_type"] == "mu":
                        mu_lookup[float(r["mu"])] = r
                
                best_omega_idx = -1
                best_omega_val = -1.0
                for idx, ow in enumerate(omega_w_values):
                    r = omega_lookup.get(ow)
                    if r and r["priority_service_percent_mean"] > best_omega_val:
                        best_omega_val = r["priority_service_percent_mean"]
                        best_omega_idx = idx
                        
                best_mu_idx = -1
                best_mu_val = -1.0
                for idx, mu_val in enumerate(mu_values):
                    r = mu_lookup.get(float(mu_val))
                    if r and r["priority_service_percent_mean"] > best_mu_val:
                        best_mu_val = r["priority_service_percent_mean"]
                        best_mu_idx = idx
                        
                csv_filename = os.path.join(tables_dir, f"stage3_sensitivity_{rows}x{cols}_robots_{n_robots}.csv")
                with open(csv_filename, mode="w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(["omega_w", "Pri. Serv. (omega_w)", "Cov. (omega_w)", "mu", "Pri. Serv. (mu)", "Cov. (mu)"])
                    for idx in range(6):
                        ow = omega_w_values[idx]
                        mu_val = mu_values[idx]
                        
                        orow = omega_lookup.get(ow)
                        mrow = mu_lookup.get(float(mu_val))
                        
                        oprio = f"{orow['priority_service_percent_mean']:.1f} \u00b1 {orow['priority_service_percent_std']:.1f}" if orow else "N/A"
                        ocov = f"{orow['coverage_percent_mean']:.1f} \u00b1 {orow['coverage_percent_std']:.1f}" if orow else "N/A"
                        
                        mprio = f"{mrow['priority_service_percent_mean']:.1f} \u00b1 {mrow['priority_service_percent_std']:.1f}" if mrow else "N/A"
                        mcov = f"{mrow['coverage_percent_mean']:.1f} \u00b1 {mrow['coverage_percent_std']:.1f}" if mrow else "N/A"
                        
                        writer.writerow([ow, oprio, ocov, mu_val, mprio, mcov])
                        
                tex_filename = os.path.join(tables_dir, f"stage3_sensitivity_{rows}x{cols}_robots_{n_robots}.tex")
                with open(tex_filename, mode="w", encoding="utf-8") as f:
                    f.write("\\begin{tabular}{rcccccc}\n")
                    f.write("\\hline\n")
                    f.write("$\\omega_W$ & Pri. Serv. & Cov. & & $\\mu$ & Pri. Serv. & Cov. \\\\\n")
                    f.write("\\hline\n")
                    for idx in range(6):
                        ow = omega_w_values[idx]
                        mu_val = mu_values[idx]
                        
                        orow = omega_lookup.get(ow)
                        mrow = mu_lookup.get(float(mu_val))
                        
                        oprio = f"{orow['priority_service_percent_mean']:.1f} \\pm {orow['priority_service_percent_std']:.1f}" if orow else "N/A"
                        ocov = f"{orow['coverage_percent_mean']:.1f} \\pm {orow['coverage_percent_std']:.1f}" if orow else "N/A"
                        
                        mprio = f"{mrow['priority_service_percent_mean']:.1f} \\pm {mrow['priority_service_percent_std']:.1f}" if mrow else "N/A"
                        mcov = f"{mrow['coverage_percent_mean']:.1f} \\pm {mrow['coverage_percent_std']:.1f}" if mrow else "N/A"
                        
                        if idx == best_omega_idx and orow:
                            oprio = f"\\textbf{{{oprio}}}"
                        if idx == best_mu_idx and mrow:
                            mprio = f"\\textbf{{{mprio}}}"
                            
                        f.write(f"{ow:.1f} & {oprio} & {ocov} & & {mu_val} & {mprio} & {mcov} \\\\\n")
                    f.write("\\hline\n")
                    f.write("\\end{tabular}\n")
        
        def export_operator_ablation_tables(summary_rows, tables_dir):
            groups = {}
            for r in summary_rows:
                key = (r["rows"], r["cols"], r["n_robots"])
                groups.setdefault(key, []).append(r)
                
            variant_order = ["None (full)", "-Swap", "-2opt", "-Relocate", "-BlockSwap", "-BlockRelocate", "-BothBlockOps"]
            
            variant_display = {
                "None (full)": "None (full)",
                "-Swap": "Swap",
                "-2opt": "2-opt",
                "-Relocate": "Relocate",
                "-BlockSwap": "Block Swap",
                "-BlockRelocate": "Block Relocate",
                "-BothBlockOps": "Both block ops"
            }
            
            function_lost_map = {
                "None (full)": "\u2014",
                "-Swap": "Inter-route fine tuning",
                "-2opt": "Intra-route untangling",
                "-Relocate": "Single-customer rebalancing",
                "-BlockSwap": "Multi-customer escape",
                "-BlockRelocate": "Multi-customer relocation",
                "-BothBlockOps": "Deep-minima escape"
            }
            
            for key, rows_list in groups.items():
                rows, cols, n_robots = key
                
                row_by_variant = {r["variant"]: r for r in rows_list}
                
                csv_filename = os.path.join(tables_dir, f"operator_ablation_{rows}x{cols}_robots_{n_robots}.csv")
                with open(csv_filename, mode="w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(["Operator removed", "Pri. Service", "Turns/robot", "Function lost"])
                    for var in variant_order:
                        r = row_by_variant.get(var)
                        op_name = variant_display[var]
                        fl_name = function_lost_map[var]
                        
                        prio_str = f"{r['priority_service_percent_mean']:.1f} \u00b1 {r['priority_service_percent_std']:.1f}" if r else "N/A"
                        turns_str = f"{r['avg_turns_per_robot_mean']:.1f} \u00b1 {r['avg_turns_per_robot_std']:.1f}" if r else "N/A"
                        
                        writer.writerow([op_name, prio_str, turns_str, fl_name])
                        
                tex_filename = os.path.join(tables_dir, f"operator_ablation_{rows}x{cols}_robots_{n_robots}.tex")
                with open(tex_filename, mode="w", encoding="utf-8") as f:
                    f.write("\\begin{tabular}{lccc}\n")
                    f.write("\\hline\n")
                    f.write("Operator removed & Pri. Service & Turns/robot & Function lost \\\\\n")
                    f.write("\\hline\n")
                    for var in variant_order:
                        r = row_by_variant.get(var)
                        op_name = variant_display[var]
                        fl_name = function_lost_map[var].replace("—", "---")
                        
                        prio_str = f"{r['priority_service_percent_mean']:.1f} \\pm {r['priority_service_percent_std']:.1f}" if r else "N/A"
                        turns_str = f"{r['avg_turns_per_robot_mean']:.1f} \\pm {r['avg_turns_per_robot_std']:.1f}" if r else "N/A"
                        
                        f.write(f"{op_name} & {prio_str} & {turns_str} & {fl_name} \\\\\n")
                    f.write("\\hline\n")
                    f.write("\\end{tabular}\n")
        
        def export_ablation_stats_tables(summary_rows, tables_dir):
            groups = {}
            for r in summary_rows:
                key = (r["rows"], r["cols"], r["n_robots"])
                groups.setdefault(key, []).append(r)
                
            variant_order = [
                "A1: -Stage1Pri",
                "A2: -Stage3Pri",
                "A3: -AllPri",
                "A4: -StateAug",
                "A5: -PriPanic",
                "A6: -AllPanic",
                "A7: -BlockOps",
                "A8: -QLearn",
                "A9: -A*DARP",
                "A10: -Anneal"
            ]
            
            for key, rows_list in groups.items():
                rows, cols, n_robots = key
                
                row_by_variant = {r["variant"]: r for r in rows_list}
                
                csv_filename = os.path.join(tables_dir, f"ablation_stats_{rows}x{cols}_robots_{n_robots}.csv")
                with open(csv_filename, mode="w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(["Variant", "Mean", "Delta (pp)", "t", "p", "Cohen's d"])
                    for var in variant_order:
                        r = row_by_variant.get(var)
                        if not r:
                            continue
                        sig = r["significant_alpha_005"]
                        var_name = f"{var}*" if sig else var
                        mean_val = f"{r['variant_mean']:.1f}"
                        delta_val = f"{r['delta_pp']:.1f}"
                        t_val = f"{r['t_statistic']:.2f}"
                        
                        p_raw = r['p_value']
                        p_val = "p < 0.001" if p_raw < 0.001 else f"{p_raw:.3f}"
                        
                        d_val = f"{r['cohens_d']:.2f} ({r['effect_size_label']})"
                        
                        writer.writerow([var_name, mean_val, delta_val, t_val, p_val, d_val])
                        
                tex_filename = os.path.join(tables_dir, f"ablation_stats_{rows}x{cols}_robots_{n_robots}.tex")
                with open(tex_filename, mode="w", encoding="utf-8") as f:
                    f.write("\\begin{tabular}{lccccc}\n")
                    f.write("\\hline\n")
                    f.write("Variant & Mean & $\\Delta$ (pp) & $t$ & $p$ & Cohen's $d$ \\\\\n")
                    f.write("\\hline\n")
                    for var in variant_order:
                        r = row_by_variant.get(var)
                        if not r:
                            continue
                        sig = r["significant_alpha_005"]
                        var_name = f"{var}*" if sig else var
                        
                        mean_val = f"{r['variant_mean']:.1f}"
                        delta_val = f"{r['delta_pp']:.1f}"
                        t_val = f"{r['t_statistic']:.2f}"
                        
                        p_raw = r['p_value']
                        p_val = "p < 0.001" if p_raw < 0.001 else f"{p_raw:.3f}"
                        
                        d_val = f"{r['cohens_d']:.2f} ({r['effect_size_label']})"
                        
                        if sig:
                            line = f"\\textbf{{{var_name}}} & \\textbf{{{mean_val}}} & \\textbf{{{delta_val}}} & \\textbf{{{t_val}}} & \\textbf{{{p_val}}} & \\textbf{{{d_val}}} \\\\\n"
                        else:
                            line = f"{var_name} & {mean_val} & {delta_val} & {t_val} & {p_val} & {d_val} \\\\\n"
                        f.write(line)
                    f.write("\\hline\n")
                    f.write("\\end{tabular}\n")
                    f.write("\\\\ \\footnotesize{* indicates statistical significance at \\(\\alpha=0.05\\).}\n")
        
        def run_ablation_study_internal(args, results_dir, tables_dir):
            map_sizes = [(20, 20), (40, 40), (80, 80)]
            robot_counts = [4, 8, 16, 32, 64]
            seeds = [7, 11, 19, 23, 31]
            
            methods_and_variants = [
                ("A*-DARP", "None"),
                ("Delta RL-MA", "None"),
                ("DARP+RL-GA", "None"),
                ("TRACE", "A0: Full"),
                ("TRACE", "A1: -Stage1Pri"),
                ("TRACE", "A2: -Stage3Pri"),
                ("TRACE", "A3: -AllPri"),
                ("TRACE", "A4: -StateAug"),
                ("TRACE", "A5: -PriPanic"),
                ("TRACE", "A6: -AllPanic"),
                ("TRACE", "A7: -BlockOps"),
                ("TRACE", "A8: -QLearn"),
                ("TRACE", "A9: -A*DARP"),
                ("TRACE", "A10: -Anneal"),
            ]
            
            all_ablation_rows = []
            
            total_runs = len(map_sizes) * len(robot_counts) * len(seeds) * len(methods_and_variants)
            current_run = 0
            
            for size in map_sizes:
                for n_robots in robot_counts:
                    for seed in seeds:
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
                        
                        env = TraceEnvironment.generate(cfg)
                        
                        path_cache = {}
                        def get_cached_path(a, b):
                            pair = (a, b)
                            if pair not in path_cache:
                                from trace.astar import shortest_path
                                path_cache[pair] = shortest_path(a, b, env.free, cfg.rows, cfg.cols)
                            return path_cache[pair]
                            
                        def expand_route_cached(inst, trips):
                            waypoints = trips_to_waypoint_sequence(inst, trips)
                            if not waypoints:
                                return []
                            full = [waypoints[0]]
                            for a, b in zip(waypoints, waypoints[1:]):
                                seg = get_cached_path(a, b)
                                if not seg:
                                    continue
                                full.extend(seg[1:])
                            return full
                        
                        decomp_cache = {}
                        
                        for method, variant in methods_and_variants:
                            current_run += 1
                            print(f"[{current_run}/{total_runs}] Map: {size[0]}x{size[1]}, Robots: {n_robots}, Seed: {seed}, Method: {method}, Variant: {variant}")
                            
                            t0 = time.perf_counter()
                            
                            if method == "A*-DARP":
                                from baselines.astar_darp import AStarDARP
                                solver = AStarDARP(env)
                                paths = solver.run()
                            elif method == "Delta RL-MA":
                                from baselines.delta_rl_ma import DeltaRLMA
                                solver = DeltaRLMA(env)
                                paths = solver.run()
                            elif method == "DARP+RL-GA":
                                from baselines.darp_rl_ga import DarpRlGA
                                solver = DarpRlGA(env)
                                paths = solver.run()
                            else:
                                if variant == "A1: -Stage1Pri":
                                    cfg.beta_start = 0.0
                                    cfg.beta_min = 0.0
                                    cfg.beta_decay = 1.0
                                    use_manhattan = False
                                    anneal = False
                                elif variant == "A3: -AllPri":
                                    cfg.beta_start = 0.0
                                    cfg.beta_min = 0.0
                                    cfg.beta_decay = 1.0
                                    use_manhattan = True
                                    anneal = False
                                else:
                                    cfg.beta_start = 0.60
                                    cfg.beta_min = 0.20
                                    cfg.beta_decay = 0.95
                                    use_manhattan = False
                                    anneal = True
                                    
                                decomp_key = (use_manhattan, anneal)
                                if decomp_key not in decomp_cache:
                                    decomp_cache[decomp_key] = PriorityAwareADARP(env, use_manhattan=use_manhattan, anneal=anneal).run()
                                decomp = decomp_cache[decomp_key]
                                
                                solver_args = {
                                    "state_aug": True,
                                    "pri_panic": True,
                                    "all_panic": True,
                                    "block_ops": True,
                                    "q_learn": True,
                                    "anneal": True
                                }
                                
                                if variant == "A4: -StateAug": solver_args["state_aug"] = False
                                elif variant == "A5: -PriPanic": solver_args["pri_panic"] = False
                                elif variant == "A6: -AllPanic": solver_args["all_panic"] = False
                                elif variant == "A7: -BlockOps": solver_args["block_ops"] = False
                                elif variant == "A8: -QLearn": solver_args["q_learn"] = False
                                elif variant == "A10: -Anneal": solver_args["anneal"] = False
                                    
                                solver = DeltaRLMemeticSolver(
                                    cfg,
                                    random.Random(cfg.seed + 99),
                                    state_aug=solver_args["state_aug"],
                                    pri_panic=solver_args["pri_panic"],
                                    all_panic=solver_args["all_panic"],
                                    block_ops=solver_args["block_ops"],
                                    q_learn=solver_args["q_learn"],
                                    anneal=solver_args["anneal"]
                                )
                                
                                if variant == "A2: -Stage3Pri":
                                    cfg.waypoint_weight = 1.0
                                    cfg.omega_w = 1.0
                                else:
                                    cfg.waypoint_weight = 5.0
                                    cfg.omega_w = 5.0
                                    
                                if variant == "A9: -A*DARP":
                                    regions = decomp.darp_regions
                                else:
                                    regions = decomp.regions
                                    
                                instances = build_instances(env, regions)
                                results = []
                                paths = []
                                for inst in instances:
                                    result = solver.solve(inst)
                                    results.append(result)
                                    paths.append(expand_route_cached(inst, result.best_trips))
                                    
                            elapsed = time.perf_counter() - t0
                            
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
                                
                            metrics_obj = compute_metrics(executed_paths, env.free, env.priority_cells, env.hidden_victims)
                            coverage_percent = metrics_obj.coverage * 100
                            priority_service_percent = metrics_obj.priority_serviced * 100
                            
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
                            mean_remaining_energy = (np.mean([e_bud - e for e in energies_used]) / e_bud) * 100
                            
                            all_ablation_rows.append({
                                "method": method,
                                "variant": variant,
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
                                "mean_energy_used": mean_energy_used,
                                "mean_remaining_energy": mean_remaining_energy,
                                "runtime_seconds": elapsed
                            })
                            
            raw_fields = [
                "method", "variant", "rows", "cols", "map_size", "n_robots", "seed",
                "coverage_percent", "priority_service_percent", "hidden_victim_discovery_percent",
                "avg_turns_per_robot", "redundancy_rate", "makespan_cells", "mean_energy_used",
                "mean_remaining_energy", "runtime_seconds"
            ]
            raw_ablation_path = os.path.join(results_dir, "ablation_study_all_seeds.csv")
            with open(raw_ablation_path, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=raw_fields)
                writer.writeheader()
                writer.writerows(all_ablation_rows)
            print(f"Saved raw ablation results to: {raw_ablation_path}")
            
            summary_groups = {}
            for r in all_ablation_rows:
                g_key = (r["method"], r["variant"], r["rows"], r["cols"], r["map_size"], r["n_robots"])
                summary_groups.setdefault(g_key, []).append(r)
                
            metrics_to_summarize = [
                "coverage_percent", "priority_service_percent", "hidden_victim_discovery_percent",
                "avg_turns_per_robot", "redundancy_rate", "makespan_cells", "mean_energy_used",
                "mean_remaining_energy", "runtime_seconds"
            ]
            
            summary_rows_temp = []
            for g_key, rows_list in summary_groups.items():
                sum_row = {
                    "method": g_key[0],
                    "variant": g_key[1],
                    "rows": g_key[2],
                    "cols": g_key[3],
                    "map_size": g_key[4],
                    "n_robots": g_key[5]
                }
                for m in metrics_to_summarize:
                    vals = [float(r[m]) for r in rows_list]
                    sum_row[f"{m}_mean"] = np.mean(vals)
                    sum_row[f"{m}_std"] = np.std(vals)
                summary_rows_temp.append(sum_row)
                
            a0_lookup = {}
            for r in summary_rows_temp:
                if r["method"] == "TRACE" and r["variant"] == "A0: Full":
                    key = (r["rows"], r["cols"], r["n_robots"])
                    a0_lookup[key] = (r["priority_service_percent_mean"], r["coverage_percent_mean"])
                    
            summary_rows_final = []
            for r in summary_rows_temp:
                key = (r["rows"], r["cols"], r["n_robots"])
                a0_vals = a0_lookup.get(key, (r["priority_service_percent_mean"], r["coverage_percent_mean"]))
                r["delta_priority_service"] = r["priority_service_percent_mean"] - a0_vals[0]
                r["delta_coverage"] = r["coverage_percent_mean"] - a0_vals[1]
                summary_rows_final.append(r)
                
            summary_fields = ["method", "variant", "rows", "cols", "map_size", "n_robots"]
            for m in metrics_to_summarize:
                summary_fields.append(f"{m}_mean")
                summary_fields.append(f"{m}_std")
            summary_fields.append("delta_priority_service")
            summary_fields.append("delta_coverage")
            
            summary_csv_path = os.path.join(results_dir, "ablation_study_summary.csv")
            with open(summary_csv_path, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=summary_fields)
                writer.writeheader()
                writer.writerows(summary_rows_final)
            print(f"Saved ablation summary statistics to: {summary_csv_path}")
            
            export_ablation_tables(summary_rows_final, tables_dir)
            print("Exported ablation CSV and LaTeX tables successfully.")
            
        if args.study == "ablation":
            run_ablation_study_internal(args, results_dir, tables_dir)
            return
            
        elif args.study == "beta_sensitivity":
            map_sizes = [(20, 20), (40, 40), (80, 80)]
            robot_counts = [4, 8, 16, 32, 64]
            seeds = [7, 11, 19, 23, 31]
            beta_values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
            
            results_dir = "results"
            tables_dir = os.path.join(results_dir, "tables")
            os.makedirs(results_dir, exist_ok=True)
            os.makedirs(tables_dir, exist_ok=True)
            
            all_sensitivity_rows = []
            
            total_runs = len(map_sizes) * len(robot_counts) * len(seeds) * (len(beta_values) + 1)
            current_run = 0
            
            for size in map_sizes:
                for n_robots in robot_counts:
                    for seed in seeds:
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
                        
                        env = TraceEnvironment.generate(cfg)
                        
                        path_cache = {}
                        def get_cached_path(a, b):
                            pair = (a, b)
                            if pair not in path_cache:
                                from trace.astar import shortest_path
                                path_cache[pair] = shortest_path(a, b, env.free, cfg.rows, cfg.cols)
                            return path_cache[pair]
                            
                        def expand_route_cached(inst, trips):
                            waypoints = trips_to_waypoint_sequence(inst, trips)
                            if not waypoints:
                                return []
                            full = [waypoints[0]]
                            for a, b in zip(waypoints, waypoints[1:]):
                                seg = get_cached_path(a, b)
                                if not seg:
                                    continue
                                full.extend(seg[1:])
                            return full
                            
                        variants = []
                        for b in beta_values:
                            variants.append((str(b), "fixed", b, b, 1.0))
                        variants.append(("annealed", "annealed", 0.60, 0.20, 0.95))
                        
                        for beta_name, beta_mode, b_start, b_min, b_decay in variants:
                            current_run += 1
                            print(f"[{current_run}/{total_runs}] Map: {size[0]}x{size[1]}, Robots: {n_robots}, Seed: {seed}, Beta: {beta_name} ({beta_mode})")
                            
                            t0 = time.perf_counter()
                            
                            cfg.beta_start = b_start
                            cfg.beta_min = b_min
                            cfg.beta_decay = b_decay
                            
                            decomp = PriorityAwareADARP(env).run()
                            
                            instances = build_instances(env, decomp.regions)
                            solver = DeltaRLMemeticSolver(cfg, random.Random(cfg.seed + 99))
                            
                            results = []
                            paths = []
                            for inst in instances:
                                result = solver.solve(inst)
                                results.append(result)
                                paths.append(expand_route_cached(inst, result.best_trips))
                                
                            elapsed = time.perf_counter() - t0
                            
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
                                
                            metrics_obj = compute_metrics(executed_paths, env.free, env.priority_cells, env.hidden_victims)
                            coverage_percent = metrics_obj.coverage * 100
                            priority_service_percent = metrics_obj.priority_serviced * 100
                            
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
                            mean_remaining_energy = (np.mean([e_bud - e for e in energies_used]) / e_bud) * 100
                            
                            cell_delta_max = float(np.max(decomp.cell_counts) - np.min(decomp.cell_counts))
                            priority_load_delta_max = float(np.max(decomp.priority_loads) - np.min(decomp.priority_loads))
                            
                            all_sensitivity_rows.append({
                                "beta": beta_name,
                                "beta_mode": beta_mode,
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
                                "cell_delta_max": cell_delta_max,
                                "priority_load_delta_max": priority_load_delta_max,
                                "mean_energy_used": mean_energy_used,
                                "mean_remaining_energy": mean_remaining_energy,
                                "runtime_seconds": elapsed,
                            })
                            
            raw_fields = [
                "beta", "beta_mode", "rows", "cols", "map_size", "n_robots", "seed",
                "coverage_percent", "priority_service_percent", "hidden_victim_discovery_percent",
                "avg_turns_per_robot", "redundancy_rate", "makespan_cells", "cell_delta_max",
                "priority_load_delta_max", "mean_energy_used", "mean_remaining_energy", "runtime_seconds"
            ]
            raw_path = os.path.join(results_dir, "beta_sensitivity_all_seeds.csv")
            with open(raw_path, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=raw_fields)
                writer.writeheader()
                writer.writerows(all_sensitivity_rows)
            print(f"Saved raw beta sensitivity results to: {raw_path}")
            
            summary_groups = {}
            for r in all_sensitivity_rows:
                g_key = (r["beta"], r["beta_mode"], r["rows"], r["cols"], r["map_size"], r["n_robots"])
                summary_groups.setdefault(g_key, []).append(r)
                
            metrics_to_summarize = [
                "coverage_percent", "priority_service_percent", "hidden_victim_discovery_percent",
                "avg_turns_per_robot", "redundancy_rate", "makespan_cells", "cell_delta_max",
                "priority_load_delta_max", "mean_energy_used", "mean_remaining_energy", "runtime_seconds"
            ]
            
            summary_rows = []
            for g_key, rows_list in summary_groups.items():
                sum_row = {
                    "beta": g_key[0],
                    "beta_mode": g_key[1],
                    "rows": g_key[2],
                    "cols": g_key[3],
                    "map_size": g_key[4],
                    "n_robots": g_key[5]
                }
                for m in metrics_to_summarize:
                    vals = [float(r[m]) for r in rows_list]
                    sum_row[f"{m}_mean"] = np.mean(vals)
                    sum_row[f"{m}_std"] = np.std(vals)
                summary_rows.append(sum_row)
                
            summary_fields = ["beta", "beta_mode", "rows", "cols", "map_size", "n_robots"]
            for m in metrics_to_summarize:
                summary_fields.append(f"{m}_mean")
                summary_fields.append(f"{m}_std")
                
            summary_csv_path = os.path.join(results_dir, "beta_sensitivity_summary.csv")
            with open(summary_csv_path, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=summary_fields)
                writer.writeheader()
                writer.writerows(summary_rows)
            print(f"Saved beta sensitivity summary statistics to: {summary_csv_path}")
            
            export_beta_sensitivity_tables(summary_rows, tables_dir)
            print("Exported beta sensitivity CSV and LaTeX tables successfully.")
            return
            
        elif args.study == "stage3_sensitivity":
            map_sizes = [(20, 20), (40, 40), (80, 80)]
            robot_counts = [4, 8, 16, 32, 64]
            seeds = [7, 11, 19, 23, 31]
            omega_w_values = [1.0, 2.0, 3.0, 5.0, 8.0, 12.0]
            mu_values = [0, 10, 25, 50, 100, 200]
            
            results_dir = "results"
            tables_dir = os.path.join(results_dir, "tables")
            os.makedirs(results_dir, exist_ok=True)
            os.makedirs(tables_dir, exist_ok=True)
            
            all_sensitivity_rows = []
            
            total_runs = len(map_sizes) * len(robot_counts) * len(seeds) * (len(omega_w_values) + len(mu_values))
            current_run = 0
            
            for size in map_sizes:
                for n_robots in robot_counts:
                    for seed in seeds:
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
                        
                        env = TraceEnvironment.generate(cfg)
                        
                        path_cache = {}
                        def get_cached_path(a, b):
                            pair = (a, b)
                            if pair not in path_cache:
                                from trace.astar import shortest_path
                                path_cache[pair] = shortest_path(a, b, env.free, cfg.rows, cfg.cols)
                            return path_cache[pair]
                            
                        def expand_route_cached(inst, trips):
                            waypoints = trips_to_waypoint_sequence(inst, trips)
                            if not waypoints:
                                return []
                            full = [waypoints[0]]
                            for a, b in zip(waypoints, waypoints[1:]):
                                seg = get_cached_path(a, b)
                                if not seg:
                                    continue
                                full.extend(seg[1:])
                            return full
                        
                        decomp = PriorityAwareADARP(env).run()
                        
                        variants = []
                        for ow in omega_w_values:
                            variants.append(("omega_w", "omega_w", ow, ow, 50.0))
                        for mu_val in mu_values:
                            variants.append(("mu", "mu", mu_val, 3.0, float(mu_val)))
                            
                        for sens_type, param_name, param_val, ow_val, mu_val in variants:
                            current_run += 1
                            print(f"[{current_run}/{total_runs}] Map: {size[0]}x{size[1]}, Robots: {n_robots}, Seed: {seed}, Sens: {sens_type}, Val: {param_val}")
                            
                            t0 = time.perf_counter()
                            
                            cfg.omega_w = ow_val
                            cfg.mu = mu_val
                            
                            instances = build_instances(env, decomp.regions)
                            solver = DeltaRLMemeticSolver(cfg, random.Random(cfg.seed + 99))
                            
                            results = []
                            paths = []
                            for inst in instances:
                                result = solver.solve(inst)
                                results.append(result)
                                paths.append(expand_route_cached(inst, result.best_trips))
                                
                            elapsed = time.perf_counter() - t0
                            
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
                                
                            metrics_obj = compute_metrics(executed_paths, env.free, env.priority_cells, env.hidden_victims)
                            coverage_percent = metrics_obj.coverage * 100
                            priority_service_percent = metrics_obj.priority_serviced * 100
                            
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
                            mean_remaining_energy = (np.mean([e_bud - e for e in energies_used]) / e_bud) * 100
                            
                            all_sensitivity_rows.append({
                                "sensitivity_type": sens_type,
                                "parameter_name": param_name,
                                "parameter_value": str(param_val),
                                "omega_w": ow_val,
                                "mu": mu_val,
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
                                "mean_energy_used": mean_energy_used,
                                "mean_remaining_energy": mean_remaining_energy,
                                "runtime_seconds": elapsed,
                            })
                            
            raw_fields = [
                "sensitivity_type", "parameter_name", "parameter_value", "omega_w", "mu", "rows", "cols", "map_size", "n_robots", "seed",
                "coverage_percent", "priority_service_percent", "hidden_victim_discovery_percent",
                "avg_turns_per_robot", "redundancy_rate", "makespan_cells", "mean_energy_used",
                "mean_remaining_energy", "runtime_seconds"
            ]
            raw_path = os.path.join(results_dir, "stage3_sensitivity_all_seeds.csv")
            with open(raw_path, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=raw_fields)
                writer.writeheader()
                writer.writerows(all_sensitivity_rows)
            print(f"Saved raw Stage-3 sensitivity results to: {raw_path}")
            
            summary_groups = {}
            for r in all_sensitivity_rows:
                g_key = (r["sensitivity_type"], r["parameter_name"], r["parameter_value"], r["omega_w"], r["mu"], r["rows"], r["cols"], r["map_size"], r["n_robots"])
                summary_groups.setdefault(g_key, []).append(r)
                
            metrics_to_summarize = [
                "coverage_percent", "priority_service_percent", "hidden_victim_discovery_percent",
                "avg_turns_per_robot", "redundancy_rate", "makespan_cells", "mean_energy_used",
                "mean_remaining_energy", "runtime_seconds"
            ]
            
            summary_rows = []
            for g_key, rows_list in summary_groups.items():
                sum_row = {
                    "sensitivity_type": g_key[0],
                    "parameter_name": g_key[1],
                    "parameter_value": g_key[2],
                    "omega_w": g_key[3],
                    "mu": g_key[4],
                    "rows": g_key[5],
                    "cols": g_key[6],
                    "map_size": g_key[7],
                    "n_robots": g_key[8]
                }
                for m in metrics_to_summarize:
                    vals = [float(r[m]) for r in rows_list]
                    sum_row[f"{m}_mean"] = np.mean(vals)
                    sum_row[f"{m}_std"] = np.std(vals)
                summary_rows.append(sum_row)
                
            summary_fields = ["sensitivity_type", "parameter_name", "parameter_value", "omega_w", "mu", "rows", "cols", "map_size", "n_robots"]
            for m in metrics_to_summarize:
                summary_fields.append(f"{m}_mean")
                summary_fields.append(f"{m}_std")
                
            summary_csv_path = os.path.join(results_dir, "stage3_sensitivity_summary.csv")
            with open(summary_csv_path, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=summary_fields)
                writer.writeheader()
                writer.writerows(summary_rows)
            print(f"Saved Stage-3 sensitivity summary statistics to: {summary_csv_path}")
            
            export_stage3_sensitivity_tables(summary_rows, tables_dir)
            print("Exported Stage-3 sensitivity CSV and LaTeX tables successfully.")
            return
            
        elif args.study == "operator_ablation":
            map_sizes = [(20, 20), (40, 40), (80, 80)]
            robot_counts = [4, 8, 16, 32, 64]
            seeds = [7, 11, 19, 23, 31]
            
            results_dir = "results"
            tables_dir = os.path.join(results_dir, "tables")
            os.makedirs(results_dir, exist_ok=True)
            os.makedirs(tables_dir, exist_ok=True)
            
            all_ablation_rows = []
            
            variant_details = {
                "None (full)": {
                    "ops": ["swap", "two_opt", "relocate", "block_swap", "block_relocate"],
                    "removed": "None"
                },
                "-Swap": {
                    "ops": ["two_opt", "relocate", "block_swap", "block_relocate"],
                    "removed": "Swap"
                },
                "-2opt": {
                    "ops": ["swap", "relocate", "block_swap", "block_relocate"],
                    "removed": "2-opt"
                },
                "-Relocate": {
                    "ops": ["swap", "two_opt", "block_swap", "block_relocate"],
                    "removed": "Relocate"
                },
                "-BlockSwap": {
                    "ops": ["swap", "two_opt", "relocate", "block_relocate"],
                    "removed": "Block Swap"
                },
                "-BlockRelocate": {
                    "ops": ["swap", "two_opt", "relocate", "block_swap"],
                    "removed": "Block Relocate"
                },
                "-BothBlockOps": {
                    "ops": ["swap", "two_opt", "relocate"],
                    "removed": "Both block ops"
                }
            }
            
            variant_order = ["None (full)", "-Swap", "-2opt", "-Relocate", "-BlockSwap", "-BlockRelocate", "-BothBlockOps"]
            
            total_runs = len(map_sizes) * len(robot_counts) * len(seeds) * len(variant_order)
            current_run = 0
            
            for size in map_sizes:
                for n_robots in robot_counts:
                    for seed in seeds:
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
                        
                        env = TraceEnvironment.generate(cfg)
                        
                        path_cache = {}
                        def get_cached_path(a, b):
                            pair = (a, b)
                            if pair not in path_cache:
                                from trace.astar import shortest_path
                                path_cache[pair] = shortest_path(a, b, env.free, cfg.rows, cfg.cols)
                            return path_cache[pair]
                            
                        def expand_route_cached(inst, trips):
                            waypoints = trips_to_waypoint_sequence(inst, trips)
                            if not waypoints:
                                return []
                            full = [waypoints[0]]
                            for a, b in zip(waypoints, waypoints[1:]):
                                seg = get_cached_path(a, b)
                                if not seg:
                                    continue
                                full.extend(seg[1:])
                            return full
                        
                        decomp = PriorityAwareADARP(env).run()
                        
                        for var in variant_order:
                            current_run += 1
                            details = variant_details[var]
                            print(f"[{current_run}/{total_runs}] Map: {size[0]}x{size[1]}, Robots: {n_robots}, Seed: {seed}, Variant: {var}")
                            
                            t0 = time.perf_counter()
                            
                            instances = build_instances(env, decomp.regions)
                            solver = DeltaRLMemeticSolver(cfg, random.Random(cfg.seed + 99), allowed_operators=details["ops"])
                            
                            results = []
                            paths = []
                            for inst in instances:
                                result = solver.solve(inst)
                                results.append(result)
                                paths.append(expand_route_cached(inst, result.best_trips))
                                
                            elapsed = time.perf_counter() - t0
                            
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
                                
                            metrics_obj = compute_metrics(executed_paths, env.free, env.priority_cells, env.hidden_victims)
                            coverage_percent = metrics_obj.coverage * 100
                            priority_service_percent = metrics_obj.priority_serviced * 100
                            
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
                            mean_remaining_energy = (np.mean([e_bud - e for e in energies_used]) / e_bud) * 100
                            
                            all_ablation_rows.append({
                                "variant": var,
                                "operator_removed": details["removed"],
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
                                "mean_energy_used": mean_energy_used,
                                "mean_remaining_energy": mean_remaining_energy,
                                "runtime_seconds": elapsed,
                            })
                            
            raw_fields = [
                "variant", "operator_removed", "rows", "cols", "map_size", "n_robots", "seed",
                "coverage_percent", "priority_service_percent", "hidden_victim_discovery_percent",
                "avg_turns_per_robot", "redundancy_rate", "makespan_cells", "mean_energy_used",
                "mean_remaining_energy", "runtime_seconds"
            ]
            raw_path = os.path.join(results_dir, "operator_ablation_all_seeds.csv")
            with open(raw_path, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=raw_fields)
                writer.writeheader()
                writer.writerows(all_ablation_rows)
            print(f"Saved raw operator ablation results to: {raw_path}")
            
            summary_groups = {}
            for r in all_ablation_rows:
                g_key = (r["variant"], r["operator_removed"], r["rows"], r["cols"], r["map_size"], r["n_robots"])
                summary_groups.setdefault(g_key, []).append(r)
                
            metrics_to_summarize = [
                "coverage_percent", "priority_service_percent", "hidden_victim_discovery_percent",
                "avg_turns_per_robot", "redundancy_rate", "makespan_cells", "mean_energy_used",
                "mean_remaining_energy", "runtime_seconds"
            ]
            
            summary_rows = []
            for g_key, rows_list in summary_groups.items():
                sum_row = {
                    "variant": g_key[0],
                    "operator_removed": g_key[1],
                    "rows": g_key[2],
                    "cols": g_key[3],
                    "map_size": g_key[4],
                    "n_robots": g_key[5]
                }
                for m in metrics_to_summarize:
                    vals = [float(r[m]) for r in rows_list]
                    sum_row[f"{m}_mean"] = np.mean(vals)
                    sum_row[f"{m}_std"] = np.std(vals)
                summary_rows.append(sum_row)
                
            summary_fields = ["variant", "operator_removed", "rows", "cols", "map_size", "n_robots"]
            for m in metrics_to_summarize:
                summary_fields.append(f"{m}_mean")
                summary_fields.append(f"{m}_std")
                
            summary_csv_path = os.path.join(results_dir, "operator_ablation_summary.csv")
            with open(summary_csv_path, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=summary_fields)
                writer.writeheader()
                writer.writerows(summary_rows)
            print(f"Saved operator ablation summary statistics to: {summary_csv_path}")
            
            export_operator_ablation_tables(summary_rows, tables_dir)
            print("Exported operator ablation CSV and LaTeX tables successfully.")
            return
            
        elif args.study == "ablation_stats":
            import scipy.stats
            
            results_dir = "results"
            tables_dir = os.path.join(results_dir, "tables")
            os.makedirs(results_dir, exist_ok=True)
            os.makedirs(tables_dir, exist_ok=True)
            
            raw_ablation_path = os.path.join(results_dir, "ablation_study_all_seeds.csv")
            if not os.path.exists(raw_ablation_path):
                print(f"{raw_ablation_path} not found. Running ablation study first to generate seed-level data...")
                run_ablation_study_internal(args, results_dir, tables_dir)
                
            rows_list = []
            with open(raw_ablation_path, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    if r["method"] == "TRACE":
                        rows_list.append(r)
                        
            groups = {}
            for r in rows_list:
                key = (int(r["rows"]), int(r["cols"]), r["map_size"], int(r["n_robots"]))
                groups.setdefault(key, []).append(r)
                
            variant_order = [
                "A1: -Stage1Pri",
                "A2: -Stage3Pri",
                "A3: -AllPri",
                "A4: -StateAug",
                "A5: -PriPanic",
                "A6: -AllPanic",
                "A7: -BlockOps",
                "A8: -QLearn",
                "A9: -A*DARP",
                "A10: -Anneal"
            ]
            
            all_stats_rows = []
            
            for key, group_rows in groups.items():
                rows, cols, map_size, n_robots = key
                
                variant_vals = {}
                for r in group_rows:
                    v = r["variant"]
                    val = float(r["priority_service_percent"])
                    variant_vals.setdefault(v, []).append(val)
                    
                a0_values = variant_vals.get("A0: Full", [])
                if not a0_values:
                    print(f"Warning: A0: Full not found for group {key}. Skipping stats.")
                    continue
                    
                a0_mean = np.mean(a0_values)
                n0 = len(a0_values)
                s0 = np.std(a0_values, ddof=1)
                
                for v in variant_order:
                    v_values = variant_vals.get(v, [])
                    if not v_values:
                        print(f"Warning: Variant {v} not found for group {key}. Skipping.")
                        continue
                        
                    v_mean = np.mean(v_values)
                    nv = len(v_values)
                    sv = np.std(v_values, ddof=1)
                    
                    delta_pp = v_mean - a0_mean
                    
                    res = scipy.stats.ttest_ind(a0_values, v_values, equal_var=False)
                    t_stat = res.statistic
                    p_val = res.pvalue
                    if np.isnan(t_stat):
                        t_stat = 0.0
                    if np.isnan(p_val):
                        p_val = 1.0
                        
                    if s0 == 0.0 and sv == 0.0:
                        d = 0.0
                    else:
                        sp = np.sqrt(((n0 - 1) * s0**2 + (nv - 1) * sv**2) / (n0 + nv - 2))
                        if sp == 0.0:
                            d = 0.0
                        else:
                            d = (a0_mean - v_mean) / sp
                            
                    abs_d = abs(d)
                    if abs_d < 0.2:
                        label = "negligible"
                    elif abs_d < 0.5:
                        label = "small"
                    elif abs_d < 0.8:
                        label = "medium"
                    else:
                        label = "large"
                        
                    sig = (p_val < 0.05)
                    
                    all_stats_rows.append({
                        "variant": v,
                        "rows": rows,
                        "cols": cols,
                        "map_size": map_size,
                        "n_robots": n_robots,
                        "metric": "priority_service_percent",
                        "a0_mean": a0_mean,
                        "variant_mean": v_mean,
                        "delta_pp": delta_pp,
                        "t_statistic": t_stat,
                        "p_value": p_val,
                        "cohens_d": d,
                        "effect_size_label": label,
                        "significant_alpha_005": sig
                    })
                    
            raw_fields = [
                "variant", "rows", "cols", "map_size", "n_robots", "metric", "a0_mean", "variant_mean",
                "delta_pp", "t_statistic", "p_value", "cohens_d", "effect_size_label", "significant_alpha_005"
            ]
            raw_stats_path = os.path.join(results_dir, "ablation_statistical_significance.csv")
            with open(raw_stats_path, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=raw_fields)
                writer.writeheader()
                writer.writerows(all_stats_rows)
            print(f"Saved raw statistical significance results to: {raw_stats_path}")
            
            export_ablation_stats_tables(all_stats_rows, tables_dir)
            print("Exported statistical CSV and LaTeX tables successfully.")
            return
            
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
        all_convergence_rows = []
        all_panic_events = []
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
                    
                    path_cache = {}
                    def get_cached_path(a, b):
                        pair = (a, b)
                        if pair not in path_cache:
                            from trace.astar import shortest_path
                            path_cache[pair] = shortest_path(a, b, env.free, cfg.rows, cfg.cols)
                        return path_cache[pair]
                        
                    def expand_route_cached(inst, trips):
                        waypoints = trips_to_waypoint_sequence(inst, trips)
                        if not waypoints:
                            return []
                        full = [waypoints[0]]
                        for a, b in zip(waypoints, waypoints[1:]):
                            seg = get_cached_path(a, b)
                            if not seg:
                                continue
                            full.extend(seg[1:])
                        return full
                    
                    for method in methods:
                        t0 = time.perf_counter()
                        paths = []
                        results = []
                        instances = []
                        
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
                                paths.append(expand_route_cached(inst, trips))
                                
                        elif method == "Delta RL-MA":
                            # 1. Baseline decomposition
                            decomp = AStarDARP(env).run()
                            # 2. Priority-weighted CVRP construction
                            instances = build_instances(env, decomp.regions)
                            # 3. Delta RL-MA routing solver
                            solver = DeltaRLMemeticSolver(cfg, random.Random(cfg.seed + 99))
                            for inst in instances:
                                result = solver.solve(inst)
                                results.append(result)
                                paths.append(expand_route_cached(inst, result.best_trips))
                                
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
                                results.append(result)
                                paths.append(expand_route_cached(inst, result.best_trips))
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
                                results.append(result)
                                paths.append(expand_route_cached(inst, result.best_trips))
                                
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
                        
                        # Convergence behaviors logging
                        if method == "A*-DARP":
                            final_cost = 0.0
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
                                inst = PWCVRPInstance(rid, depot, customers, {c: 1.0 for c in customers}, cfg.energy_budget, region | {depot})
                                c_cost, _, _ = route_cost(ordered, inst, cfg)
                                final_cost += c_cost
                                
                            all_convergence_rows.append({
                                "method": method,
                                "rows": size[0],
                                "cols": size[1],
                                "map_size": f"{size[0]}x{size[1]}",
                                "n_robots": n_robots,
                                "seed": seed,
                                "generation": 0,
                                "best_cost": final_cost,
                                "mean_cost": final_cost,
                                "priority_coverage_ratio": priority_service_percent / 100.0,
                                "coverage_percent": coverage_percent,
                                "priority_service_percent": priority_service_percent,
                                "hidden_victim_discovery_percent": hidden_victim_discovery_percent,
                                "avg_turns_per_robot": avg_turns_per_robot,
                                "redundancy_rate": redundancy_rate,
                                "makespan_cells": makespan_cells,
                                "mean_energy_used": mean_energy_used,
                                "mean_remaining_energy": mean_remaining_energy,
                                "panic_event": 0,
                                "panic_type": "none"
                            })
                        else:
                            num_gens = cfg.generations
                            team_best_cost_hist = []
                            team_rho_W_hist = []
                            gen_metrics = []
                            
                            for gen in range(num_gens):
                                gen_best_cost = sum(res.cost_history[gen] for res in results)
                                gen_mean_cost = sum(res.mean_cost_history[gen] for res in results)
                                gen_trips = [res.best_trips_history[gen] for res in results]
                                gen_paths = [expand_route_cached(instances[rid], gen_trips[rid]) for rid in range(n_robots)]
                                
                                gen_exec_paths = []
                                gen_energies_used = []
                                for r_path in gen_paths:
                                    exec_p, e_used = truncate_path_by_mission_constraints(
                                        path=r_path,
                                        env=env,
                                        mission_time_limit=t_lim,
                                        energy_budget=e_bud,
                                        victim_energy_cost=args.victim_energy_cost,
                                        priority_victim_energy_cost=args.priority_victim_energy_cost,
                                        hidden_victim_energy_cost=args.hidden_victim_energy_cost,
                                    )
                                    gen_exec_paths.append(exec_p)
                                    gen_energies_used.append(e_used)
                                    
                                m_obj = compute_metrics(gen_exec_paths, env.free, env.priority_cells, env.hidden_victims)
                                g_cov = m_obj.coverage * 100
                                g_prio = m_obj.priority_serviced * 100
                                
                                cov_cells = set()
                                for p in gen_exec_paths:
                                    for c in p:
                                        if c in env.free:
                                            cov_cells.add(c)
                                g_hidden = (len(cov_cells & set(env.hidden_victims)) / max(len(env.hidden_victims), 1)) * 100
                                
                                g_turns = m_obj.total_turns / n_robots
                                g_red = m_obj.redundancy
                                g_make = m_obj.mission_time
                                
                                g_energy_used = (np.mean(gen_energies_used) / e_bud) * 100
                                g_remaining_energy = (np.mean([e_bud - e for e in gen_energies_used]) / e_bud) * 100
                                
                                priority_coverage_ratio = m_obj.priority_serviced
                                
                                team_best_cost_hist.append(gen_best_cost)
                                team_rho_W_hist.append(priority_coverage_ratio)
                                
                                gen_metrics.append({
                                    "generation": gen,
                                    "best_cost": gen_best_cost,
                                    "mean_cost": gen_mean_cost,
                                    "priority_coverage_ratio": priority_coverage_ratio,
                                    "coverage_percent": g_cov,
                                    "priority_service_percent": g_prio,
                                    "hidden_victim_discovery_percent": g_hidden,
                                    "avg_turns_per_robot": g_turns,
                                    "redundancy_rate": g_red,
                                    "makespan_cells": g_make,
                                    "mean_energy_used": g_energy_used,
                                    "mean_remaining_energy": g_remaining_energy,
                                })
                                
                            # Panic window detection
                            panic_until = -1
                            g_rho = 0
                            w_c = cfg.panic_cost_window
                            w_prio = cfg.panic_priority_window
                            theta_c = cfg.panic_threshold
                            panic_duration = cfg.panic_duration
                            
                            for gen in range(num_gens):
                                if gen > 0 and team_rho_W_hist[gen] > team_rho_W_hist[gen - 1]:
                                    g_rho = gen
                                    
                                panic_event = 0
                                panic_type = "none"
                                
                                if gen >= panic_until:
                                    cost_triggered = False
                                    if gen >= w_c:
                                        old_c = team_best_cost_hist[gen - w_c]
                                        new_c = team_best_cost_hist[gen]
                                        rel_diff = abs(old_c - new_c) / max(abs(old_c), 10**-9)
                                        if rel_diff < theta_c:
                                            cost_triggered = True
                                            
                                    prio_triggered = False
                                    if (gen - g_rho) >= w_prio and team_rho_W_hist[gen] < 0.999:
                                        prio_triggered = True
                                        
                                    if cost_triggered and prio_triggered:
                                        panic_event = 1
                                        panic_type = "both"
                                        panic_until = gen + panic_duration
                                    elif cost_triggered:
                                        panic_event = 1
                                        panic_type = "cost_driven"
                                        panic_until = gen + panic_duration
                                    elif prio_triggered:
                                        panic_event = 1
                                        panic_type = "priority_driven"
                                        panic_until = gen + panic_duration
                                        
                                if panic_event == 1:
                                    cost_before = team_best_cost_hist[gen]
                                    rho_before = team_rho_W_hist[gen]
                                    after_gen = min(gen + panic_duration, num_gens - 1)
                                    cost_after = team_best_cost_hist[after_gen]
                                    rho_after = team_rho_W_hist[after_gen]
                                    cost_reduction_percent = ((cost_before - cost_after) / max(abs(cost_before), 10**-9)) * 100
                                    rho_improvement = rho_after - rho_before
                                    
                                    all_panic_events.append({
                                        "method": method,
                                        "rows": size[0],
                                        "cols": size[1],
                                        "map_size": f"{size[0]}x{size[1]}",
                                        "n_robots": n_robots,
                                        "seed": seed,
                                        "generation": gen,
                                        "panic_type": panic_type,
                                        "cost_before": cost_before,
                                        "cost_after": cost_after,
                                        "cost_reduction_percent": cost_reduction_percent,
                                        "rho_before": rho_before,
                                        "rho_after": rho_after,
                                        "rho_improvement": rho_improvement
                                    })
                                    
                                m_d = gen_metrics[gen]
                                all_convergence_rows.append({
                                    "method": method,
                                    "rows": size[0],
                                    "cols": size[1],
                                    "map_size": f"{size[0]}x{size[1]}",
                                    "n_robots": n_robots,
                                    "seed": seed,
                                    "generation": gen,
                                    "best_cost": m_d["best_cost"],
                                    "mean_cost": m_d["mean_cost"],
                                    "priority_coverage_ratio": m_d["priority_coverage_ratio"],
                                    "coverage_percent": m_d["coverage_percent"],
                                    "priority_service_percent": m_d["priority_service_percent"],
                                    "hidden_victim_discovery_percent": m_d["hidden_victim_discovery_percent"],
                                    "avg_turns_per_robot": m_d["avg_turns_per_robot"],
                                    "redundancy_rate": m_d["redundancy_rate"],
                                    "makespan_cells": m_d["makespan_cells"],
                                    "mean_energy_used": m_d["mean_energy_used"],
                                    "mean_remaining_energy": m_d["mean_remaining_energy"],
                                    "panic_event": panic_event,
                                    "panic_type": panic_type
                                })
                        
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
        
        # Save raw all seeds convergence CSV
        conv_fields = [
            "method", "rows", "cols", "map_size", "n_robots", "seed", "generation",
            "best_cost", "mean_cost", "priority_coverage_ratio", "coverage_percent",
            "priority_service_percent", "hidden_victim_discovery_percent", "avg_turns_per_robot",
            "redundancy_rate", "makespan_cells", "mean_energy_used", "mean_remaining_energy",
            "panic_event", "panic_type"
        ]
        conv_all_path = os.path.join(results_dir, "convergence_behaviour_all_seeds.csv")
        with open(conv_all_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=conv_fields)
            writer.writeheader()
            writer.writerows(all_convergence_rows)
        print(f"Saved raw convergence behavior data to: {conv_all_path}")
        
        # Save panic events CSV
        panic_fields = [
            "method", "rows", "cols", "map_size", "n_robots", "seed", "generation", "panic_type",
            "cost_before", "cost_after", "cost_reduction_percent", "rho_before", "rho_after", "rho_improvement"
        ]
        panic_events_path = os.path.join(results_dir, "convergence_panic_events.csv")
        with open(panic_events_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=panic_fields)
            writer.writeheader()
            writer.writerows(all_panic_events)
        print(f"Saved panic events to: {panic_events_path}")
        
        # Select best seeds based on TRACE convergence data
        trace_finals = {}
        for r in all_convergence_rows:
            if r["method"] == "TRACE" and r["generation"] == 279:
                key = (r["map_size"], r["n_robots"])
                trace_finals.setdefault(key, []).append(r)
                
        best_seed_rows = []
        for key, rows_list in trace_finals.items():
            sorted_rows = sorted(
                rows_list,
                key=lambda x: (
                    -x["priority_service_percent"],
                    x["best_cost"],
                    x["makespan_cells"],
                    x["redundancy_rate"],
                    x["seed"]
                )
            )
            best_s = sorted_rows[0]["seed"]
            r_val = sorted_rows[0]["rows"]
            c_val = sorted_rows[0]["cols"]
            best_seed_rows.append({
                "rows": r_val,
                "cols": c_val,
                "map_size": key[0],
                "n_robots": key[1],
                "best_seed": best_s
            })
            
        best_seed_fields = ["rows", "cols", "map_size", "n_robots", "best_seed"]
        best_seed_csv_path = os.path.join(results_dir, "convergence_behaviour_best_seed.csv")
        with open(best_seed_csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=best_seed_fields)
            writer.writeheader()
            writer.writerows(best_seed_rows)
        print(f"Saved best seeds selection to: {best_seed_csv_path}")
        
        # Save convergence summary CSV
        summary_groups = {}
        for r in all_convergence_rows:
            g_key = (r["method"], r["rows"], r["cols"], r["map_size"], r["n_robots"], r["generation"])
            summary_groups.setdefault(g_key, []).append(r)
            
        summary_rows = []
        metrics_to_summarize = [
            "best_cost", "mean_cost", "priority_coverage_ratio", "coverage_percent",
            "priority_service_percent", "hidden_victim_discovery_percent", "avg_turns_per_robot",
            "redundancy_rate", "makespan_cells", "mean_energy_used", "mean_remaining_energy"
        ]
        
        for g_key, rows_list in summary_groups.items():
            sum_row = {
                "method": g_key[0],
                "rows": g_key[1],
                "cols": g_key[2],
                "map_size": g_key[3],
                "n_robots": g_key[4],
                "generation": g_key[5]
            }
            for metric in metrics_to_summarize:
                vals = [float(r[metric]) for r in rows_list]
                sum_row[f"{metric}_mean"] = np.mean(vals)
                sum_row[f"{metric}_std"] = np.std(vals)
            summary_rows.append(sum_row)
            
        summary_fields = ["method", "rows", "cols", "map_size", "n_robots", "generation"]
        for metric in metrics_to_summarize:
            summary_fields.append(f"{metric}_mean")
            summary_fields.append(f"{metric}_std")
            
        summary_csv_path = os.path.join(results_dir, "convergence_behaviour_summary.csv")
        with open(summary_csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=summary_fields)
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"Saved convergence summary statistics to: {summary_csv_path}")
        
        # Calculate summary metrics (mean & std) for performance
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
