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
def plot_decomposition_results(csv_path: str, best_seed_path: str, plots_dir: str):
    import os
    import csv
    import matplotlib.pyplot as plt
    
    os.makedirs(plots_dir, exist_ok=True)
    
    # Load best seeds mapping: (rows, cols, n_robots) -> best_seed
    best_seeds = {}
    with open(best_seed_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            r = int(row["rows"])
            c = int(row["cols"])
            n = int(row["n_robots"])
            s = int(row["best_seed"])
            best_seeds[(r, c, n)] = s
            
    # Load all raw data
    raw_data = []
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_data.append({
                "method": row["method"],
                "rows": int(row["rows"]),
                "cols": int(row["cols"]),
                "n_robots": int(row["n_robots"]),
                "seed": int(row["seed"]),
                "max_cell_count_discrepancy": float(row["max_cell_count_discrepancy"]),
                "max_priority_load_discrepancy": float(row["max_priority_load_discrepancy"])
            })
            
    robot_counts = [4, 8, 16, 32, 64]
    map_sizes = [(20, 20), (40, 40), (80, 80)]
    
    plt.rcParams.update({
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9
    })
    
    # 1. Priority-load discrepancy comparison
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
    for idx, (rows, cols) in enumerate(map_sizes):
        ax = axes[idx]
        baseline_vals = []
        trace_vals = []
        
        for n in robot_counts:
            best_s = best_seeds.get((rows, cols, n), 7)
            b_row = next((x for x in raw_data if x["method"] == "Baseline A*-DARP" and x["rows"] == rows and x["cols"] == cols and x["n_robots"] == n and x["seed"] == best_s), None)
            t_row = next((x for x in raw_data if x["method"] == "Priority-aware TRACE" and x["rows"] == rows and x["cols"] == cols and x["n_robots"] == n and x["seed"] == best_s), None)
            
            baseline_vals.append(b_row["max_priority_load_discrepancy"] if b_row else 0.0)
            trace_vals.append(t_row["max_priority_load_discrepancy"] if t_row else 0.0)
            
        ax.plot(robot_counts, baseline_vals, marker='o', linestyle='--', color='#1f77b4', linewidth=2, label="Baseline A*-DARP")
        ax.plot(robot_counts, trace_vals, marker='s', linestyle='-', color='#d62728', linewidth=2, label="Priority-aware TRACE")
        ax.set_title(f"{rows}x{cols} Map")
        ax.set_xlabel("Number of Robots")
        if idx == 0:
            ax.set_ylabel("Max Priority-Load Discrepancy ($\Delta_w$)")
        ax.set_xticks(robot_counts)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend()
        
    plt.suptitle("Decomposition Priority-Load Discrepancy (Best Seed Results)", y=0.98, fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "decomp_priority_load_discrepancy_best_seed_all_maps.pdf"), format="pdf")
    plt.savefig(os.path.join(plots_dir, "decomp_priority_load_discrepancy_best_seed_all_maps.png"), format="png")
    plt.close()
    
    # 2. Cell-count discrepancy comparison
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
    for idx, (rows, cols) in enumerate(map_sizes):
        ax = axes[idx]
        baseline_vals = []
        trace_vals = []
        
        for n in robot_counts:
            best_s = best_seeds.get((rows, cols, n), 7)
            b_row = next((x for x in raw_data if x["method"] == "Baseline A*-DARP" and x["rows"] == rows and x["cols"] == cols and x["n_robots"] == n and x["seed"] == best_s), None)
            t_row = next((x for x in raw_data if x["method"] == "Priority-aware TRACE" and x["rows"] == rows and x["cols"] == cols and x["n_robots"] == n and x["seed"] == best_s), None)
            
            baseline_vals.append(b_row["max_cell_count_discrepancy"] if b_row else 0.0)
            trace_vals.append(t_row["max_cell_count_discrepancy"] if t_row else 0.0)
            
        ax.plot(robot_counts, baseline_vals, marker='o', linestyle='--', color='#1f77b4', linewidth=2, label="Baseline A*-DARP")
        ax.plot(robot_counts, trace_vals, marker='s', linestyle='-', color='#d62728', linewidth=2, label="Priority-aware TRACE")
        ax.set_title(f"{rows}x{cols} Map")
        ax.set_xlabel("Number of Robots")
        if idx == 0:
            ax.set_ylabel("Max Cell-Count Discrepancy ($\Delta_k$)")
        ax.set_xticks(robot_counts)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend()
        
    plt.suptitle("Decomposition Cell-Count Discrepancy (Best Seed Results)", y=0.98, fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "decomp_cell_count_discrepancy_best_seed_all_maps.pdf"), format="pdf")
    plt.savefig(os.path.join(plots_dir, "decomp_cell_count_discrepancy_best_seed_all_maps.png"), format="png")
    plt.close()


def plot_aggregate_results(csv_path: str, plots_dir: str):
    import os
    import csv
    import numpy as np
    import matplotlib.pyplot as plt
    
    os.makedirs(plots_dir, exist_ok=True)
    
    data = []
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append({
                "method": row["method"],
                "rows": int(row["rows"]),
                "cols": int(row["cols"]),
                "map_size": row["map_size"],
                "n_robots": int(row["n_robots"]),
                "seed": int(row["seed"]),
                "coverage_percent": float(row["coverage_percent"]),
                "priority_service_percent": float(row["priority_service_percent"]),
                "hidden_victim_discovery_percent": float(row["hidden_victim_discovery_percent"]),
                "avg_turns_per_robot": float(row["avg_turns_per_robot"]),
                "redundancy_rate": float(row["redundancy_rate"]),
                "makespan_cells": float(row["makespan_cells"]),
                "runtime_seconds": float(row["runtime_seconds"]),
                "mean_energy_used": float(row["mean_energy_used"]),
                "max_energy_used": float(row["max_energy_used"]),
                "mean_remaining_energy": float(row["mean_remaining_energy"])
            })
            
    methods = ["A*-DARP", "Delta RL-MA", "DARP+RL-GA", "TRACE"]
    robot_counts = [4, 8, 16, 32, 64]
    map_sizes = [(20, 20), (40, 40), (80, 80)]
    
    metrics = {
        "coverage_percent": ("Coverage (%)", "aggregate_constrained_coverage_all_maps.pdf"),
        "priority_service_percent": ("Priority Service (%)", "aggregate_constrained_priority_service_all_maps.pdf"),
        "hidden_victim_discovery_percent": ("Hidden Victim Discovery (%)", "aggregate_constrained_hidden_victim_discovery_all_maps.pdf"),
        "avg_turns_per_robot": ("Avg. Turns per Robot", "aggregate_constrained_avg_turns_all_maps.pdf"),
        "redundancy_rate": ("Redundancy Rate", "aggregate_constrained_redundancy_all_maps.pdf"),
        "makespan_cells": ("Makespan (cells)", "aggregate_constrained_makespan_all_maps.pdf"),
        "mean_energy_used": ("Mean Energy Used (%)", "aggregate_constrained_energy_used_all_maps.pdf"),
        "mean_remaining_energy": ("Mean Remaining Energy (%)", "aggregate_constrained_remaining_energy_all_maps.pdf")
    }
    
    colors = {
        "A*-DARP": "#1f77b4",
        "Delta RL-MA": "#ff7f0e",
        "DARP+RL-GA": "#2ca02c",
        "TRACE": "#d62728"
    }
    markers = {
        "A*-DARP": "o",
        "Delta RL-MA": "s",
        "DARP+RL-GA": "^",
        "TRACE": "D"
    }
    linestyles = {
        "A*-DARP": "--",
        "Delta RL-MA": "-.",
        "DARP+RL-GA": ":",
        "TRACE": "-"
    }
    
    plt.rcParams.update({
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9
    })
    
    for metric_col, (metric_name, filename) in metrics.items():
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
        
        for idx, (rows, cols) in enumerate(map_sizes):
            ax = axes[idx]
            map_str = f"{rows}x{cols}"
            
            for method in methods:
                mean_vals = []
                std_vals = []
                
                for n_robots in robot_counts:
                    vals = [
                        r[metric_col] for r in data 
                        if r["method"] == method and r["rows"] == rows and r["cols"] == cols and r["n_robots"] == n_robots
                    ]
                    if vals:
                        mean_vals.append(np.mean(vals))
                        std_vals.append(np.std(vals))
                    else:
                        mean_vals.append(0.0)
                        std_vals.append(0.0)
                        
                mean_arr = np.array(mean_vals)
                std_arr = np.array(std_vals)
                
                ax.plot(
                    robot_counts, mean_arr, 
                    linestyle=linestyles[method], 
                    marker=markers[method], 
                    color=colors[method], 
                    linewidth=2, 
                    label=method
                )
                ax.fill_between(
                    robot_counts, 
                    mean_arr - std_arr, 
                    mean_arr + std_arr, 
                    color=colors[method], 
                    alpha=0.12
                )
                
            ax.set_title(f"{map_str} Map")
            ax.set_xlabel("Number of Robots")
            if idx == 0:
                ax.set_ylabel(metric_name)
            ax.set_xticks(robot_counts)
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.legend(loc="best")
            
        plt.suptitle(f"Aggregate Performance: {metric_name} (Mean $\pm$ Std across Seeds)", y=0.98, fontsize=13)
        plt.tight_layout()
        pdf_path = os.path.join(plots_dir, filename)
        plt.savefig(pdf_path, format="pdf")
        
        png_path = pdf_path.replace(".pdf", ".png")
        plt.savefig(png_path, format="png")
        plt.close()


def export_aggregate_tables(summary_rows: List[Dict[str, Any]], output_dir: str):
    import os
    import csv
    
    os.makedirs(os.path.join(output_dir, "tables"), exist_ok=True)
    
    # We want to group by (rows, cols, n_robots)
    groups = {}
    for r in summary_rows:
        key = (r["rows"], r["cols"], r["n_robots"])
        groups.setdefault(key, []).append(r)
        
    metrics_mapping = [
        ("coverage_percent", "Coverage (%)", True),
        ("priority_service_percent", "Priority service (%)", True),
        ("hidden_victim_discovery_percent", "Hidden-victim disc. (%)", True),
        ("avg_turns_per_robot", "Avg. turns / robot", False),
        ("redundancy_rate", "Redundancy rate", False),
        ("makespan_cells", "Makespan (cells)", False),
        ("mean_energy_used", "Mean energy used (%)", False),
        ("mean_remaining_energy", "Mean remaining energy (%)", True)
    ]
    
    methods_order = ["A*-DARP", "Delta RL-MA", "DARP+RL-GA", "TRACE"]
    
    for (rows, cols, n_robots), rows_list in groups.items():
        # Build dictionary from method name to row data
        method_data = {r["method"]: r for r in rows_list}
        
        csv_filename = f"aggregate_performance_{rows}x{cols}_robots_{n_robots}.csv"
        csv_path = os.path.join(output_dir, "tables", csv_filename)
        
        tex_filename = f"aggregate_performance_{rows}x{cols}_robots_{n_robots}.tex"
        tex_path = os.path.join(output_dir, "tables", tex_filename)
        
        # 1. Format CSV
        csv_headers = ["Metric", "A*-DARP", "Delta RL-MA", "DARP+RL-GA", "TRACE"]
        csv_rows = []
        
        for metric_col, metric_label, maximize in metrics_mapping:
            best_val = -float('inf') if maximize else float('inf')
            best_method = None
            
            for m in methods_order:
                r = method_data.get(m)
                if r is not None:
                    m_val = r[f"{metric_col}_mean"]
                    if maximize:
                        if m_val > best_val:
                            best_val = m_val
                            best_method = m
                    else:
                        if m_val < best_val:
                            best_val = m_val
                            best_method = m
                            
            csv_row = {"Metric": metric_label}
            for m in methods_order:
                r = method_data.get(m)
                if r is not None:
                    mean_val = r[f"{metric_col}_mean"]
                    std_val = r[f"{metric_col}_std"]
                    cell_str = f"{mean_val:.2f} ± {std_val:.2f}"
                    if m == best_method:
                        csv_row[m] = f"**{cell_str}**"
                    else:
                        csv_row[m] = cell_str
                else:
                    csv_row[m] = "-"
            csv_rows.append(csv_row)
            
        with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=csv_headers)
            writer.writeheader()
            writer.writerows(csv_rows)
            
        # 2. Format TeX
        with open(tex_path, mode="w", encoding="utf-8") as f:
            f.write("\\begin{table}[htbp]\n")
            f.write(f"\\caption{{Aggregate Performance Comparison for {rows}x{cols} Map with {n_robots} Robots}}\n")
            f.write(f"\\label{{tab:perf_{rows}x{cols}_robots_{n_robots}}}\n")
            f.write("\\centering\n")
            f.write("\\begin{tabular}{lcccc}\n")
            f.write("\\toprule\n")
            f.write("Metric & A*-DARP & Delta RL-MA & DARP+RL-GA & TRACE \\\\\n")
            f.write("\\midrule\n")
            
            for metric_col, metric_label, maximize in metrics_mapping:
                best_val = -float('inf') if maximize else float('inf')
                best_method = None
                
                for m in methods_order:
                    r = method_data.get(m)
                    if r is not None:
                        m_val = r[f"{metric_col}_mean"]
                        if maximize:
                            if m_val > best_val:
                                best_val = m_val
                                best_method = m
                        else:
                            if m_val < best_val:
                                best_val = m_val
                                best_method = m
                                
                cells = []
                for m in methods_order:
                    r = method_data.get(m)
                    if r is not None:
                        mean_val = r[f"{metric_col}_mean"]
                        std_val = r[f"{metric_col}_std"]
                        cell_str = f"{mean_val:.2f} \\pm {std_val:.2f}"
                        if m == best_method:
                            cells.append(f"\\textbf{{{cell_str}}}")
                        else:
                            cells.append(cell_str)
                    else:
                        cells.append("-")
                        
                escaped_label = metric_label.replace("%", "\\%")
                f.write(f"{escaped_label} & {cells[0]} & {cells[1]} & {cells[2]} & {cells[3]} \\\\\n")
                
            f.write("\\bottomrule\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n")


if __name__ == "__main__":
    main()