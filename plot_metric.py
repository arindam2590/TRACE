from __future__ import annotations

import argparse
import os
import csv
from typing import List, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

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
    p.add_argument("--plot", type=str, default=None, choices=["convergence", "ablation", "beta_sensitivity", "stage3_sensitivity", "operator_ablation", "ablation_stats"], help="Type of plot to generate")

    return p.parse_args()


def plot_ablation_results(metrics_dir: str) -> None:
    csv_path = os.path.join(metrics_dir, "ablation_study_summary.csv")
    if not os.path.exists(csv_path):
        print(f"Summary file not found: {csv_path}")
        return
        
    rows = []
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
            
    data = {}
    for r in rows:
        if r["method"] == "TRACE":
            key = (r["map_size"], int(r["n_robots"]), r["variant"])
            data[key] = r
            
    map_sizes = ["20x20", "40x40", "80x80"]
    robot_counts = [4, 8, 16, 32, 64]
    variants = [
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
    variant_labels = ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10"]
    
    plots_dir = os.path.join(metrics_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    
    def get_val(map_size, n_robots, var, field):
        key = (map_size, n_robots, var)
        if key in data:
            return float(data[key][field])
        return 0.0
        
    # 1. Priority Service Drop
    fig, axes = plt.subplots(1, 5, figsize=(22, 4), sharey=True)
    fig.suptitle("Priority Service Drop by Ablation Variant (Relative to A0 Full TRACE)", fontsize=14, y=0.98)
    
    for idx, n_robots in enumerate(robot_counts):
        ax = axes[idx]
        x = np.arange(len(variants))
        width = 0.25
        
        for m_idx, map_size in enumerate(map_sizes):
            y_vals = [get_val(map_size, n_robots, var, "delta_priority_service") for var in variants]
            ax.bar(x + (m_idx - 1) * width, y_vals, width, label=map_size, color=colors[m_idx])
            
        ax.set_title(f"Robots: {n_robots}")
        ax.set_xticks(x)
        ax.set_xticklabels(variant_labels)
        ax.grid(True, linestyle=":", alpha=0.6)
        if idx == 0:
            ax.set_ylabel("Delta Priority Service (%)")
            ax.legend()
            
    plt.tight_layout()
    fig_path = os.path.join(plots_dir, "ablation_priority_service_drop_all_maps.pdf")
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig_path}")
    
    # 2. Coverage Drop
    fig, axes = plt.subplots(1, 5, figsize=(22, 4), sharey=True)
    fig.suptitle("Coverage Drop by Ablation Variant (Relative to A0 Full TRACE)", fontsize=14, y=0.98)
    
    for idx, n_robots in enumerate(robot_counts):
        ax = axes[idx]
        x = np.arange(len(variants))
        width = 0.25
        
        for m_idx, map_size in enumerate(map_sizes):
            y_vals = [get_val(map_size, n_robots, var, "delta_coverage") for var in variants]
            ax.bar(x + (m_idx - 1) * width, y_vals, width, label=map_size, color=colors[m_idx])
            
        ax.set_title(f"Robots: {n_robots}")
        ax.set_xticks(x)
        ax.set_xticklabels(variant_labels)
        ax.grid(True, linestyle=":", alpha=0.6)
        if idx == 0:
            ax.set_ylabel("Delta Coverage (%)")
            ax.legend()
            
    plt.tight_layout()
    fig_path = os.path.join(plots_dir, "ablation_coverage_drop_all_maps.pdf")
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig_path}")
    
    # 3. Turns/Robot Increase
    fig, axes = plt.subplots(1, 5, figsize=(22, 4), sharey=True)
    fig.suptitle("Turns/Robot Increase by Ablation Variant (Relative to A0 Full TRACE)", fontsize=14, y=0.98)
    
    for idx, n_robots in enumerate(robot_counts):
        ax = axes[idx]
        x = np.arange(len(variants))
        width = 0.25
        
        for m_idx, map_size in enumerate(map_sizes):
            a0_turns = get_val(map_size, n_robots, "A0: Full", "avg_turns_per_robot_mean")
            y_vals = []
            for var in variants:
                var_turns = get_val(map_size, n_robots, var, "avg_turns_per_robot_mean")
                y_vals.append(var_turns - a0_turns)
            ax.bar(x + (m_idx - 1) * width, y_vals, width, label=map_size, color=colors[m_idx])
            
        ax.set_title(f"Robots: {n_robots}")
        ax.set_xticks(x)
        ax.set_xticklabels(variant_labels)
        ax.grid(True, linestyle=":", alpha=0.6)
        if idx == 0:
            ax.set_ylabel("Delta Turns/Robot")
            ax.legend()
            
    plt.tight_layout()
    fig_path = os.path.join(plots_dir, "ablation_turns_increase_all_maps.pdf")
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig_path}")


def plot_beta_sensitivity_results(metrics_dir: str) -> None:
    csv_path = os.path.join(metrics_dir, "beta_sensitivity_summary.csv")
    if not os.path.exists(csv_path):
        print(f"Summary file not found: {csv_path}")
        return
        
    rows = []
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
            
    data = {}
    for r in rows:
        key = (r["map_size"], int(r["n_robots"]), r["beta_mode"], r["beta"])
        data[key] = r
        
    map_sizes = ["20x20", "40x40", "80x80"]
    robot_counts = [4, 8, 16, 32, 64]
    beta_values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    
    plots_dir = os.path.join(metrics_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    
    metrics = {
        "priority_service": ("priority_service_percent", "Priority Service (%)", "Priority Service vs. Stage-1 Priority Weight ($\\beta$)"),
        "coverage": ("coverage_percent", "Coverage (%)", "Coverage vs. Stage-1 Priority Weight ($\\beta$)"),
        "turns": ("avg_turns_per_robot", "Turns/robot", "Turns/robot vs. Stage-1 Priority Weight ($\\beta$)"),
        "cell_delta": ("cell_delta_max", "Cell $\\Delta_{\\max}$", "Cell $\\Delta_{\\max}$ vs. Stage-1 Priority Weight ($\\beta$)")
    }
    
    for metric_key, (field, ylabel, title) in metrics.items():
        fig, axes = plt.subplots(1, 5, figsize=(22, 4), sharey=False)
        fig.suptitle(title, fontsize=14, y=0.98)
        
        for idx, n_robots in enumerate(robot_counts):
            ax = axes[idx]
            
            for m_idx, map_size in enumerate(map_sizes):
                y_means = []
                y_stds = []
                valid_betas = []
                for b in beta_values:
                    key = (map_size, n_robots, "fixed", str(b))
                    if key in data:
                        y_means.append(float(data[key][f"{field}_mean"]))
                        y_stds.append(float(data[key][f"{field}_std"]))
                        valid_betas.append(b)
                
                if valid_betas:
                    ax.errorbar(valid_betas, y_means, yerr=y_stds, label=map_size, color=colors[m_idx], marker='o', capsize=3, elinewidth=1)
                
                ann_key = (map_size, n_robots, "annealed", "annealed")
                if ann_key in data:
                    ann_mean = float(data[ann_key][f"{field}_mean"])
                    ax.axhline(ann_mean, color=colors[m_idx], linestyle="--", alpha=0.7)
            
            ax.set_title(f"Robots: {n_robots}")
            ax.set_xlabel("$\\beta$")
            ax.grid(True, linestyle=":", alpha=0.6)
            if idx == 0:
                ax.set_ylabel(ylabel)
                handles, labels = ax.get_legend_handles_labels()
                if labels:
                    ax.legend()
                
        plt.tight_layout()
        fig_path = os.path.join(plots_dir, f"beta_sensitivity_{metric_key}_all_maps.pdf")
        plt.savefig(fig_path, bbox_inches="tight")
        plt.close()
        print(f"Saved: {fig_path}")


def plot_stage3_sensitivity_results(metrics_dir: str) -> None:
    csv_path = os.path.join(metrics_dir, "stage3_sensitivity_summary.csv")
    if not os.path.exists(csv_path):
        print(f"Summary file not found: {csv_path}")
        return
        
    rows = []
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
            
    data = {}
    for r in rows:
        try:
            val_str = str(float(r["parameter_value"]))
        except ValueError:
            val_str = r["parameter_value"]
        key = (r["map_size"], int(r["n_robots"]), r["sensitivity_type"], val_str)
        data[key] = r
        
    map_sizes = ["20x20", "40x40", "80x80"]
    robot_counts = [4, 8, 16, 32, 64]
    
    omega_w_values = [1.0, 2.0, 3.0, 5.0, 8.0, 12.0]
    mu_values = [0, 10, 25, 50, 100, 200]
    
    plots_dir = os.path.join(metrics_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    
    plot_configs = [
        ("omega", "omega_w", "priority_service_percent", "Priority Service (%)", "Priority Service vs. $\\omega_W$ ($\\mu=50$)", omega_w_values),
        ("omega", "omega_w", "coverage_percent", "Coverage (%)", "Coverage vs. $\\omega_W$ ($\\mu=50$)", omega_w_values),
        ("mu", "mu", "priority_service_percent", "Priority Service (%)", "Priority Service vs. $\\mu$ ($\\omega_W=3.0$)", mu_values),
        ("mu", "mu", "coverage_percent", "Coverage (%)", "Coverage vs. $\\mu$ ($\\omega_W=3.0$)", mu_values),
    ]
    
    for prefix, sens_type, field, ylabel, title, param_vals in plot_configs:
        fig, axes = plt.subplots(1, 5, figsize=(22, 4), sharey=False)
        fig.suptitle(title, fontsize=14, y=0.98)
        
        for idx, n_robots in enumerate(robot_counts):
            ax = axes[idx]
            
            for m_idx, map_size in enumerate(map_sizes):
                y_means = []
                y_stds = []
                valid_params = []
                for val in param_vals:
                    key = (map_size, n_robots, sens_type, str(float(val)))
                    if key in data:
                        y_means.append(float(data[key][f"{field}_mean"]))
                        y_stds.append(float(data[key][f"{field}_std"]))
                        valid_params.append(val)
                
                if valid_params:
                    ax.errorbar(valid_params, y_means, yerr=y_stds, label=map_size, color=colors[m_idx], marker='o', capsize=3, elinewidth=1)
                    
            ax.set_title(f"Robots: {n_robots}")
            ax.set_xlabel("$\\omega_W$" if sens_type == "omega_w" else "$\\mu$")
            ax.grid(True, linestyle=":", alpha=0.6)
            if idx == 0:
                ax.set_ylabel(ylabel)
                handles, labels = ax.get_legend_handles_labels()
                if labels:
                    ax.legend()
                
        plt.tight_layout()
        fig_path = os.path.join(plots_dir, f"stage3_sensitivity_{prefix}_{field.replace('_percent', '')}_all_maps.pdf")
        plt.savefig(fig_path, bbox_inches="tight")
        plt.close()
        print(f"Saved: {fig_path}")


def plot_operator_ablation_results(metrics_dir: str) -> None:
    csv_path = os.path.join(metrics_dir, "operator_ablation_summary.csv")
    if not os.path.exists(csv_path):
        print(f"Summary file not found: {csv_path}")
        return
        
    rows = []
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
            
    data = {}
    for r in rows:
        key = (r["map_size"], int(r["n_robots"]), r["variant"])
        data[key] = r
        
    map_sizes = ["20x20", "40x40", "80x80"]
    robot_counts = [4, 8, 16, 32, 64]
    
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
    
    plots_dir = os.path.join(metrics_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    
    plot_configs = [
        ("priority_service", "priority_service_percent", "Priority Service (%)", "Priority Service vs. Operator Removed"),
        ("turns", "avg_turns_per_robot", "Turns/robot", "Turns/robot vs. Operator Removed"),
        ("makespan", "makespan_cells", "Makespan (cells)", "Makespan vs. Operator Removed"),
        ("redundancy", "redundancy_rate", "Redundancy Rate", "Redundancy Rate vs. Operator Removed")
    ]
    
    for prefix, field, ylabel, title in plot_configs:
        fig, axes = plt.subplots(1, 5, figsize=(24, 5), sharey=False)
        fig.suptitle(title, fontsize=14, y=0.98)
        
        for idx, n_robots in enumerate(robot_counts):
            ax = axes[idx]
            
            for m_idx, map_size in enumerate(map_sizes):
                y_means = []
                y_stds = []
                x_labels = []
                for var in variant_order:
                    key = (map_size, n_robots, var)
                    if key in data:
                        y_means.append(float(data[key][f"{field}_mean"]))
                        y_stds.append(float(data[key][f"{field}_std"]))
                        x_labels.append(variant_display[var])
                        
                if y_means:
                    ax.errorbar(range(len(y_means)), y_means, yerr=y_stds, label=map_size, color=colors[m_idx], marker='o', capsize=3, elinewidth=1)
            
            ax.set_title(f"Robots: {n_robots}")
            ax.set_xticks(range(len(variant_order)))
            ax.set_xticklabels([variant_display[v] for v in variant_order], rotation=45, ha="right")
            ax.grid(True, linestyle=":", alpha=0.6)
            if idx == 0:
                ax.set_ylabel(ylabel)
                handles, labels = ax.get_legend_handles_labels()
                if labels:
                    ax.legend()
                    
        plt.tight_layout()
        fig_path = os.path.join(plots_dir, f"operator_ablation_{prefix}_all_maps.pdf")
        plt.savefig(fig_path, bbox_inches="tight")
        plt.close()
        print(f"Saved: {fig_path}")


def plot_ablation_stats_results(metrics_dir: str) -> None:
    csv_path = os.path.join(metrics_dir, "ablation_statistical_significance.csv")
    if not os.path.exists(csv_path):
        print(f"Statistical results file not found: {csv_path}")
        return
        
    rows = []
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
            
    data = {}
    for r in rows:
        key = (r["map_size"], int(r["n_robots"]), r["variant"])
        data[key] = r
        
    map_sizes = ["20x20", "40x40", "80x80"]
    robot_counts = [4, 8, 16, 32, 64]
    
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
    
    x_labels = [v.split(": ")[0] for v in variant_order]
    
    plots_dir = os.path.join(metrics_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    
    def get_neg_log_p(p_str):
        p_val = float(p_str)
        p_val = max(p_val, 1e-6)
        return -np.log10(p_val)
        
    plot_configs = [
        ("cohens_d", lambda r: float(r["cohens_d"]), "Cohen's $d$", "Cohen's $d$ vs. Ablation Variant"),
        ("pvalue", lambda r: get_neg_log_p(r["p_value"]), "$-\\log_{10}(p)$", "$-\\log_{10}(p)$ vs. Ablation Variant"),
        ("delta_priority", lambda r: float(r["delta_pp"]), "Priority Service Degradation $\\Delta$ (pp)", "Priority Service Degradation vs. Ablation Variant")
    ]
    
    for filename_key, val_extractor, ylabel, title in plot_configs:
        fig, axes = plt.subplots(1, 5, figsize=(24, 5), sharey=False)
        fig.suptitle(title, fontsize=14, y=0.98)
        
        for idx, n_robots in enumerate(robot_counts):
            ax = axes[idx]
            
            for m_idx, map_size in enumerate(map_sizes):
                y_vals = []
                x_indices = []
                for v_idx, var in enumerate(variant_order):
                    key = (map_size, n_robots, var)
                    if key in data:
                        y_vals.append(val_extractor(data[key]))
                        x_indices.append(v_idx)
                        
                if y_vals:
                    ax.plot(x_indices, y_vals, label=map_size, color=colors[m_idx], marker='o')
                    
            ax.set_title(f"Robots: {n_robots}")
            ax.set_xticks(range(len(variant_order)))
            ax.set_xticklabels(x_labels, rotation=0)
            ax.grid(True, linestyle=":", alpha=0.6)
            
            if filename_key == "pvalue":
                ax.axhline(-np.log10(0.05), color="red", linestyle="--", alpha=0.5, label="p=0.05")
            elif filename_key == "cohens_d":
                ax.axhline(0.8, color="purple", linestyle="--", alpha=0.5, label="d=0.8")
                
            if idx == 0:
                ax.set_ylabel(ylabel)
                handles, labels = ax.get_legend_handles_labels()
                if labels:
                    ax.legend()
                    
        plt.tight_layout()
        fig_path = os.path.join(plots_dir, f"ablation_stats_{filename_key}_all_maps.pdf")
        plt.savefig(fig_path, bbox_inches="tight")
        plt.close()
        print(f"Saved: {fig_path}")


def main() -> None:
    args = parse_args()

    if args.plot == "convergence":
        plot_convergence_results(args.metrics_dir)
        return
    elif args.plot == "ablation":
        plot_ablation_results(args.metrics_dir)
        return
    elif args.plot == "beta_sensitivity":
        plot_beta_sensitivity_results(args.metrics_dir)
        return
    elif args.plot == "stage3_sensitivity":
        plot_stage3_sensitivity_results(args.metrics_dir)
        return
    elif args.plot == "operator_ablation":
        plot_operator_ablation_results(args.metrics_dir)
        return
    elif args.plot == "ablation_stats":
        plot_ablation_stats_results(args.metrics_dir)
        return

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


def plot_convergence_results(results_dir: str):
    print("Generating convergence behaviour plots...")
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # 1. Load best seeds selection mapping
    best_seed_path = os.path.join(results_dir, "convergence_behaviour_best_seed.csv")
    if not os.path.exists(best_seed_path):
        raise FileNotFoundError(f"Could not find best seeds file: {best_seed_path}")
    
    best_seeds = {}
    with open(best_seed_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            best_seeds[(row["map_size"], int(row["n_robots"]))] = int(row["best_seed"])

    # 2. Load all seeds convergence behavior
    conv_path = os.path.join(results_dir, "convergence_behaviour_all_seeds.csv")
    if not os.path.exists(conv_path):
        raise FileNotFoundError(f"Could not find convergence data: {conv_path}")

    all_data = []
    with open(conv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            all_data.append({
                "method": row["method"],
                "rows": int(row["rows"]),
                "cols": int(row["cols"]),
                "map_size": row["map_size"],
                "n_robots": int(row["n_robots"]),
                "seed": int(row["seed"]),
                "generation": int(row["generation"]),
                "best_cost": float(row["best_cost"]),
                "mean_cost": float(row["mean_cost"]),
                "priority_coverage_ratio": float(row["priority_coverage_ratio"]),
                "coverage_percent": float(row["coverage_percent"]),
                "priority_service_percent": float(row["priority_service_percent"]),
                "hidden_victim_discovery_percent": float(row["hidden_victim_discovery_percent"]),
                "avg_turns_per_robot": float(row["avg_turns_per_robot"]),
                "redundancy_rate": float(row["redundancy_rate"]),
                "makespan_cells": float(row["makespan_cells"]),
                "mean_energy_used": float(row["mean_energy_used"]),
                "mean_remaining_energy": float(row["mean_remaining_energy"]),
                "panic_event": int(row["panic_event"]),
                "panic_type": row["panic_type"]
            })

    # 3. Load panic events
    panic_path = os.path.join(results_dir, "convergence_panic_events.csv")
    panic_events = []
    if os.path.exists(panic_path):
        with open(panic_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                panic_events.append({
                    "method": row["method"],
                    "rows": int(row["rows"]),
                    "cols": int(row["cols"]),
                    "map_size": row["map_size"],
                    "n_robots": int(row["n_robots"]),
                    "seed": int(row["seed"]),
                    "generation": int(row["generation"]),
                    "panic_type": row["panic_type"]
                })

    robot_counts = [4, 8, 16, 32, 64]
    map_sizes = ["20x20", "40x40", "80x80"]
    colors = {
        "A*-DARP": "#1f77b4",
        "Delta RL-MA": "#ff7f0e",
        "DARP+RL-GA": "#2ca02c",
        "TRACE": "#d62728"
    }

    plt.rcParams.update({
        'font.size': 9,
        'axes.labelsize': 10,
        'axes.titlesize': 11,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8
    })

    # 4. Generate single plots for paper-facing best-seeds
    for map_size in map_sizes:
        for n_robots in robot_counts:
            seed = best_seeds.get((map_size, n_robots))
            if seed is None:
                continue

            fig, ax_left = plt.subplots(figsize=(7, 4.5))
            ax_right = ax_left.twinx()

            for method in ["A*-DARP", "Delta RL-MA", "DARP+RL-GA", "TRACE"]:
                rows = [r for r in all_data if r["method"] == method and r["map_size"] == map_size and r["n_robots"] == n_robots and r["seed"] == seed]
                if method == "A*-DARP":
                    if rows:
                        # Draw a flat reference line for the non-iterative baseline
                        ax_left.plot([0, 279], [rows[0]["best_cost"], rows[0]["best_cost"]], linestyle="-", color=colors[method], linewidth=1.5, label="A*-DARP Cost")
                        ax_right.plot([0, 279], [rows[0]["priority_coverage_ratio"], rows[0]["priority_coverage_ratio"]], linestyle="--", color=colors[method], linewidth=1.5, label="A*-DARP $\\rho_W$")
                else:
                    rows = sorted(rows, key=lambda x: x["generation"])
                    if not rows:
                        continue
                    gens = [r["generation"] for r in rows]
                    costs = [r["best_cost"] for r in rows]
                    rhos = [r["priority_coverage_ratio"] for r in rows]

                    ax_left.plot(gens, costs, linestyle="-", color=colors[method], linewidth=2, label=f"{method} Cost")
                    ax_right.plot(gens, rhos, linestyle="--", color=colors[method], linewidth=2, label=f"{method} $\\rho_W$")

                    # Draw panic lines
                    run_panics = [p for p in panic_events if p["method"] == method and p["map_size"] == map_size and p["n_robots"] == n_robots and p["seed"] == seed]
                    for p in run_panics:
                        gen_p = p["generation"]
                        pt = p["panic_type"]
                        if pt == "cost_driven":
                            ax_left.axvline(x=gen_p, linestyle="--", color=colors[method], alpha=0.5, linewidth=1.2)
                        elif pt == "priority_driven":
                            ax_left.axvline(x=gen_p, linestyle=":", color=colors[method], alpha=0.5, linewidth=1.2)
                        elif pt == "both":
                            ax_left.axvline(x=gen_p, linestyle="-.", color=colors[method], alpha=0.5, linewidth=1.2)

            ax_left.set_xlabel("Generation")
            ax_left.set_ylabel("Best Cost (solid lines)")
            ax_right.set_ylabel("Priority Coverage Ratio (\\rho_W, dashed lines)")
            ax_right.set_ylim(-0.05, 1.05)
            ax_left.grid(True, linestyle="--", alpha=0.5)

            panic_legend_handles = [
                Line2D([0], [0], color="gray", linestyle="--", label="Cost-driven Panic"),
                Line2D([0], [0], color="gray", linestyle=":", label="Priority-driven Panic"),
                Line2D([0], [0], color="gray", linestyle="-.", label="Both Panic Triggers")
            ]

            handles_l, labels_l = ax_left.get_legend_handles_labels()
            handles_r, labels_r = ax_right.get_legend_handles_labels()
            all_handles = handles_l + handles_r + panic_legend_handles
            all_labels = labels_l + labels_r + [h.get_label() for h in panic_legend_handles]

            ax_left.legend(all_handles, all_labels, loc="lower left", fontsize=8)
            ax_left.set_title(f"Convergence Curves (Map {map_size}, {n_robots} Robots, Best Seed {seed})")

            plt.tight_layout()
            out_filename = f"convergence_best_seed__{map_size}__robots_{n_robots}__cost_rho_panic.pdf"
            plt.savefig(os.path.join(plots_dir, out_filename), format="pdf")
            plt.close()
            print(f"Saved: {out_filename}")

    # 5. Generate combined plots across robot counts for each map size
    for map_size in map_sizes:
        fig, axes = plt.subplots(1, 5, figsize=(22, 4.5))

        for idx, n_robots in enumerate(robot_counts):
            ax_left = axes[idx]
            ax_right = ax_left.twinx()
            seed = best_seeds.get((map_size, n_robots))
            if seed is None:
                continue

            for method in ["A*-DARP", "Delta RL-MA", "DARP+RL-GA", "TRACE"]:
                rows = [r for r in all_data if r["method"] == method and r["map_size"] == map_size and r["n_robots"] == n_robots and r["seed"] == seed]
                if method == "A*-DARP":
                    if rows:
                        ax_left.plot([0, 279], [rows[0]["best_cost"], rows[0]["best_cost"]], linestyle="-", color=colors[method], linewidth=1.5, label="A*-DARP Cost")
                        ax_right.plot([0, 279], [rows[0]["priority_coverage_ratio"], rows[0]["priority_coverage_ratio"]], linestyle="--", color=colors[method], linewidth=1.5, label="A*-DARP $\\rho_W$")
                else:
                    rows = sorted(rows, key=lambda x: x["generation"])
                    if not rows:
                        continue
                    gens = [r["generation"] for r in rows]
                    costs = [r["best_cost"] for r in rows]
                    rhos = [r["priority_coverage_ratio"] for r in rows]

                    ax_left.plot(gens, costs, linestyle="-", color=colors[method], linewidth=2, label=f"{method} Cost")
                    ax_right.plot(gens, rhos, linestyle="--", color=colors[method], linewidth=2, label=f"{method} $\\rho_W$")

                    # Draw panic lines
                    run_panics = [p for p in panic_events if p["method"] == method and p["map_size"] == map_size and p["n_robots"] == n_robots and p["seed"] == seed]
                    for p in run_panics:
                        gen_p = p["generation"]
                        pt = p["panic_type"]
                        if pt == "cost_driven":
                            ax_left.axvline(x=gen_p, linestyle="--", color=colors[method], alpha=0.5, linewidth=1.2)
                        elif pt == "priority_driven":
                            ax_left.axvline(x=gen_p, linestyle=":", color=colors[method], alpha=0.5, linewidth=1.2)
                        elif pt == "both":
                            ax_left.axvline(x=gen_p, linestyle="-.", color=colors[method], alpha=0.5, linewidth=1.2)

            ax_left.set_xlabel("Generation")
            if idx == 0:
                ax_left.set_ylabel("Best Cost (solid lines)")
            if idx == 4:
                ax_right.set_ylabel("Priority Coverage Ratio (\\rho_W, dashed lines)")
            ax_right.set_ylim(-0.05, 1.05)
            ax_left.grid(True, linestyle="--", alpha=0.5)
            ax_left.set_title(f"{n_robots} Robots (Seed {seed})")

            if idx == 0:
                panic_legend_handles = [
                    Line2D([0], [0], color="gray", linestyle="--", label="Cost-driven Panic"),
                    Line2D([0], [0], color="gray", linestyle=":", label="Priority-driven Panic"),
                    Line2D([0], [0], color="gray", linestyle="-.", label="Both Panic Triggers")
                ]
                handles_l, labels_l = ax_left.get_legend_handles_labels()
                handles_r, labels_r = ax_right.get_legend_handles_labels()
                all_handles = handles_l + handles_r + panic_legend_handles
                all_labels = labels_l + labels_r + [h.get_label() for h in panic_legend_handles]
                ax_left.legend(all_handles, all_labels, loc="lower left", fontsize=7)

        plt.suptitle(f"Convergence Comparison across Robot Counts ({map_size} Map)", y=0.98, fontsize=13)
        plt.tight_layout()
        out_combined = f"convergence_best_seed__{map_size}__all_robots__cost_rho_panic.pdf"
        plt.savefig(os.path.join(plots_dir, out_combined), format="pdf")
        plt.close()
        print(f"Saved combined map plot: {out_combined}")

    print("Convergence behaviour plotting completed successfully.")


if __name__ == "__main__":
    main()
