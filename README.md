# TRACE: Target-aware Routing And Coverage with Evolutionary Reinforcement Learning

This repository contains a consolidated, publication-grade Python implementation of the **TRACE** framework for multi-robot Search and Rescue (SAR) missions:

> **TRACE: Target-aware Routing And Coverage with Evolutionary Reinforcement Learning**

The pipeline integrates three core stages to deliver target-aware area partitioning, energy-constrained vehicle routing, and reinforcement learning-guided evolutionary trajectory optimization:

1. **Priority-Aware Area Decomposition (Stage 1):** Extension of the DARP algorithm using priority-weighted grids and Manhattan distance mappings to guarantee balanced spatial workload allocation among heterogeneous robots.
2. **Priority-Weighted CVRP Construction (Stage 2):** Formulation of Capacity Constrained Vehicle Routing Problems (CVRP) for each sub-region, incorporating robot battery constraints, linear transit costs, and extra hover consumption on victim-critical cells.
3. **Delta RL-Guided Memetic Algorithm (Stage 3):** Routing optimizer combining a Genetic Algorithm (GA) with local search operators. A Q-learning agent dynamically selects the best local search operators based on a 7-feature normalized state representation of the population.

---

## Interactive UI Simulator
The Pygame-based simulator window consists of three panels:
* **Left Panel:** Displays simulation parameters, active constraints, live mission statistics (coverage %, priority cells serviced %, hidden victims found), and individual robot load meters.
* **Center Panel:** Visualizes the 2D grid-world search area containing obstacles, partitioned robot sub-regions, high-priority cells, hidden victims, active trajectories, and live robot movements.
* **Right Panel:** Plots live metrics (coverage and priority servicing percentage curves) alongside Delta RL solver performance charts.

### UI Controls
* `Space`: Pause / Resume simulation.
* `R`: Replay the trajectory animation after execution finishes.

---

## Installation

Ensure you have Python 3.10+ installed. It is recommended to run inside a virtual environment.

```bash
# Create and activate virtual environment
python -m venv venv
# On Windows PowerShell:
.\venv\Scripts\Activate.ps1
# On Linux/macOS:
source venv/bin/activate

# Install package dependencies
pip install -r requirements.txt
```

*(Dependencies include: `pygame`, `numpy`, `matplotlib`, and `scipy`)*

---

## Basic Execution

Run the simulation interactively with default configurations:
```bash
python run_trace.py --robots 4
```

### Useful CLI Parameters:
* `--robots {2,4,8}`: Set the number of robots.
* `--generations N`: Number of solver generations (default: 300).
* `--population M`: Solver population size (default: 50).
* `--energy E`: Robot battery budget in energy units (default: 95).
* `--no-ui`: Run the simulation headless (outputs final execution metrics only).
* `--mission-time-limit T`: Turn limit constraint (optional).

Example:
```bash
python run_trace.py --robots 8 --generations 350 --population 60 --energy 120 --no-ui
```

---

## Section 8 Experimental Studies

This package implements CLI support for reproducing the ablation studies, sensitivity analyses, and statistical evaluations detailed in Section 8 of the paper. Use `--experiment true` in combination with one of the `--study` choices below.

### 1. Comparative Performance Study
Compare TRACE against baseline algorithms (`A*-DARP`, `Delta RL-MA`, `DARP+RL-GA`) over all map sizes, robot counts, and seeds.
* **Run:**
  ```bash
  python run_trace.py --experiment true --study performance
  ```
* **Plot results:**
  ```bash
  python plot_metric.py --plot convergence
  ```

### 2. Component Ablation Study
Evaluate the impact of disabling core elements of TRACE (e.g. panic scheduling, priority weighting, annealing, or state augmentation) across variants $A_0$ to $A_{10}$.
* **Run:**
  ```bash
  python run_trace.py --experiment true --study ablation
  ```
* **Plot results:**
  ```bash
  python plot_metric.py --plot ablation
  ```

### 3. Stage-1 Beta Weight Sensitivity Analysis
Analyze the impact of varying the Stage-1 area decomposition priority scaling factor ($\beta$) from $0.0$ to $1.0$ versus using the annealed schedules.
* **Run:**
  ```bash
  python run_trace.py --experiment true --study beta_sensitivity
  ```
* **Plot results:**
  ```bash
  python plot_metric.py --plot beta_sensitivity
  ```

### 4. Stage-3 Parameter Sensitivity Analysis
Analyze the sensitivity of the memetic search algorithm to the priority reward scale ($\omega_W$) and RL reward multiplier ($\mu$).
* **Run:**
  ```bash
  python run_trace.py --experiment true --study stage3_sensitivity
  ```
* **Plot results:**
  ```bash
  python plot_metric.py --plot stage3_sensitivity
  ```

### 5. Local Search Operator Set Ablation
Ablate individual local search operators (Swap, 2-opt, Relocate, Block Swap, Block Relocate) to examine their contributions to route untangling and escaping deep local minima.
* **Run:**
  ```bash
  python run_trace.py --experiment true --study operator_ablation
  ```
* **Plot results:**
  ```bash
  python plot_metric.py --plot operator_ablation
  ```

### 6. Statistical Significance & Effect Size Analysis
Compute statistical significance using Welch’s two-sample $t$-test and effect size (Cohen's $d$) for all ablation variants compared against the full TRACE configuration ($A_0$).
* **Run:**
  ```bash
  python run_trace.py --experiment true --study ablation_stats
  ```
* **Plot results:**
  ```bash
  python plot_metric.py --plot ablation_stats
  ```

All raw results (`results/*.csv`), summary tables (`results/*_summary.csv`), LaTeX outputs (`results/tables/*.tex`), and figures (`results/plots/*.pdf`) are written into the `results/` directory.

---

## Directory & Package Structure

* `run_trace.py`: Main entry script executing the Pygame UI simulation and running the 6 Section 8 evaluation studies.
* `plot_metric.py`: Dedicated plotting script containing matplotlib functions to generate performance curves.
* `trace/`: Core TRACE engine library:
  * [config.py](file:///c:/Users/DELL/Documents/Thesis%20Code%20WorkSpace/TRACE_implementation/trace/config.py): TRACE configuration parameters and default values.
  * [environment.py](file:///c:/Users/DELL/Documents/Thesis%20Code%20WorkSpace/TRACE_implementation/trace/environment.py): SARS environment generator, grid mapping, cell coordinates, and victim priors.
  * [decomposition.py](file:///c:/Users/DELL/Documents/Thesis%20Code%20WorkSpace/TRACE_implementation/trace/decomposition.py): Priority-Aware A\*-DARP area decomposition partitioning.
  * [cvrp.py](file:///c:/Users/DELL/Documents/Thesis%20Code%20WorkSpace/TRACE_implementation/trace/cvrp.py): CVRP instances creation and trajectory constraints checking.
  * [delta_rl_memetic.py](file:///c:/Users/DELL/Documents/Thesis%20Code%20WorkSpace/TRACE_implementation/trace/delta_rl_memetic.py): Evolutionary reinforcement learning solver and local search operators.
  * [simulator.py](file:///c:/Users/DELL/Documents/Thesis%20Code%20WorkSpace/TRACE_implementation/trace/simulator.py): End-to-end trace workflow simulation executor.
  * [metrics.py](file:///c:/Users/DELL/Documents/Thesis%20Code%20WorkSpace/TRACE_implementation/trace/metrics.py): Performance metrics calculations.
  * [ui.py](file:///c:/Users/DELL/Documents/Thesis%20Code%20WorkSpace/TRACE_implementation/trace/ui.py): Pygame graphics interface.
  * [astar.py](file:///c:/Users/DELL/Documents/Thesis%20Code%20WorkSpace/TRACE_implementation/trace/astar.py): A\* search pathfinder.
* `baselines/`: Benchmark baseline implementations (`astar_darp.py`, `delta_rl_ma.py`, `darp_rl_ga.py`).
* `results/`: Directory created during runs containing CSV outputs, LaTeX tables (`results/tables/`), and visual charts (`results/plots/`).
