# TRACE: Target-Aware Routing And Coverage with Evolutionary Reinforcement Learning

TRACE is a hierarchical multi-robot search-and-rescue (SAR) simulation framework that integrates:

1. **Priority-aware A*-DARP decomposition** for balanced multi-robot area allocation
2. **Priority-weighted CVRP formulation** for energy-aware routing
3. **Delta Reinforcement Learning-guided Memetic Optimization** for adaptive route improvement
4. **Victim-aware search and rescue modeling** using a population-prior map
5. **Interactive simulation and visualization UI** with live metrics and graph plots

The framework consolidates ideas from:

- A*-DARP based multi-robot coverage path planning
- Cooperative QMIX-based victim prioritization
- Delta RL-guided memetic optimization for CVRP

The implementation is designed for reproducible experimentation in:

- Multi-agent search and rescue
- UAV swarm coordination
- Victim-aware area coverage
- Energy-aware path planning
- RL-guided combinatorial optimization
- Coverage redundancy reduction
- Coverage-performance evaluation

---

# Repository Structure

```text
TRACE_implementation/
│
├── run_trace.py
├── plot_metrics.py
├── requirements.txt
├── README.md
│
├── results/
│   └── Saved metric spreadsheets (.csv)
│
├── trace/
│   ├── __init__.py
│   ├── config.py
│   ├── environment.py
│   ├── decomposition.py
│   ├── pw_cvrp.py
│   ├── delta_rl_ma.py
│   ├── planner.py
│   ├── metrics.py
│   ├── ui.py
│   └── utils.py
│
└── assets/
    └── Optional images / screenshots
```

---

# Core Components

## 1. `environment.py`

Responsible for:

- Grid-world generation
- Obstacle placement
- Priority-cell generation
- Hidden victim generation
- Population-prior map generation
- Environment initialization

### Main Concepts

- Free cells
- Obstacle cells
- Priority victim regions
- Hidden victims
- Robot starting positions

---

## 2. `decomposition.py`

Implements the **Priority-aware A*-DARP decomposition** stage.

### Responsibilities

- A* shortest-path computation
- Obstacle-aware evaluation matrix
- Fair multi-robot partitioning
- Priority-load balancing
- Connectivity preservation
- Annealed beta scheduling

### Output

Returns balanced robot regions:

```python
robot_regions[i]
```

Each robot receives:

- Connected subregion
- Balanced cell count
- Balanced priority-victim load

---

## 3. `pw_cvrp.py`

Builds the **Priority-Weighted Capacitated Vehicle Routing Problem (PW-CVRP)**.

### Responsibilities

- Convert robot regions into routing problems
- Create customer nodes
- Apply energy constraints
- Create priority weights
- Compute routing distances

### Key Concepts

- Energy-aware routing
- Multi-trip routing
- Priority-weighted objective
- Turn-aware travel cost

---

## 4. `delta_rl_ma.py`

Implements the **Delta Reinforcement Learning-guided Memetic Algorithm**.

### Features

- Population-level Q-learning
- Operator selection
- Deep local search
- Panic-window diversification
- True elitism
- Multi-customer operators

### Supported Operators

| Operator | Description |
|---|---|
| Swap | Exchange customers between routes |
| 2-opt | Reverse route segments |
| Relocate | Move customer to another route |
| Block Swap | Swap multiple customers |
| Block Relocate | Relocate customer blocks |

### RL State Features

The Q-learning agent observes:

- Total cost
- Maximum route cost
- Minimum route cost
- Capacity violation
- Route variance
- Number of routes
- Priority coverage ratio

### Panic Window

Diversification activates when:

- Cost stagnates
- Priority coverage stagnates

---

## 5. `planner.py`

Main TRACE pipeline orchestrator.

### Pipeline Stages

```text
Environment Generation
        ↓
Priority-aware A*-DARP
        ↓
PW-CVRP Construction
        ↓
Delta RL-MA Optimization
        ↓
Trajectory Generation
        ↓
Metric Evaluation
```

### Main Entry Point

```python
build_trace_plan(cfg)
```

Returns:

- Robot paths
- Simulation metrics
- Solver histories
- Coverage information

---

## 6. `metrics.py`

Responsible for:

- Metric computation
- Metric-curve generation
- Metric-file naming
- Spreadsheet saving
- Spreadsheet loading

### Metrics Supported

| Metric | Description |
|---|---|
| Coverage | Percentage of free cells visited |
| Priority Serviced | Priority victim coverage ratio |
| Redundancy | Repeated coverage ratio |
| Total Turns | Aggregate turning cost |
| Mission Time | Maximum route duration |
| Hidden Victim Score | Hidden victims discovered |
| Global Cost | Delta RL-MA convergence cost |
| Global Priority | Priority convergence history |

---

## 7. `ui.py`

Interactive simulation visualization.

### UI Layout

#### Left Panel

Displays:

- Simulation parameters
- Coverage metrics
- Priority servicing
- Mission statistics
- Hidden victim score
- Robot-specific details

#### Center Panel

Displays:

- Environment grid
- Obstacles
- Robot partitions
- Robot trajectories
- Priority cells
- Hidden victims
- Robot positions

#### Right Panel

Displays:

- Coverage plots
- Priority-servicing plots
- Delta RL-MA convergence graphs

---

# Installation

## 1. Clone Repository

```bash
git clone <repository-url>
cd TRACE_implementation
```

---

## 2. Create Virtual Environment (Recommended)

### Linux / macOS

```bash
python -m venv venv
source venv/bin/activate
```

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Required Python Version

Recommended:

```text
Python 3.10+
```

---

# Dependencies

Main dependencies:

```text
numpy
matplotlib
pygame
networkx
pandas
```

---

# Running TRACE Simulation

## Basic Execution

```bash
python run_trace.py
```

---

## Example with Parameters

```bash
python run_trace.py \
    --robots 4 \
    --rows 20 \
    --cols 20 \
    --seed 7 \
    --obstacle-ratio 0.10 \
    --generations 280 \
    --population 42 \
    --energy 95
```

---

# Command-Line Parameters

| Parameter | Description |
|---|---|
| `--robots` | Number of robots/UAVs |
| `--rows` | Grid rows |
| `--cols` | Grid columns |
| `--seed` | Random seed |
| `--obstacle-ratio` | Obstacle density |
| `--generations` | Delta RL-MA generations |
| `--population` | Population size |
| `--energy` | Robot energy budget |
| `--no-ui` | Run without graphical interface |

---

# Running Without UI

Useful for experiments and batch execution.

```bash
python run_trace.py --no-ui
```

---

# Saved Simulation Metrics

After every simulation, all metrics are automatically saved.

### Save Location

```text
results/
```

### File Naming Convention

```text
TRACE_metrics_R20x20_UAV4_OBS0100_HP0100_HV0200_E95_POP42_GEN280_SEED7.csv
```

The filename includes:

| Tag | Meaning |
|---|---|
| R20x20 | Grid size |
| UAV4 | Number of robots |
| OBS0100 | Obstacle ratio |
| HP0100 | High-priority ratio |
| HV0200 | Hidden victim ratio |
| E95 | Energy budget |
| POP42 | Population size |
| GEN280 | RL generations |
| SEED7 | Random seed |

---

# Spreadsheet Contents

Each CSV file stores:

- Simulation parameters
- Step-wise coverage metrics
- Priority servicing history
- Redundancy history
- Turn history
- Hidden victim discovery history
- Delta RL-MA convergence history
- Final mission statistics

The spreadsheet can be opened directly in:

- Microsoft Excel
- LibreOffice Calc
- Google Sheets
- Pandas

---

# Plotting Saved Metrics

## Plot Existing Simulation Results

```bash
python plot_metrics.py \
    --robots 4 \
    --rows 20 \
    --cols 20 \
    --seed 7 \
    --obstacle-ratio 0.10 \
    --generations 280 \
    --population 42 \
    --energy 95
```

The plotting script:

1. Reconstructs the metric filename using the provided parameters
2. Reads the CSV spreadsheet
3. Generates plots from saved data

---

# Generated Plots

The plotting pipeline generates:

1. Coverage vs Simulation Step
2. Priority Servicing vs Simulation Step
3. Coverage Redundancy
4. Total Turns
5. Hidden Victim Discovery
6. Delta RL-MA Cost Convergence
7. Delta RL-MA Priority Convergence

---

# Experiment Workflow

Typical workflow:

```text
1. Configure parameters
2. Run simulation
3. Metrics automatically saved
4. Reconstruct metric filename
5. Load spreadsheet
6. Generate plots
7. Compare experiments
```

---

# TRACE Algorithm Summary

## Stage 1 — Priority-aware A*-DARP

Goals:

- Balanced decomposition
- Connectivity preservation
- Obstacle-aware partitioning
- Priority balancing

---

## Stage 2 — PW-CVRP

Goals:

- Energy-aware routing
- Priority weighting
- Turn minimization
- Route feasibility

---

## Stage 3 — Delta RL-MA

Goals:

- Adaptive operator selection
- Local-search optimization
- Stagnation escape
- Priority-aware optimization

---

# Evaluation Metrics

## Coverage

Measures the percentage of free cells visited.

---

## Priority Servicing

Measures how effectively high-priority victim regions are visited.

---

## Redundancy

Measures repeated visits to already-covered cells.

Lower is better.

---

## Total Turns

Measures path smoothness and energy consumption.

Lower is better.

---

## Mission Time

Measures the total execution duration.

Lower is better.

---

## Hidden Victim Score

Measures successful discovery of previously unknown victims.

Higher is better.

---

# Reproducibility

To ensure reproducible experiments:

- Use fixed seeds
- Store all metric spreadsheets
- Maintain parameter-specific filenames
- Compare experiments using identical configurations

---

# Future Extensions

Potential future improvements:

- Dynamic obstacles
- Multi-trip recharge stations
- Real-time communication constraints
- ROS2 integration
- Drone swarm hardware integration
- QMIX online adaptation
- Transformer-based coordination
- 3D UAV environments
- Wind-aware dynamics
- Battery degradation models
- Sim-to-real transfer

---

# Citation

If using this implementation in academic work, please cite the associated TRACE framework and the underlying A*-DARP, QMIX, and Delta RL-MA references.

---

# License

This project is intended for research and academic experimentation.

Please add the appropriate open-source license before public release.

---

# Contact

AIMS Lab
Department of Information Technology
IIIT Allahabad
Prayagraj, India

