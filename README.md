# TRACE Implementation

This folder contains a consolidated Python implementation of the combined TRACE paper:

**TRACE: Target-aware Routing And Coverage with Evolutionary Reinforcement Learning**

The implementation joins the three uploaded ideas/codebases into a single working pipeline:

1. **Priority-aware A\*-DARP decomposition** for balanced multi-robot area allocation.
2. **Priority-weighted CVRP construction** for energy-aware regional routing.
3. **Delta RL-guided Memetic Algorithm** for adaptive local-search operator selection.

The simulator window has three parts:

- **Left panel:** simulation parameters, mission details, live coverage/victim statistics, and per-robot region loads.
- **Center panel:** grid-world SAR simulation with obstacles, partitions, priority cells, hidden victims, paths, and robot positions.
- **Right panel:** live graph plot for coverage and priority servicing, plus Delta RL-MA solver metrics.

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
python run_trace.py --robots 4
```

Useful options:

```bash
python run_trace.py --robots 8 --generations 350 --population 50 --energy 100
python run_trace.py --robots 4 --no-ui
```

Keyboard controls inside the simulation:

- `Space`: pause/resume
- `R`: replay the trajectory animation

## File structure

```text
run_trace.py                  # main entry point
trace/config.py               # configuration and hyperparameters
trace/environment.py          # SAR grid, obstacles, population prior, victim map
trace/astar.py                # A* and BFS utilities
trace/decomposition.py        # priority-aware A*-DARP stage
trace/cvrp.py                 # priority-weighted CVRP instance and route expansion
trace/delta_rl_memetic.py     # Delta RL-MA solver with 7-feature victim-aware state
trace/metrics.py              # coverage, redundancy, turn, and victim metrics
trace/simulator.py            # end-to-end TRACE plan builder
trace/ui.py                   # Pygame simulation window with left/right panels
```

## Notes

- The implementation is designed as a clean, runnable research prototype rather than a direct copy-paste of the original zip code. The original DARP, QMIX/victim-prioritization, and Delta RL-MA logic were consolidated into a common interface.
- QMIX is used conceptually through its victim-aware reward structure and priority-map formulation; the final TRACE execution uses the paper's hierarchical implementation: A*-DARP partitioning followed by priority-weighted routing and Delta RL-MA route improvement.
- Default generations/population are reduced compared with the paper to keep the GUI responsive. Increase `--generations` and `--population` for stronger optimization.
