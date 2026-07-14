from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple
import random
import numpy as np
from .config import Cell, TraceConfig
from .cvrp import PWCVRPInstance, route_cost

OPERATORS = ["swap", "2-opt", "relocate", "block-swap", "block-relocate"]

@dataclass
class SolverResult:
    best_order: List[Cell]
    best_trips: List[List[Cell]]
    best_cost: float
    best_turns: int
    cost_history: List[float]
    priority_history: List[float]
    action_history: List[str]
    panic_events: int
    mean_cost_history: List[float]
    best_trips_history: List[List[List[Cell]]]

class DeltaRLMemeticSolver:
    """TRACE Stage 3: Delta RL-guided memetic solver with victim-aware state."""

    def __init__(self, cfg: TraceConfig, rng: random.Random | None = None,
                 state_aug: bool = True, pri_panic: bool = True, all_panic: bool = True,
                 block_ops: bool = True, q_learn: bool = True, anneal: bool = True,
                 allowed_operators: List[str] | None = None):
        self.cfg = cfg
        self.rng = rng or random.Random(cfg.seed)
        self.q: Dict[Tuple[str, int], float] = {}
        self.feature_min = np.full(7, np.inf)
        self.feature_max = np.full(7, -np.inf)
        self.epsilon = cfg.epsilon_start
        self.state_aug = state_aug
        self.pri_panic = pri_panic
        self.all_panic = all_panic
        self.block_ops = block_ops
        self.q_learn = q_learn
        self.anneal = anneal
        
        op_map = {
            "swap": 0,
            "two_opt": 1,
            "relocate": 2,
            "block_swap": 3,
            "block_relocate": 4
        }
        if allowed_operators is not None:
            self.allowed_actions = [op_map[op] for op in allowed_operators if op in op_map]
        else:
            self.allowed_actions = [0, 1, 2, 3, 4] if block_ops else [0, 1, 2]

    def solve(self, inst: PWCVRPInstance) -> SolverResult:
        cfg = self.cfg
        if not inst.customers:
            return SolverResult([], [], 0.0, 0, [0.0] * cfg.generations, [1.0] * cfg.generations, [], 0, [0.0] * cfg.generations, [[] for _ in range(cfg.generations)])

        population = self._initial_population(inst)
        costs, trips_list, turns_list = self._evaluate(population, inst)
        best_idx = int(np.argmin(costs))
        best_order = population[best_idx].copy()
        best_cost = float(costs[best_idx])
        best_trips = [t.copy() for t in trips_list[best_idx]]
        best_turns = int(turns_list[best_idx])

        cost_history: List[float] = []
        priority_history: List[float] = []
        action_history: List[str] = []
        mean_cost_history: List[float] = []
        best_trips_history: List[List[List[Cell]]] = []
        panic_events = 0
        last_priority_improvement = 0
        panic_until = -1

        for gen in range(cfg.generations):
            state, total_cost, violation, rho = self._state(population, costs, trips_list, inst)
            action = self._choose_action(state)
            action_history.append(OPERATORS[action])
            before_total = total_cost
            before_rho = rho

            elite_idx = int(np.argmin(costs))
            elite = population[elite_idx].copy()
            new_population: List[List[Cell]] = []
            blind_accept = gen < panic_until
            order_indices = list(range(len(population)))
            for idx in order_indices:
                chrom = population[idx].copy()
                if idx == elite_idx:
                    new_population.append(elite.copy())
                    continue
                candidate = chrom.copy()
                candidate_cost = float(costs[idx])
                for _ in range(cfg.local_search_depth):
                    mutated = self._apply_operator(candidate, action)
                    mutated_cost, _, _ = route_cost(mutated, inst, cfg)
                    accept_degrading = blind_accept and idx >= len(population) // 2
                    if mutated_cost <= candidate_cost or accept_degrading:
                        candidate = mutated
                        candidate_cost = mutated_cost
                new_population.append(candidate)

            population = new_population
            costs, trips_list, turns_list = self._evaluate(population, inst)

            gen_best_idx = int(np.argmin(costs))
            if costs[gen_best_idx] < best_cost:
                best_cost = float(costs[gen_best_idx])
                best_order = population[gen_best_idx].copy()
                best_trips = [t.copy() for t in trips_list[gen_best_idx]]
                best_turns = int(turns_list[gen_best_idx])

            next_state, after_total, after_violation, after_rho = self._state(population, costs, trips_list, inst)
            if after_rho > before_rho + 1e-9:
                last_priority_improvement = gen
            reward = (before_total - after_total) - cfg.violation_penalty * after_violation + cfg.mu * (after_rho - before_rho)
            self._update_q(state, action, reward, next_state)
            if self.anneal:
                self.epsilon = max(cfg.epsilon_floor, self.epsilon * cfg.epsilon_decay)

            cost_history.append(best_cost)
            priority_history.append(after_rho)
            mean_cost_history.append(float(np.mean(costs)))
            best_trips_history.append([t.copy() for t in best_trips])

            if self._panic_required(gen, cost_history, priority_history, last_priority_improvement):
                panic_events += 1
                panic_until = gen + cfg.panic_duration
                self.epsilon = 1.0
                order = np.argsort(costs)
                top = [population[int(i)].copy() for i in order[: len(population) // 2]]
                bottom = [p.copy() for p in top]
                population = top + bottom
                while len(population) < cfg.population_size:
                    population.append(best_order.copy())
                costs, trips_list, turns_list = self._evaluate(population, inst)

        return SolverResult(
            best_order,
            best_trips,
            best_cost,
            best_turns,
            cost_history,
            priority_history,
            action_history,
            panic_events,
            mean_cost_history,
            best_trips_history
        )

    def _initial_population(self, inst: PWCVRPInstance) -> List[List[Cell]]:
        population = []
        seed_order = sorted(inst.customers, key=lambda c: -inst.priority_weights.get(c, 1.0))
        population.append(seed_order)
        nearest = [inst.depot]
        remaining = set(inst.customers)
        while remaining:
            last = nearest[-1]
            nxt = min(remaining, key=lambda c: abs(c[0]-last[0]) + abs(c[1]-last[1]) - 0.15 * inst.priority_weights.get(c, 1.0))
            nearest.append(nxt)
            remaining.remove(nxt)
        population.append(nearest[1:])
        while len(population) < self.cfg.population_size:
            chrom = inst.customers.copy()
            self.rng.shuffle(chrom)
            population.append(chrom)
        return population

    def _evaluate(self, population: Sequence[Sequence[Cell]], inst: PWCVRPInstance):
        costs, trips, turns = [], [], []
        for chrom in population:
            c, t, k = route_cost(chrom, inst, self.cfg)
            costs.append(c)
            trips.append(t)
            turns.append(k)
        return np.array(costs, dtype=float), trips, np.array(turns, dtype=int)

    def _state(self, population, costs, trips_list, inst: PWCVRPInstance) -> Tuple[str, float, float, float]:
        route_counts = np.array([len(t) for t in trips_list], dtype=float)
        # Priority coverage in early part of current best chromosome.
        best_idx = int(np.argmin(costs))
        best = population[best_idx]
        priority_set = set(c for c in inst.customers if inst.priority_weights.get(c, 1.0) > 1.5)
        if self.state_aug and priority_set:
            early = max(1, int(len(best) * self.cfg.early_service_fraction))
            rho = len(set(best[:early]) & priority_set) / len(priority_set)
        else:
            rho = 1.0
        raw = np.array([
            float(np.sum(costs)),
            float(np.max(costs)),
            float(np.min(costs)),
            0.0,
            float(np.mean(route_counts)) if len(route_counts) else 0.0,
            float(np.var(costs)),
            float(rho),
        ])
        self.feature_min = np.minimum(self.feature_min, raw)
        self.feature_max = np.maximum(self.feature_max, raw)
        denom = self.feature_max - self.feature_min
        norm = np.full(7, 0.5)
        mask = denom > 1e-12
        norm[mask] = (raw[mask] - self.feature_min[mask]) / denom[mask]
        bins = tuple("L" if v < 0.33 else "M" if v < 0.66 else "H" for v in norm)
        return "(" + ",".join(bins) + ")", float(raw[0]), float(raw[3]), rho

    def _choose_action(self, state: str) -> int:
        if not self.q_learn or self.rng.random() < self.epsilon:
            return self.rng.choice(self.allowed_actions)
        qs = [self.q.get((state, a), 0.0) for a in self.allowed_actions]
        return self.allowed_actions[int(np.argmax(qs))]

    def _update_q(self, state: str, action: int, reward: float, next_state: str) -> None:
        old = self.q.get((state, action), 0.0)
        next_best = max(self.q.get((next_state, a), 0.0) for a in self.allowed_actions)
        self.q[(state, action)] = old + self.cfg.rl_alpha * (reward + self.cfg.rl_gamma * next_best - old)

    def _apply_operator(self, chrom: List[Cell], action: int) -> List[Cell]:
        n = len(chrom)
        if n < 2:
            return chrom.copy()
        c = chrom.copy()
        if action == 0:  # swap
            i, j = self.rng.sample(range(n), 2)
            c[i], c[j] = c[j], c[i]
        elif action == 1:  # 2-opt
            i, j = sorted(self.rng.sample(range(n), 2))
            c[i:j+1] = reversed(c[i:j+1])
        elif action == 2:  # relocate
            i, j = self.rng.sample(range(n), 2)
            val = c.pop(i)
            c.insert(j, val)
        elif action == 3 and n >= 4:  # block swap
            size = self.rng.randint(1, min(5, n // 2))
            i = self.rng.randint(0, n - size)
            j = self.rng.randint(0, n - size)
            if abs(i - j) >= size:
                a, b = sorted((i, j))
                block1 = c[a:a+size]
                block2 = c[b:b+size]
                c[a:a+size] = block2
                c[b:b+size] = block1
        elif action == 4 and n >= 3:  # block relocate
            size = self.rng.randint(1, min(5, n - 1))
            i = self.rng.randint(0, n - size)
            block = c[i:i+size]
            del c[i:i+size]
            j = self.rng.randint(0, len(c))
            c[j:j] = block
        return c

    def _panic_required(self, gen: int, costs: List[float], rho_hist: List[float], last_prio: int) -> bool:
        if not self.all_panic:
            return False
        if gen < max(self.cfg.panic_cost_window, 10):
            return False
        w = self.cfg.panic_cost_window
        old = costs[-w]
        new = costs[-1]
        cost_stagnant = old > 0 and abs(old - new) / abs(old) < self.cfg.panic_threshold
        priority_stagnant = self.pri_panic and (gen - last_prio) >= self.cfg.panic_priority_window and rho_hist[-1] < 0.999
        return cost_stagnant or priority_stagnant
