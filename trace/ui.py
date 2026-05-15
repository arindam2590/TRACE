from __future__ import annotations
import io
from typing import List, Sequence, Tuple
import numpy as np
from .config import TraceConfig, Cell
from .simulator import TracePlan
from .metrics import compute_metrics

ROBOT_COLORS = [
    (79, 145, 255), (87, 204, 153), (255, 184, 77), (230, 100, 255),
    (255, 100, 100), (100, 220, 255), (180, 180, 255), (190, 230, 95)
]


def run_ui(plan: TracePlan) -> None:
    import pygame
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = plan.env.cfg
    grid_w = cfg.cols * cfg.cell_size
    grid_h = cfg.rows * cfg.cell_size
    width = cfg.left_panel_width + grid_w + cfg.right_panel_width
    height = max(grid_h + 2 * cfg.top_margin, 720)
    pygame.init()
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("TRACE: Target-aware Routing And Coverage with Evolutionary RL")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 15)
    small = pygame.font.SysFont("consolas", 13)
    title = pygame.font.SysFont("consolas", 20, bold=True)

    paused = False
    step = 0
    frame = 0
    max_step = max((len(p) for p in plan.paths), default=1)
    coverage_curve: List[float] = []
    priority_curve: List[float] = []

    def text(surface, s, pos, color=(230, 235, 245), fnt=None):
        surface.blit((fnt or font).render(str(s), True, color), pos)

    def render_graph() -> pygame.Surface:
        fig, ax = plt.subplots(figsize=(4.3, 2.7), dpi=100)
        ax.plot(coverage_curve, label="coverage")
        ax.plot(priority_curve, label="priority")
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("simulation step")
        ax.set_ylabel("ratio")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="lower right", fontsize=8)
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        return pygame.image.load(buf, "graph.png").convert()

    graph_surf = None

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    step = 0
                    coverage_curve.clear()
                    priority_curve.clear()

        if not paused and frame % cfg.animate_every_n_frames == 0:
            step = min(step + 1, max_step)
        frame += 1

        prefix_paths = [p[: min(step, len(p))] for p in plan.paths]
        live = compute_metrics(prefix_paths, plan.env.free, plan.env.priority_cells, plan.env.hidden_victims)
        if len(coverage_curve) < step:
            coverage_curve.append(live.coverage)
            priority_curve.append(live.priority_serviced)
        if step % 8 == 0 or graph_surf is None:
            graph_surf = render_graph()

        screen.fill((18, 20, 28))
        left = pygame.Rect(0, 0, cfg.left_panel_width, height)
        right_x = cfg.left_panel_width + grid_w
        right = pygame.Rect(right_x, 0, cfg.right_panel_width, height)
        pygame.draw.rect(screen, (28, 31, 43), left)
        pygame.draw.rect(screen, (28, 31, 43), right)

        # Left panel: parameters and live details
        y = 18
        text(screen, "TRACE Simulation", (18, y), (255, 220, 120), title); y += 34
        lines = [
            "Pipeline: A*-DARP -> PW-CVRP -> Delta RL-MA",
            f"Grid: {cfg.rows} x {cfg.cols}",
            f"Robots: {cfg.n_robots}",
            f"Obstacles: {len(plan.env.obstacles)}",
            f"Priority cells: {len(plan.env.priority_cells)}",
            f"Hidden victim cells: {len(plan.env.hidden_victims)}",
            f"Energy budget/trip: {cfg.energy_budget}",
            f"DARP beta: {cfg.beta_start:.2f} -> {cfg.beta_min:.2f}",
            f"Memetic generations: {cfg.generations}",
            f"Operators: swap, 2-opt, relocate, block moves",
        ]
        for line in lines:
            text(screen, line, (18, y), (210, 218, 235), small); y += 21
        y += 10
        text(screen, "Live mission details", (18, y), (150, 210, 255), font); y += 26
        live_lines = [
            f"Step: {step}/{max_step} {'PAUSED' if paused else ''}",
            f"Coverage: {100*live.coverage:6.2f}%",
            f"Priority serviced: {100*live.priority_serviced:6.2f}%",
            f"Redundancy: {live.redundancy:6.3f}",
            f"Hidden victim score: {live.discovered_hidden_victims}",
            f"Final mission time: {plan.metrics.mission_time}",
            f"Final total turns: {plan.metrics.total_turns}",
        ]
        for line in live_lines:
            text(screen, line, (18, y), (235, 235, 235), small); y += 21
        y += 12
        text(screen, "Per-robot region load", (18, y), (150, 210, 255), font); y += 25
        for rid, (cnt, load) in enumerate(zip(plan.decomposition.cell_counts, plan.decomposition.priority_loads)):
            col = ROBOT_COLORS[rid % len(ROBOT_COLORS)]
            pygame.draw.rect(screen, col, pygame.Rect(18, y + 3, 12, 12))
            text(screen, f"R{rid}: cells={cnt:3d}, p-load={load:5.2f}", (38, y), (230, 230, 230), small)
            y += 20
        y += 16
        text(screen, "Keys: Space=pause/resume, R=replay", (18, height - 30), (180, 190, 205), small)

        # Center grid
        ox, oy = cfg.left_panel_width, cfg.top_margin
        pygame.draw.rect(screen, (246, 247, 250), pygame.Rect(ox, oy, grid_w, grid_h))
        for r in range(cfg.rows):
            for c in range(cfg.cols):
                rect = pygame.Rect(ox + c * cfg.cell_size, oy + r * cfg.cell_size, cfg.cell_size, cfg.cell_size)
                cell = (r, c)
                if cell in plan.env.obstacles:
                    pygame.draw.rect(screen, (20, 20, 20), rect)
                else:
                    rid = int(plan.decomposition.assignment[cell]) if plan.decomposition.assignment[cell] >= 0 else -1
                    if rid >= 0:
                        base = ROBOT_COLORS[rid % len(ROBOT_COLORS)]
                        fill = tuple(int(235 * 0.72 + x * 0.28) for x in base)
                        pygame.draw.rect(screen, fill, rect)
                    if cell in plan.env.priority_cells:
                        pygame.draw.circle(screen, (210, 55, 45), rect.center, max(4, cfg.cell_size // 5))
                    elif cell in plan.env.hidden_victims:
                        pygame.draw.circle(screen, (255, 180, 85), rect.center, max(3, cfg.cell_size // 7))
                pygame.draw.rect(screen, (120, 124, 135), rect, 1)

        # Covered path traces and robot positions.
        visited_global = set()
        for rid, path in enumerate(plan.paths):
            col = ROBOT_COLORS[rid % len(ROBOT_COLORS)]
            prefix = path[: min(step, len(path))]
            for cell in prefix:
                visited_global.add(cell)
            for a, b in zip(prefix[:-1], prefix[1:]):
                pygame.draw.line(screen, col,
                                 (ox + a[1] * cfg.cell_size + cfg.cell_size // 2, oy + a[0] * cfg.cell_size + cfg.cell_size // 2),
                                 (ox + b[1] * cfg.cell_size + cfg.cell_size // 2, oy + b[0] * cfg.cell_size + cfg.cell_size // 2), 3)
            if prefix:
                cur = prefix[-1]
                pygame.draw.circle(screen, col,
                                   (ox + cur[1] * cfg.cell_size + cfg.cell_size // 2, oy + cur[0] * cfg.cell_size + cfg.cell_size // 2),
                                   cfg.cell_size // 3)
                text(screen, str(rid), (ox + cur[1] * cfg.cell_size + 8, oy + cur[0] * cfg.cell_size + 5), (0, 0, 0), small)

        # Right panel: graph and solver summary
        y = 18
        text(screen, "Performance Plot", (right_x + 18, y), (255, 220, 120), title); y += 35
        if graph_surf:
            screen.blit(graph_surf, (right_x + 10, y)); y += graph_surf.get_height() + 20
        text(screen, "Delta RL-MA summary", (right_x + 18, y), (150, 210, 255), font); y += 26
        for rid, res in enumerate(plan.solver_results):
            line = f"R{rid}: cost={res.best_cost:7.2f}, turns={res.best_turns:3d}, panic={res.panic_events}"
            text(screen, line, (right_x + 18, y), (232, 235, 242), small); y += 20
        y += 12
        text(screen, "Final mission metrics", (right_x + 18, y), (150, 210, 255), font); y += 26
        summary = [
            f"Coverage: {100*plan.metrics.coverage:.2f}%",
            f"Priority servicing: {100*plan.metrics.priority_serviced:.2f}%",
            f"Redundancy: {plan.metrics.redundancy:.3f}",
            f"Mission time: {plan.metrics.mission_time}",
            f"Total turns: {plan.metrics.total_turns}",
            f"Hidden victim score: {plan.metrics.discovered_hidden_victims}",
        ]
        for line in summary:
            text(screen, line, (right_x + 18, y), (232, 235, 242), small); y += 21

        pygame.display.flip()
        clock.tick(cfg.fps)

    pygame.quit()
