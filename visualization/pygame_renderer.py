import math

import pygame
import numpy as np
from typing import Any


class PygameRenderer:
    """Real-time visualizer for MARL inference mimicking Matplotlib scatter layouts.

    Layout: a square map canvas (window_size x window_size) on the left plus a
    fixed-width dark telemetry sidebar on the right, giving a
    (window_size + PANEL_WIDTH) x window_size window. Everything is drawn with
    raw pygame primitives (rect / line / font.render) so per-frame cost stays
    negligible — no Matplotlib figure conversion ever happens.
    """

    PANEL_WIDTH = 400
    SPARKLINE_WINDOW = 160

    def __init__(self, map_dim: list, window_size: int = 800):
        pygame.init()
        self.map_dim = map_dim
        self.scale = window_size / max(map_dim)
        self.window_size = window_size
        self.panel_width = self.PANEL_WIDTH

        self.screen = pygame.display.set_mode(
            (window_size + self.panel_width, window_size)
        )
        pygame.display.set_caption("5G Multi-Agent Coverage Inference")
        self.font = pygame.font.SysFont('Arial', 14)
        self.font_title = pygame.font.SysFont('Arial', 19, bold=True)
        self.font_section = pygame.font.SysFont('Arial', 13, bold=True)
        self.font_mono = pygame.font.SysFont(
            'Consolas,Menlo,DejaVu Sans Mono,Courier New', 13)
        self.font_small = pygame.font.SysFont(
            'Consolas,Menlo,DejaVu Sans Mono,Courier New', 11)

        # Color Palette matching your reference image (map canvas)
        self.colors = {
            "bg": (255, 255, 255),
            "grid": (230, 230, 230),
            "graph_edge": (150, 150, 150),
            "graph_node": (0, 0, 0),
            "user_uncovered": (255, 50, 50),
            "user_covered": (0, 180, 0),
            "vbs": (80, 0, 255),  # Purple/Blue
            "fbs": (255, 165, 0),  # Orange
            "vbs_cov": (120, 50, 255, 50),  # Alpha Purple
            "fbs_cov": (255, 200, 50, 50),  # Alpha Orange
            "text_bg": (200, 220, 240, 200)
        }

        # Dark/neutral telemetry sidebar palette
        self.panel = {
            "bg": (24, 26, 32),
            "border": (70, 76, 88),
            "divider": (56, 62, 74),
            "text": (228, 231, 238),
            "text_dim": (148, 154, 166),
            "accent": (0, 194, 255),
            "good": (52, 220, 130),
            "warn": (255, 168, 46),
            "bad": (255, 84, 84),
            "bar_bg": (44, 49, 60),
            "bar_fill": (0, 194, 255),
            "bar_fill_good": (52, 220, 130),
            "bar_fill_warn": (255, 168, 46),
            "threshold": (255, 216, 61),
            "spark": (0, 194, 255),
            "chip_off": (90, 96, 108),
        }

        # Rolling telemetry state for the coverage sparkline
        self._cov_history: list = []
        self._last_rendered_step = 0

    # ------------------------------------------------------------------ #
    # Coordinate helpers                                                 #
    # ------------------------------------------------------------------ #

    def _to_px(self, x: float, y: float) -> tuple:
        """Translates environment coordinates to screen pixels (Y is inverted in Pygame)."""
        return int(x * self.scale), int(self.window_size - (y * self.scale))

    def _draw_dashed_circle(self, surface, color, center, radius, width=1, dash_length=10):
        """Helper to draw dashed coverage boundaries."""
        circumference = 2 * np.pi * radius
        dashes = int(circumference / dash_length)
        for i in range(dashes):
            if i % 2 == 0:
                start_angle = (i / dashes) * 2 * np.pi
                end_angle = ((i + 1) / dashes) * 2 * np.pi
                pygame.draw.arc(surface, color,
                                (center[0] - radius, center[1] - radius, radius * 2, radius * 2),
                                start_angle, end_angle, width)

    # ------------------------------------------------------------------ #
    # Main entry point                                                   #
    # ------------------------------------------------------------------ #

    def render(self, env: Any, step: int, episode_done: bool = False):
        """Draws one frame: map canvas (left) + live telemetry HUD (right).

        `episode_done` lets callers flag the terminal frame; omitting it keeps
        the legacy call signature `render(env, step)` fully working.
        """
        # New-episode detection: the step counter rewinds between episodes.
        if step <= self._last_rendered_step:
            self._cov_history.clear()
        self._last_rendered_step = step

        coverage = float(getattr(env, "last_true_coverage", 0.0))
        self._cov_history.append(coverage)
        if len(self._cov_history) > self.SPARKLINE_WINDOW:
            del self._cov_history[:len(self._cov_history) - self.SPARKLINE_WINDOW]

        self._draw_map(env, step)
        self._draw_sidebar(env, step, coverage, episode_done)

        pygame.display.flip()

    # ------------------------------------------------------------------ #
    # Map canvas (0..window_size, 0..window_size)                        #
    # ------------------------------------------------------------------ #

    def _draw_map(self, env: Any, step: int):
        self.screen.fill(self.colors["bg"])

        # 1. Draw Grid (clipped to the map canvas, never bleeding into the sidebar)
        for i in range(0, int(self.map_dim[0]), 10):
            px_x, _ = self._to_px(i, 0)
            pygame.draw.line(self.screen, self.colors["grid"],
                             (px_x, 0), (px_x, self.window_size))
        for i in range(0, int(self.map_dim[1]), 10):
            _, px_y = self._to_px(0, i)
            pygame.draw.line(self.screen, self.colors["grid"],
                             (0, px_y), (self.window_size, px_y))

        # 2. Draw Graph Topology (read directly from the native NetworkX object)
        graph = env.graph_engine.graph
        nodes = [{"id": n_id, **data} for n_id, data in graph.nodes(data=True)]
        links = [{"source": u, "target": v} for u, v in graph.edges()]

        for link in links:
            s_node = next(n for n in nodes if n["id"] == link["source"])
            t_node = next(n for n in nodes if n["id"] == link["target"])
            pygame.draw.line(self.screen, self.colors["graph_edge"],
                             self._to_px(s_node["x"], s_node["y"]),
                             self._to_px(t_node["x"], t_node["y"]), 3)

        for node in nodes:
            pygame.draw.circle(self.screen, self.colors["graph_node"],
                               self._to_px(node["x"], node["y"]), 6)

        # 3. Extract Simulation Data
        users = env.sim_adapter.user_coords
        agent_coords = []
        coverage_radii = []
        for agent_id in env.agents:
            obj, is_vbs = env._get_agent_obj(agent_id)
            x, y = env._calculate_world_coords(obj, is_vbs)
            agent_coords.append([x, y])
            coverage_radii.append(obj.coverage_radius)

        # 4. Draw Users
        if len(users) > 0 and len(agent_coords) > 0:
            np_coords = np.array(agent_coords)
            np_radii = np.array(coverage_radii)

            # Distance matrix
            diff = np_coords[:, None, :] - users[None, :, :]
            dist = np.linalg.norm(diff, axis=2)
            covered_mask = np.any(dist <= np_radii[:, None], axis=0)
        else:
            covered_mask = np.zeros(len(users), dtype=bool)

        for i, (ux, uy) in enumerate(users):
            color = self.colors["user_covered"] if covered_mask[i] else self.colors["user_uncovered"]
            pygame.draw.circle(self.screen, color, self._to_px(ux, uy), 4)

        # 5. Draw Agents & Coverage
        for agent_id in env.agents:
            obj, is_vbs = env._get_agent_obj(agent_id)
            ax, ay = env._calculate_world_coords(obj, is_vbs)
            px, py = self._to_px(ax, ay)
            rad_px = int(obj.coverage_radius * self.scale)

            if is_vbs:
                self._draw_dashed_circle(self.screen, self.colors["vbs"], (px, py), rad_px)
                pygame.draw.rect(self.screen, self.colors["vbs"], (px - 8, py - 8, 16, 16))
            else:
                self._draw_dashed_circle(self.screen, self.colors["fbs"], (px, py), rad_px)
                # Draw Triangle for FBS
                pygame.draw.polygon(self.screen, self.colors["fbs"],
                                    [(px, py - 8), (px - 8, py + 8), (px + 8, py + 8)])
                # Draw tether line to host VBS
                host_obj = env.agent_manager.vbs_registry[obj.host_vbs_id]
                hx, hy = env._calculate_world_coords(host_obj, True)
                pygame.draw.line(self.screen, self.colors["fbs"], (px, py), self._to_px(hx, hy), 1)

            # Draw Stats Box
            stat_text = f"{agent_id.upper()} Cov: {obj.current_coverage_count}/{obj.capacity}"
            text_surf = self.font.render(stat_text, True, (0, 0, 0))
            self.screen.blit(text_surf, (px + 10, py - 20))

        # 6. Title
        title = self.font.render(f"Real-Time Inference GUI - Step {step}", True, (0, 0, 0))
        self.screen.blit(title, (20, 20))

    # ------------------------------------------------------------------ #
    # Telemetry sidebar (window_size..window_size+panel_width)           #
    # ------------------------------------------------------------------ #

    def _draw_sidebar(self, env: Any, step: int, coverage: float, episode_done: bool):
        x0 = self.window_size
        y0, y1 = 0, self.window_size
        w = self.panel_width

        # Panel background over the full right strip (also masks any map bleed)
        pygame.draw.rect(self.screen, self.panel["bg"], (x0, y0, w, y1 - y0))
        pygame.draw.line(self.screen, self.panel["border"], (x0, 0), (x0, y1), 2)

        px = x0 + 16          # inner padding
        pw = w - 32           # inner width

        goal = float(getattr(env, "termination_goal", 0.9))
        max_cycles = int(getattr(env, "max_cycles", 100))

        # -- Header ------------------------------------------------------ #
        self.screen.blit(self.font_title.render("LIVE TELEMETRY", True, self.panel["accent"]),
                         (px, 14))
        step_surf = self.font_mono.render(
            f"STEP {step:4d} / {max_cycles}", True, self.panel["text"])
        self.screen.blit(step_surf, (px, 44))
        # Remaining step budget mini-bar
        budget = min(step / max(max_cycles, 1), 1.0)
        pygame.draw.rect(self.screen, self.panel["bar_bg"], (px + 170, 48, pw - 170, 10))
        if budget > 0:
            pygame.draw.rect(self.screen, self.panel["bar_fill_warn"],
                             (px + 170, 48, int((pw - 170) * budget), 10))

        status, status_color, status_note = self._status(env, coverage, goal, episode_done)
        self._draw_status_chip(px, 68, pw, status, status_color, status_note)
        self._draw_divider(px, y0 + 100, pw)

        # -- Coverage telemetry ------------------------------------------ #
        self._draw_section_header(px, 112, pw, "COVERAGE TELEMETRY")
        cov_pct = coverage * 100.0
        self.screen.blit(self.font_mono.render("TRUE COVERAGE (UNION)", True, self.panel["text_dim"]),
                         (px, 132))
        val_color = self.panel["good"] if coverage >= goal else (
            self.panel["warn"] if coverage >= 0.5 * goal else self.panel["bad"])
        val_surf = self.font_mono.render(f"{cov_pct:5.1f}%", True, val_color)
        self.screen.blit(val_surf, (px + pw - val_surf.get_width(), 132))

        self._draw_bar(px, 152, pw, 20, coverage, val_color, threshold=goal)
        goal_label = self.font_small.render(
            f"goal threshold {goal * 100.0:.0f}%", True, self.panel["threshold"])
        self.screen.blit(goal_label, (px, 176))

        self._draw_sparkline(px, 198, pw, 96, goal)
        self.screen.blit(
            self.font_small.render(
                f"rolling trend (last {min(len(self._cov_history), self.SPARKLINE_WINDOW)} steps)",
                True, self.panel["text_dim"]),
            (px, 298))
        self._draw_divider(px, 316, pw)

        # -- Network health & efficiency ---------------------------------- #
        self._draw_section_header(px, 328, pw, "NETWORK HEALTH & EFFICIENCY")
        util = float(env.agent_manager.get_capacity_utilization())
        self.screen.blit(self.font_mono.render("CAPACITY UTILIZATION", True, self.panel["text_dim"]),
                         (px, 348))
        util_surf = self.font_mono.render(f"{util * 100.0:5.1f}%", True, self.panel["bar_fill"])
        self.screen.blit(util_surf, (px + pw - util_surf.get_width(), 348))
        self._draw_bar(px, 366, pw, 14, util, self.panel["bar_fill"])
        self.screen.blit(
            self.font_small.render("filled capacity / total capacity (raw, double-counts)",
                                   True, self.panel["text_dim"]),
            (px, 384))

        multi_users, redundancy = self._overlap_diagnostics(env)
        self.screen.blit(self.font_mono.render("OVERLAP / REDUNDANCY", True, self.panel["text_dim"]),
                         (px, 406))
        red_color = self.panel["bad"] if redundancy >= 0.25 else (
            self.panel["bar_fill_warn"] if redundancy >= 0.10 else self.panel["good"])
        red_surf = self.font_mono.render(f"{redundancy * 100.0:5.1f}%", True, red_color)
        self.screen.blit(red_surf, (px + pw - red_surf.get_width(), 406))
        self._draw_bar(px, 424, pw, 14, redundancy, red_color)
        self.screen.blit(
            self.font_small.render(f"{multi_users} users covered by 2+ stations",
                                   True, self.panel["text_dim"]),
            (px, 442))
        self._draw_divider(px, 460, pw)

        # -- Agent status matrix ------------------------------------------ #
        self._draw_section_header(px, 472, pw, "AGENT STATUS MATRIX")
        self._draw_agent_matrix(env, px, pw, 492, self.window_size - 12)

    def _status(self, env: Any, coverage: float, goal: float, episode_done: bool):
        if episode_done:
            if coverage >= goal:
                return "TERMINATED", self.panel["good"], "goal reached"
            return "TERMINATED", self.panel["warn"], "max cycles"
        if coverage >= goal:
            return "SOLVING", self.panel["good"], "goal satisfied"
        return "RUNNING", self.panel["accent"], "solving in progress"

    def _overlap_diagnostics(self, env: Any):
        """Users covered by 2+ stations and the redundancy ratio, derived from
        the env's own coverage matrix snapshot (None before the first step)."""
        matrix = getattr(env, "last_coverage_matrix", None)
        total_users = int(getattr(env.sim_adapter, "num_users", 0))
        if matrix is None or total_users <= 0:
            return 0, 0.0
        per_user = np.asarray(matrix).sum(axis=0)
        multi_users = int(np.count_nonzero(per_user > 1))
        return multi_users, multi_users / total_users

    def _draw_status_chip(self, x, y, w, label, color, note):
        pygame.draw.rect(self.screen, color, (x, y + 3, 10, 10), border_radius=2)
        label_surf = self.font_mono.render(f"STATUS  {label}", True, color)
        self.screen.blit(label_surf, (x + 18, y))
        note_surf = self.font_small.render(note, True, self.panel["text_dim"])
        self.screen.blit(note_surf, (x + w - note_surf.get_width(), y + 2))

    def _draw_divider(self, x, y, w):
        pygame.draw.line(self.screen, self.panel["divider"], (x, y), (x + w, y), 1)

    def _draw_section_header(self, x, y, w, text):
        surf = self.font_section.render(text, True, self.panel["text"])
        self.screen.blit(surf, (x, y))
        pygame.draw.line(self.screen, self.panel["accent"],
                         (x, y + 18), (x + w, y + 18), 1)

    def _draw_bar(self, x, y, w, h, frac, fill_color, threshold=None):
        """Horizontal gauge built from plain rects; optional threshold tick."""
        pygame.draw.rect(self.screen, self.panel["bar_bg"], (x, y, w, h), border_radius=2)
        fill_w = int(w * min(max(frac, 0.0), 1.0))
        if fill_w > 0:
            pygame.draw.rect(self.screen, fill_color, (x, y, fill_w, h), border_radius=2)
        if threshold is not None:
            tx = x + int(w * min(max(threshold, 0.0), 1.0))
            pygame.draw.line(self.screen, self.panel["threshold"],
                             (tx, y - 4), (tx, y + h + 4), 2)

    def _draw_sparkline(self, x, y, w, h, goal):
        """Rolling line chart of true coverage; fixed [0, 1] y-scale plus a
        dashed goal line. Pure pygame.draw.line — no per-frame figure cost."""
        pygame.draw.rect(self.screen, self.panel["bar_bg"], (x, y, w, h))
        pygame.draw.rect(self.screen, self.panel["divider"], (x, y, w, h), 1)

        # Midline + goal dashed line
        mid_y = y + h // 2
        pygame.draw.line(self.screen, self.panel["divider"],
                         (x + 2, mid_y), (x + w - 2, mid_y), 1)
        goal_y = y + h - int(h * min(max(goal, 0.0), 1.0))
        dash, seg = 8, 5
        cx = x + 2
        while cx < x + w - 2:
            pygame.draw.line(self.screen, self.panel["threshold"],
                             (cx, goal_y), (min(cx + dash, x + w - 2), goal_y), 1)
            cx += dash + seg

        series = self._cov_history[-self.SPARKLINE_WINDOW:]
        n = len(series)
        if n >= 2:
            span = max(self.SPARKLINE_WINDOW - 1, n - 1)
            points = []
            for i, v in enumerate(series):
                px = x + 2 + int((w - 4) * (i / span))
                py = y + h - 2 - int((h - 4) * min(max(v, 0.0), 1.0))
                points.append((px, py))
            pygame.draw.lines(self.screen, self.panel["spark"], False, points, 2)
            # Endpoint marker (green once the goal is met)
            dot_color = self.panel["good"] if series[-1] >= goal else self.panel["spark"]
            pygame.draw.circle(self.screen, dot_color, points[-1], 3)

        # Scale labels
        self.screen.blit(self.font_small.render("1.0", True, self.panel["text_dim"]), (x + 2, y - 1))
        self.screen.blit(self.font_small.render("0.0", True, self.panel["text_dim"]),
                         (x + 2, y + h - 13))

    def _draw_agent_matrix(self, env: Any, x, w, y_top, y_bottom):
        """Grouped VBS rows with tethered-FBS sub-rows: (branch, slot) and
        capacity ratio per VBS; offset zone + distance-to-host per FBS."""
        am = env.agent_manager
        y = y_top
        for vbs_id in sorted(am.vbs_registry):
            vbs = am.vbs_registry[vbs_id]
            if y + 16 > y_bottom:
                return
            filled = min(vbs.current_coverage_count, vbs.capacity)
            ratio = filled / vbs.capacity if vbs.capacity > 0 else 0.0
            ratio_color = (self.panel["warn"] if ratio >= 1.0 else
                           self.panel["bar_fill_good"] if ratio >= 0.5 else
                           self.panel["bar_fill"])
            header = f"VBS_{vbs_id}  B{vbs.current_branch_id} S{vbs.current_slot_index:<2d}  {filled}/{vbs.capacity}"
            self.screen.blit(self.font_mono.render(header, True, self.panel["text"]), (x, y))
            pct = self.font_small.render(f"{ratio * 100.0:3.0f}%", True, ratio_color)
            self.screen.blit(pct, (x + w - pct.get_width(), y + 1))
            y += 17
            if y + 8 > y_bottom:
                return
            self._draw_bar(x + 16, y, w - 16, 6, ratio, ratio_color)
            y += 12

            host_x, host_y = env._calculate_world_coords(vbs, True)
            for fbs_id in vbs.tethered_fbs_ids:
                fbs = am.fbs_registry.get(fbs_id)
                if fbs is None:
                    continue
                if y + 15 > y_bottom:
                    return
                fbs_x, fbs_y = env._calculate_world_coords(fbs, False)
                dist = math.hypot(fbs_x - host_x, fbs_y - host_y)
                line = f"+- FBS_{fbs_id}  Z:{fbs.current_offset_zone:2d}  d:{dist:6.1f}m"
                line_color = self.panel["warn"] if fbs.current_offset_zone == 0 else self.panel["text_dim"]
                self.screen.blit(self.font_mono.render(line, True, line_color), (x + 10, y))
                y += 15
            y += 6
