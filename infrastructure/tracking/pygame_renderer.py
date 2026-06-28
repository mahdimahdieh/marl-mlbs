import pygame
import numpy as np
from typing import Dict, Any


class PygameRenderer:
    """Real-time visualizer for MARL inference mimicking Matplotlib scatter layouts."""

    def __init__(self, map_dim: list, window_size: int = 800):
        pygame.init()
        self.map_dim = map_dim
        self.scale = window_size / max(map_dim)
        self.window_size = window_size

        self.screen = pygame.display.set_mode((window_size, window_size))
        pygame.display.set_caption("5G Multi-Agent Coverage Inference")
        self.font = pygame.font.SysFont('Arial', 14)

        # Color Palette matching your reference image
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

    def _to_px(self, x: float, y: float) -> tuple:
        """Translates environment coordinates to screen pixels (Y is inverted in Pygame)."""
        return int(x * self.scale), int(self.window_size - (y * self.scale))

    def _draw_dashed_circle(self, surface, color, center, radius, width=2, dash_length=10):
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

    def render(self, env: Any, step: int):
        self.screen.fill(self.colors["bg"])

        # 1. Draw Grid
        for i in range(0, int(self.map_dim[0]), 10):
            px_x, _ = self._to_px(i, 0)
            pygame.draw.line(self.screen, self.colors["grid"], (px_x, 0), (px_x, self.window_size))
        for i in range(0, int(self.map_dim[1]), 10):
            _, px_y = self._to_px(0, i)
            pygame.draw.line(self.screen, self.colors["grid"], (0, px_y), (self.window_size, px_y))

        # 2. Draw Graph Topology
        nodes = env.graph_engine.get_nodes()
        links = env.graph_engine.get_links()

        for link in links:
            s_node = next(n for n in nodes if n["id"] == link["source"])
            t_node = next(n for n in nodes if n["id"] == link["target"])
            pygame.draw.line(self.screen, self.colors["graph_edge"],
                             self._to_px(s_node["x"], s_node["y"]),
                             self._to_px(t_node["x"], t_node["y"]), 3)

        for node in nodes:
            pygame.draw.circle(self.screen, self.colors["graph_node"], self._to_px(node["x"], node["y"]), 6)

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
        # Using the fast vectorized check just for rendering colors
        np_coords = np.array(agent_coords)
        np_radii = np.array(coverage_radii)

        # Distance matrix
        diff = np_coords[:, None, :] - users[None, :, :]
        dist = np.linalg.norm(diff, axis=2)
        covered_mask = np.any(dist <= np_radii[:, None], axis=0)

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
        title = self.font.render(f"5G Real-Time Inference - Step {step}", True, (0, 0, 0))
        self.screen.blit(title, (20, 20))

        pygame.display.flip()