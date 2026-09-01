import numpy as np
from typing import Dict, List, Tuple
from core.interfaces.network_sim_abc import NetworkSimABC


# NOTE: Untouched for training loop speed, import pywisim here for eval mode
# import pywisim

class PyWiSimAdapter(NetworkSimABC):
    """Vectorized NumPy spatial simulation adapter for fast RL training, with
    commented hooks for heavy PyWiSim logic in eval mode."""

    def __init__(self, num_users: int = 100, map_dimensions: List[float] = None):
        self.num_users = num_users
        self.map_dimensions = map_dimensions if map_dimensions is not None else [100.0, 100.0]
        self.user_coords = np.zeros((self.num_users, 2), dtype=np.float32)
        self.eval_mode = False

        # --- PYWISIM TIE-IN STRUCTURES ---
        self.pywisim_env = None
        self.pywisim_users = []

    def set_evaluation_mode(self, enabled: bool):
        """Toggle this switch to swap from fast NumPy matrices to deep PyWiSim calls."""
        self.eval_mode = enabled

    def reset_spatial_distribution(self, seed: int = None) -> None:
        """Generates a strict, reproducible uniform user distribution."""
        rng = np.random.default_rng(seed)

        # Vectorized generation of user positions across 2D plane
        self.user_coords[:, 0] = rng.uniform(0.0, self.map_dimensions[0], size=self.num_users)
        self.user_coords[:, 1] = rng.uniform(0.0, self.map_dimensions[1], size=self.num_users)

        # --- PYWISIM INTEGRATION HOOK ---
        # Once instantiated, pywisim.WirelessEnv(...) must receive this same
        # seed, or its internal channel RNG reintroduces non-determinism.
        if self.eval_mode:
            pass

    def compute_coverage_matrix(
            self, agent_coords: np.ndarray, coverage_radii: np.ndarray
    ) -> np.ndarray:
        """Returns the (N_agents, N_users) boolean intersection matrix — the
        only correct input for differential reward computation (per-agent
        counts lose the per-user association data).

        Args:
            agent_coords:   (N, 2) float32 — world coordinates of all active agents
            coverage_radii: (N,)   float32 — coverage radius per agent
        Returns:
            within_radius:  (N, M) bool    — True if user j is within agent i's radius
        """
        # (N, 1, 2) - (1, M, 2) = (N, M, 2) delta matrix
        diff = agent_coords[:, None, :] - self.user_coords[None, :, :]
        # (N, M) Euclidean distance matrix, no Python loops
        distances = np.linalg.norm(diff, axis=2)
        # (N, M) bool — vectorized threshold comparison
        return distances <= coverage_radii[:, None]

    def compute_batched_coverage(
            self, agent_coords: np.ndarray, coverage_radii: np.ndarray
    ) -> np.ndarray:
        """Returns (N,) int32 per-agent user counts, derived from
        compute_coverage_matrix() for a single source of truth."""
        if self.eval_mode:
            return self._compute_pywisim_coverage_eval(agent_coords, coverage_radii)
        # Sum the boolean matrix along the user axis to get counts per agent
        return self.compute_coverage_matrix(
            agent_coords, coverage_radii
        ).sum(axis=1, dtype=np.int32)

    def _compute_pywisim_coverage_eval(self, agent_coords: np.ndarray, coverage_radii: np.ndarray) -> np.ndarray:
        """Slower, high-fidelity cellular math reserved for evaluation."""
        counts = np.zeros(len(agent_coords), dtype=np.int32)

        # --- PYWISIM CODE COUPLING GUIDE ---
        # 1. Set base station positions/radii on self.pywisim_env.base_stations
        # 2. Run self.pywisim_env.compute_sinr_maps()
        # 3. Read each station's associated UEs into counts[i]

        # Fallback to normal calculations if PyWiSim calls are commented out
        return self.compute_batched_coverage(agent_coords, coverage_radii)

    def compute_uncovered_density_grid(self, covered_mask: np.ndarray, grid_size: int = 4) -> np.ndarray:
        """Fixed (grid_size, grid_size) shape regardless of num_users — safe static critic input dim."""
        uncovered = self.user_coords[~covered_mask]
        if len(uncovered) == 0:
            return np.zeros((grid_size, grid_size), dtype=np.float32)
        x_bins = np.clip((uncovered[:, 0] / self.map_dimensions[0] * grid_size).astype(int), 0, grid_size - 1)
        y_bins = np.clip((uncovered[:, 1] / self.map_dimensions[1] * grid_size).astype(int), 0, grid_size - 1)
        grid = np.zeros((grid_size, grid_size), dtype=np.float32)
        np.add.at(grid, (x_bins, y_bins), 1.0)
        return grid / max(len(uncovered), 1)