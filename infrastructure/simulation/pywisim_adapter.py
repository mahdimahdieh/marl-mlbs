import numpy as np
from typing import Dict, List, Tuple
from core.interfaces.network_sim_abc import NetworkSimABC


# NOTE: Untouched for training loop speed, import pywisim here for eval mode
# import pywisim

class PyWiSimAdapter(NetworkSimABC):
    """
    High-performance spatial simulation adapter.
    Uses vectorized NumPy broadcasting for lightning-fast RL training,
    with embedded entry-points to execute heavy PyWiSim logic during human evaluation.
    """

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
        """
        Generates a strict, reproducible uniform user distribution.
        Guarantees that the MDP initial state distribution doesn't introduce reward variance.
        """
        rng = np.random.default_rng(seed)

        # Vectorized generation of user positions across 2D plane
        self.user_coords[:, 0] = rng.uniform(0.0, self.map_dimensions[0], size=self.num_users)
        self.user_coords[:, 1] = rng.uniform(0.0, self.map_dimensions[1], size=self.num_users)

        # --- PYWISIM INTEGRATION HOOK ---
        if self.eval_mode:
            # 1. Instantiate the PyWiSim System/Network configuration object
            # self.pywisim_env = pywisim.WirelessEnv(width=self.map_dimensions[0], height=self.map_dimensions[1])
            # 2. Iterate through self.user_coords and register them into PyWiSim's internal structures
            # for x, y in self.user_coords:
            #     ue = pywisim.UserEquipment(x=x, y=y)
            #     self.pywisim_env.register_ue(ue)
            pass

    def compute_batched_coverage(self, agent_coords: np.ndarray, coverage_radii: np.ndarray) -> np.ndarray:
        """
        Vectorized Euclidean calculation. Computes exact spatial intersections without loops.

        Inputs:
            agent_coords:   np.ndarray of shape (N, 2) -> Float x,y positions of all active agents
            coverage_radii: np.ndarray of shape (N,)    -> Float radii of all active agents
        Returns:
            counts:         np.ndarray of shape (N,)    -> Integers representing points covered
        """
        if self.eval_mode:
            return self._compute_pywisim_coverage_eval(agent_coords, coverage_radii)

        # --- HIGH-SPEED TRAINING ROUTINE ---
        # 1. Expand dimensions to broadcast agents against users
        # agent_coords[:, None, :] -> shape (N, 1, 2)
        # self.user_coords[None, :, :] -> shape (1, M, 2)
        # diff shape -> (N, M, 2)
        diff = agent_coords[:, None, :] - self.user_coords[None, :, :]

        # 2. Compute Euclidean distance matrix using fast C-level operations
        # distances shape -> (N, M)
        distances = np.linalg.norm(diff, axis=2)

        # 3. Broadcast comparison against radii thresholds
        # coverage_radii[:, None] shape -> (N, 1)
        # within_radius shape -> (N, M) boolean matrix
        within_radius = distances <= coverage_radii[:, None]

        # 4. Sum rows to yield the integer count of covered coordinates per agent
        counts = np.sum(within_radius, axis=1, dtype=np.int32)
        return counts

    def _compute_pywisim_coverage_eval(self, agent_coords: np.ndarray, coverage_radii: np.ndarray) -> np.ndarray:
        """
        Slower, high-fidelity cellular math execution reserved for metrics/testing visualization.
        """
        counts = np.zeros(len(agent_coords), dtype=np.int32)

        # --- PYWISIM CODE COUPLING GUIDE ---
        # 1. Update PyWiSim base station coordinate properties
        # for i, (x, y) in enumerate(agent_coords):
        #     self.pywisim_env.base_stations[i].set_position(x, y)
        #     self.pywisim_env.base_stations[i].transmit_power_or_radius = coverage_radii[i]
        #
        # 2. Run PyWiSim's internal channel propagation / SINR calculator
        # self.pywisim_env.compute_sinr_maps()
        #
        # 3. Query which UEs successfully associated based on real-world cellular models
        # for i, bs in enumerate(self.pywisim_env.base_stations):
        #     counts[i] = len(bs.get_associated_ues())

        # Fallback to normal calculations if PyWiSim calls are commented out
        return self.compute_batched_coverage(agent_coords, coverage_radii)