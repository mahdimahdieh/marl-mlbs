from abc import ABC, abstractmethod
import numpy as np


class NetworkSimABC(ABC):
    """Contract for the PyWiSim spatial coverage adapter."""

    @abstractmethod
    def reset_spatial_distribution(self, seed: int = None) -> None:
        """Called by env.reset() to generate a new user distribution for the episode."""
        pass

    @abstractmethod
    def compute_batched_coverage(self, agent_coords: np.ndarray, coverage_radii: np.ndarray) -> np.ndarray:
        """(N, 2) coords + (N,) radii in, (N,) coverage counts out —
        vectorized, no Python loops."""
        pass