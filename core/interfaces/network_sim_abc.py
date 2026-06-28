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
        """
        STRICT RL REQUIREMENT:
        Takes an (N, 2) array of coordinates and an (N,) array of radii.
        Returns an (N,) integer array of coverage counts.
        NO FOR LOOPS. Concrete implementations must use vectorized spatial math.
        """
        pass