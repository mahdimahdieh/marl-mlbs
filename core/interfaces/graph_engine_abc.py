from abc import ABC, abstractmethod
from typing import Tuple, List
import numpy as np


class GraphEngineABC(ABC):
    """Contract for VBS topological movement."""

    @abstractmethod
    def get_edge_coordinates(self, start_vertex: int, end_vertex: int, traveled: float) -> Tuple[float, float]:
        """Calculates physical (x,y) world coordinates for observation spaces."""
        pass

    @abstractmethod
    def get_action_mask(self, current_branch_id: int) -> np.ndarray:
        """Must return a 1D np.int8 array of valid discrete actions
        (1 = legal, 0 = masked)."""
        pass

    @abstractmethod
    def get_map_dimension(self) -> List[float]:
        """Used to normalize observation spaces to [-1, 1]."""
        pass