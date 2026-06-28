from abc import ABC, abstractmethod
from typing import Dict, Any


class TrackingABC(ABC):
    """Contract for WandB / Metrics tracking."""

    @abstractmethod
    def log_episode(self, metrics: Dict[str, float], step: int) -> None:
        """
        Logs terminal metrics (Episodic Reward, Length, System Efficiency).
        Called ONLY at terminal states to prevent I/O blocking during the step loop.
        """
        pass

    @abstractmethod
    def render_frame(self, state_data: Dict[str, Any]) -> None:
        """Generates visual debugging frames if visualization is active."""
        pass