from dataclasses import dataclass, field
from typing import List, Dict


# --- Core Data Containers ---

@dataclass
class BaseStation:
    """Parent container for base stations."""
    id: int
    capacity: int
    coverage_radius: float

    # Track coverage as an integer count
    current_coverage_count: int = 0

    @property
    def is_at_capacity(self) -> bool:
        return self.current_coverage_count >= self.capacity

    def get_coverage_efficiency(self) -> float:
        if self.capacity > 0:
            # STRICT FIX: Clamp the efficiency to [0.0, 1.0] max.
            return min(self.current_coverage_count, self.capacity) / self.capacity
        return 0.0

    def reset_state(self):
        self.current_coverage_count = 0


@dataclass
class FlyingBaseStation(BaseStation):
    host_vbs_id: int = None
    maximum_distance: int = None
    # Action Space: 0 to 16
    # 0: Hover
    # 1-8: N, NE, E, SE, S, SW, W, NW (Half Distance)
    # 9-16: N, NE, E, SE, S, SW, W, NW  (Full Distance)
    current_offset_zone: int = 0


    def reset_state(self):
        super().reset_state()
        self.current_offset_zone = 0


@dataclass
class VehicleBaseStation(BaseStation):
    current_branch_id: int = 0
    current_slot_index: int = 0

    # Link to tethered VBS
    tethered_fbs_ids: List[int] = field(default_factory=list)

    def reset_state(self):
        super().reset_state()
        self.current_branch_id = 0
        self.current_slot_index = 0


# --- The Station Tracker ---

class AgentManager:
    """Registry for all base stations in the environment."""

    def __init__(self):
        self.vbs_registry: Dict[int, VehicleBaseStation] = {}
        self.fbs_registry: Dict[int, FlyingBaseStation] = {}

    def register_vbs(self, vbs: VehicleBaseStation):
        self.vbs_registry[vbs.id] = vbs

    def register_fbs(self, fbs: FlyingBaseStation):
        self.fbs_registry[fbs.id] = fbs
        if fbs.host_vbs_id in self.vbs_registry:
            self.vbs_registry[fbs.host_vbs_id].tethered_fbs_ids.append(fbs.id)

    def reset_all_agents(self):
        """Called during PettingZoo's env.reset() to strictly clear all state."""
        for vbs in self.vbs_registry.values():
            vbs.reset_state()
        for fbs in self.fbs_registry.values():
            fbs.reset_state()

    def get_total_efficiency(self) -> float:
        # STRICT FIX: Sum the clamped efficiencies, not the raw counts,
        total_capacity = (sum(v.capacity for v in self.vbs_registry.values()) +
                          sum(f.capacity for f in self.fbs_registry.values()))

        if total_capacity <= 0:
            return 0.0

        total_effective_coverage = (
                sum(min(v.current_coverage_count, v.capacity) for v in self.vbs_registry.values()) +
                sum(min(f.current_coverage_count, f.capacity) for f in self.fbs_registry.values())
        )

        return total_effective_coverage / total_capacity