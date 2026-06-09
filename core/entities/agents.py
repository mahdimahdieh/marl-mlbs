from dataclasses import dataclass, field
from typing import List, Dict


# --- Core Data Containers ---

@dataclass
class BaseStation:
    """Parent container for base stations."""
    id: int
    capacity: int
    maximum_coverage_radius: float

    # Track coverage as an integer count
    current_coverage_count: int = 0

    @property
    def is_at_capacity(self) -> bool:
        return self.current_coverage_count >= self.capacity

    def get_coverage_efficiency(self) -> float:
        if self.capacity > 0:
            return self.current_coverage_count / self.capacity
        return 0.0


@dataclass
class FlyingBaseStation(BaseStation):
    host_vbs_id: int = None

    # Action Space: 0 to 16
    # 0: Hover
    # 1-8: N, NE, E, SE, S, SW, W, NW (Half Distance)
    # 9-16: N, NE, E, SE, S, SW, W, NW  (Full Distance)
    current_offset_zone: int = 0


@dataclass
class VehicleBaseStation(BaseStation):
    current_branch_id: int = 0
    current_slot_index: int = 0

    # Link to tethered drones
    tethered_fbs_ids: List[int] = field(default_factory=list)


# --- The Station Tracker ---

class AgentManager:
    """Registry for all base stations in the environment."""

    def __init__(self):
        self.vbs_registry: Dict[int, VehicleBaseStation] = {}
        self.fbs_registry: Dict[int, FlyingBaseStation] = {}

    def register_vbs(self, vbs: VehicleBaseStation):
        self.vbs_registry[vbs.id] = vbs

    def register_fbs(self, fbs: FlyingBaseStation):
        # tether the drone to the host's registry
        self.fbs_registry[fbs.id] = fbs
        if fbs.host_vbs_id in self.vbs_registry:
            self.vbs_registry[fbs.host_vbs_id].tethered_fbs_ids.append(fbs.id)

    def reset_coverage(self):
        for vbs in self.vbs_registry.values():
            vbs.current_coverage_count = 0
        for fbs in self.fbs_registry.values():
            fbs.current_coverage_count = 0

    def get_total_efficiency(self) -> float:
        total_covered = (sum(v.current_coverage_count for v in self.vbs_registry.values()) +
                         sum(f.current_coverage_count for f in self.fbs_registry.values()))
        total_capacity = (sum(v.capacity for v in self.vbs_registry.values()) +
                          sum(f.capacity for f in self.fbs_registry.values()))

        return total_covered / total_capacity if total_capacity > 0 else 0.0