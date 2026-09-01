from dataclasses import dataclass, field
from typing import List, Dict


# --- Core Data Containers ---

@dataclass
class BaseStation:
    """Parent container for base stations."""
    id: int
    capacity: int
    coverage_radius: float

    # RAW count from coverage_matrix.sum(axis=1); double-counts overlapping
    # users. Diagnostics/visualisation ONLY — never reward or termination.
    current_coverage_count: int = 0

    # injective slot, decoupled from home_branch_id's mod-3 collisions
    identity_index: int = 0

    @property
    def is_at_capacity(self) -> bool:
        return self.current_coverage_count >= self.capacity

    def get_coverage_efficiency(self) -> float:
        """Per-station capacity saturation [0.0, 1.0]. Diagnostic only — NOT
        the RL objective (see AgentManager.get_capacity_utilization)."""
        if self.capacity > 0:
            return min(self.current_coverage_count, self.capacity) / self.capacity
        return 0.0

    def reset_state(self):
        self.current_coverage_count = 0


@dataclass
class FlyingBaseStation(BaseStation):
    host_vbs_id: int = None
    maximum_distance: float = None
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
    home_branch_id: int = 0
    ema_x: float = None
    ema_y: float = None
    tethered_fbs_ids: List[int] = field(default_factory=list)

    def update_ema(self, x: float, y: float, decay: float = 0.9) -> None:
        if self.ema_x is None:
            self.ema_x, self.ema_y = x, y          # cold start: snap
        else:
            self.ema_x = decay * self.ema_x + (1 - decay) * x
            self.ema_y = decay * self.ema_y + (1 - decay) * y

    def reset_state(self):
        super().reset_state()
        self.current_branch_id = 0
        self.current_slot_index = 0
        self.ema_x = None
        self.ema_y = None


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

    def get_capacity_utilization(self) -> float:
        """DIAGNOSTIC ONLY — fraction of total station capacity filled
        (double-counts users). The actual RL objective is CoverageParallelEnv's
        env.last_true_coverage: unique users covered / total users."""
        total_capacity = (
            sum(v.capacity for v in self.vbs_registry.values()) +
            sum(f.capacity for f in self.fbs_registry.values())
        )
        if total_capacity <= 0:
            return 0.0

        total_filled = (
            sum(min(v.current_coverage_count, v.capacity) for v in self.vbs_registry.values()) +
            sum(min(f.current_coverage_count, f.capacity) for f in self.fbs_registry.values())
        )
        return total_filled / total_capacity

    def get_total_efficiency(self) -> float:
        """Clamped per-station efficiencies summed over total capacity — a
        capacity-saturation diagnostic, not the RL objective."""
        total_capacity = (sum(v.capacity for v in self.vbs_registry.values()) +
                          sum(f.capacity for f in self.fbs_registry.values()))

        if total_capacity <= 0:
            return 0.0

        total_effective_coverage = (
                sum(min(v.current_coverage_count, v.capacity) for v in self.vbs_registry.values()) +
                sum(min(f.current_coverage_count, f.capacity) for f in self.fbs_registry.values())
        )

        return total_effective_coverage / total_capacity

    def assign_home_branches(self, num_branches: int) -> None:
        """Call once after all VBS registration, before env.reset()."""
        for idx, vbs in enumerate(sorted(self.vbs_registry.values(), key=lambda v: v.id)):
            vbs.home_branch_id = (idx % num_branches) + 1  # branches are 1-indexed node ids

    def assign_identity_indices(self) -> None:
        """Call once after all registration, before env.reset(). Injective
        per-type slot so a shared-weight policy never sees bit-identical
        inputs for two physically distinct agents."""
        for idx, vbs in enumerate(sorted(self.vbs_registry.values(), key=lambda v: v.id)):
            vbs.identity_index = idx
        for idx, fbs in enumerate(sorted(self.fbs_registry.values(), key=lambda f: f.id)):
            fbs.identity_index = idx