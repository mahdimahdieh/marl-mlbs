from dataclasses import dataclass, field
from typing import List, Dict


# --- Core Data Containers ---

@dataclass
class BaseStation:
    """Parent container for base stations."""
    id: int
    capacity: int
    coverage_radius: float

    # RAW count from coverage_matrix.sum(axis=1). May double-count users covered
    # by multiple stations simultaneously. Used for capacity headroom tracking and
    # visualisation ONLY — never pass this to reward or termination logic.
    current_coverage_count: int = 0

    @property
    def is_at_capacity(self) -> bool:
        return self.current_coverage_count >= self.capacity

    def get_coverage_efficiency(self) -> float:
        """
        Per-station capacity saturation [0.0, 1.0]. Diagnostic only.
        This is NOT the RL objective. A station can be "full" while the network
        covers only a small fraction of the user population.
        """
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
        """
        DIAGNOSTIC ONLY — this is NOT the RL objective.

        Returns the fraction of total station capacity that is filled, including
        double-counted users. This metric saturates to 1.0 immediately at step 1
        whenever the number of users in range exceeds any station's capacity limit,
        even if only a tiny fraction of total users are uniquely covered.

        True Network Coverage Efficiency — the actual RL objective — is:
            unique_users_covered (set-union) / total_users
        This is maintained by CoverageParallelEnv as `env.last_true_coverage`.
        """
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

    def assign_home_branches(self, num_branches: int) -> None:
        """Call once after all VBS registration, before env.reset()."""
        for idx, vbs in enumerate(sorted(self.vbs_registry.values(), key=lambda v: v.id)):
            vbs.home_branch_id = (idx % num_branches) + 1  # branches are 1-indexed node ids