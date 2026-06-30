import functools
import numpy as np
from typing import Dict, Any, Tuple
from pettingzoo import ParallelEnv
from gymnasium import spaces

from core.entities.agents import AgentManager, VehicleBaseStation, FlyingBaseStation
from infrastructure.graph.networkx_engine import NetworkXRoadEngine
from infrastructure.simulation.pywisim_adapter import PyWiSimAdapter


class CoverageParallelEnv(ParallelEnv):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "vbs_fbs_coverage_v1"
    }

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        # 1. Dependency Injection of High-Performance Adapters
        self.agent_manager: AgentManager = config["agent_manager"]
        self.graph_engine: NetworkXRoadEngine = config["graph_engine"]
        self.sim_adapter: PyWiSimAdapter = config["sim_adapter"]

        self.max_cycles = config.get("max_cycles", 100)
        self.map_dim = self.graph_engine.get_map_dimension()
        self.max_slot_per_branch = float(config.get("max_slot_per_branch", 10))

        self.possible_agents = (
            [f"vbs_{v.id}" for v in self.agent_manager.vbs_registry.values()] +
            [f"fbs_{f.id}" for f in self.agent_manager.fbs_registry.values()]
        )
        self.agents = self.possible_agents[:]
        self.step_count = 0

        # --- TRUE COVERAGE STATE ---
        # These replace every downstream use of AgentManager.get_total_efficiency().
        # last_true_coverage: unique users covered / total users, via set-union semantics.
        # last_coverage_matrix: (N_agents, N_users) bool snapshot from the previous step,
        # used to build the third observation dimension without re-running spatial math.
        self.last_true_coverage: float = 0.0
        self.last_coverage_matrix: np.ndarray = None

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: str) -> spaces.Box:
        if "vbs" in agent:
            # [norm_x, norm_y, coverage_frac, norm_slot_idx,
            #  branch_0_hot, branch_1_hot, branch_2_hot]
            # norm_slot_idx: 0.0=at center, 1.0=at edge terminus
            # branch_hot:    one-hot for active branch; all-zeros when at center (slot==0)
            return spaces.Box(low=0.0, high=1.0, shape=(7,), dtype=np.float32)
        else:
            # [norm_x, norm_y, coverage_frac, norm_offset_zone]
            # norm_offset_zone: current polar zone / 16 → tells FBS which half-/full-dist ring it's in
            return spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: str) -> spaces.Discrete:
        if "vbs" in agent:
            return spaces.Discrete(3)
        else:
            return spaces.Discrete(17)

    def reset(self, seed: int = None, options: Dict = None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        self.agents = self.possible_agents[:]
        self.step_count = 0

        # Clear coverage snapshots so _compute_observations_and_masks knows it is
        # in cold-start mode and returns 0.0 for the coverage obs dimension.
        self.last_true_coverage = 0.0
        self.last_coverage_matrix = None

        self.agent_manager.reset_all_agents()
        self.sim_adapter.reset_spatial_distribution(seed=seed)

        # Generate Initial Tensors
        obs, infos = self._compute_observations_and_masks()
        return obs, infos

    def step(self, actions: Dict[str, int]) -> Tuple[
        Dict[str, np.ndarray], Dict[str, float],
        Dict[str, bool], Dict[str, bool], Dict[str, Any]
    ]:
        # ------------------------------------------------------------------ #
        # PHASE 1: PHYSICS & MOVEMENT (unchanged)                             #
        # ------------------------------------------------------------------ #
        self._apply_actions(actions)

        # ------------------------------------------------------------------ #
        # PHASE 2: SPATIAL VECTORIZATION                                      #
        # ------------------------------------------------------------------ #
        agent_coords = []
        coverage_radii = []
        agent_mapping = []

        for agent_id in self.agents:
            agent_obj, is_vbs = self._get_agent_obj(agent_id)
            x, y = self._calculate_world_coords(agent_obj, is_vbs)
            agent_coords.append([x, y])
            coverage_radii.append(agent_obj.coverage_radius)
            agent_mapping.append(agent_obj)

        np_coords = np.array(agent_coords, dtype=np.float32)   # (N, 2)
        np_radii = np.array(coverage_radii, dtype=np.float32)  # (N,)

        # (N, M) bool — the single source of truth for all coverage math this step
        coverage_matrix = self.sim_adapter.compute_coverage_matrix(np_coords, np_radii)

        # Update raw per-station counts for diagnostics / visualisation only.
        # These WILL double-count overlapping users — that is expected and correct
        # for the capacity headroom display. Never use these for reward/termination.
        coverage_counts = coverage_matrix.sum(axis=1, dtype=np.int32)
        for obj, count in zip(agent_mapping, coverage_counts):
            obj.current_coverage_count = int(count)

        # Snapshot matrix for observation computation in _compute_observations_and_masks.
        # Index alignment is guaranteed: self.agents has the same order as the loop above.
        self.last_coverage_matrix = coverage_matrix

        # ------------------------------------------------------------------ #
        # PHASE 3: TRUE NETWORK COVERAGE EFFICIENCY                           #
        # ------------------------------------------------------------------ #
        total_users = self.sim_adapter.num_users
        n_agents = len(self.agents)

        # SET-UNION semantics: a user is counted ONCE regardless of how many agents
        # cover them. This is the real-world network coverage metric.
        any_covered_mask = np.any(coverage_matrix, axis=0)    # (M,) bool
        total_covered = int(any_covered_mask.sum())

        # ∈ [0.0, 1.0] — this is the RL objective. Starts at ~0.60-0.65 (all agents
        # co-located with FBS radius=45 covering ~63% of the 100×100 map), and the
        # theoretical maximum rises as agents learn to spread to uncovered regions.
        true_coverage_efficiency = float(total_covered) / float(max(total_users, 1))
        self.last_true_coverage = true_coverage_efficiency

        # ------------------------------------------------------------------ #
        # PHASE 4: ENGINEERED REWARD SIGNAL                                   #
        # ------------------------------------------------------------------ #
        # Three-component reward designed to simultaneously:
        #   (a) assign individual credit via counterfactual (anti-free-rider)
        #   (b) align every agent with the global cooperative objective
        #   (c) directly penalise spatial redundancy
        #
        # Reward range at extreme states (400 users, 10 agents, radius-45 FBS):
        #   All bunched at center: ~(-0.73) per step  ← negative signal to spread
        #   Optimal spread (~40% unique): ~(1.04) per step  ← positive convergence target
        #
        # Tune these three weights as hyperparameters if the graph layout changes:
        REWARD_SCALE = 1.0          # Scales reward to [≈-7.3, ≈10.4] — stable for PPO clip=0.2
        MARGINAL_WEIGHT = 0.65        # Individual Shapley-value approximation
        TEAM_WEIGHT = 0.3            # Shared cooperative gradient
        OVERLAP_PENALTY_WEIGHT = 0.05 # Explicit redundancy suppressor

        rewards = {}
        for i, agent_id in enumerate(self.agents):
            # Agent i's raw boolean coverage vector
            agent_i_vec = coverage_matrix[i]          # (M,) bool
            agent_i_count = int(agent_i_vec.sum())

            if n_agents > 1:
                # Boolean mask selecting every agent EXCEPT agent i
                others_mask = np.ones(n_agents, dtype=bool)
                others_mask[i] = False

                # Union of coverage from all agents except i
                others_union = np.any(coverage_matrix[others_mask], axis=0)  # (M,) bool
                covered_without_i = int(others_union.sum())

                # Overlap ratio: fraction of agent i's own covered users that are
                # ALREADY covered by the rest of the team.
                # 0.0 = fully unique coverage,  1.0 = fully redundant with others
                if agent_i_count > 0:
                    overlap_ratio = float(
                        np.count_nonzero(agent_i_vec & others_union)
                    ) / float(agent_i_count)
                else:
                    # Agent covers nobody → no redundancy penalty, but no credit either
                    overlap_ratio = 0.0
            else:
                covered_without_i = 0
                overlap_ratio = 0.0

            # Counterfactual marginal contribution ∈ [0.0, 1.0]
            # = unique users agent i brings to the team union / total population
            # Naturally collapses to 0 for fully overlapping agents without any
            # explicit "if overlapping: penalise" branch.
            marginal_contribution = float(
                total_covered - covered_without_i
            ) / float(max(total_users, 1))


            # Final blended reward. The overlap_penalty term is SEPARATE from the
            # marginal term: marginal punishes redundancy by reducing the reward to 0,
            # while overlap_penalty actively pushes redundant agents negative —
            # providing a gradient even when two agents cover exactly the same users
            # (where marginal_contribution AND team_coverage might still be positive).
            rewards[agent_id] = REWARD_SCALE * (
                MARGINAL_WEIGHT * marginal_contribution
                + TEAM_WEIGHT * true_coverage_efficiency
                - OVERLAP_PENALTY_WEIGHT * overlap_ratio
            )



        # ------------------------------------------------------------------ #
        # PHASE 5: TERMINATION                                                #
        # ------------------------------------------------------------------ #
        self.step_count += 1
        env_truncation = self.step_count >= self.max_cycles

        # FIX: terminate when 95% of USERS are uniquely covered, not when stations
        # are full. With FBS radius=45 at step 1, true_coverage ≈ 0.63 — the
        # episode will now run for max_cycles steps until PPO optimises agent spread.
        env_termination = true_coverage_efficiency >= 0.95

        terminations = {agent: env_termination for agent in self.agents}
        truncations = {agent: env_truncation for agent in self.agents}

        # ------------------------------------------------------------------ #
        # PHASE 6: OBSERVATIONS & MASKS                                       #
        # ------------------------------------------------------------------ #
        obs, infos = self._compute_observations_and_masks()

        if env_termination or env_truncation:
            self.agents = []

        return obs, rewards, terminations, truncations, infos

    # --- INTERNAL MECHANICS ---

    def _apply_actions(self, actions: Dict[str, int]):
        for agent_id, action in actions.items():
            agent_obj, is_vbs = self._get_agent_obj(agent_id)
            if is_vbs:
                if agent_obj.current_slot_index == 0:
                    agent_obj.current_branch_id = action + 1
                    agent_obj.current_slot_index += 1
                else:
                    if action == agent_obj.current_branch_id - 1:
                        agent_obj.current_slot_index = min(10, agent_obj.current_slot_index + 1)
                    else:
                        agent_obj.current_slot_index = max(0, agent_obj.current_slot_index - 1)
            else:
                agent_obj.current_offset_zone = action

    def _calculate_world_coords(self, agent_obj, is_vbs) -> Tuple[float, float]:
        if is_vbs:
            if agent_obj.current_slot_index == 0:
                return self.graph_engine.get_edge_coordinates(0, 1, 0.0)
            else:
                traveled = agent_obj.current_slot_index / 10.0
                return self.graph_engine.get_edge_coordinates(0, agent_obj.current_branch_id, traveled)
        else:
            host_vbs = self.agent_manager.vbs_registry[agent_obj.host_vbs_id]
            hx, hy = self._calculate_world_coords(host_vbs, True)
            if agent_obj.current_offset_zone == 0:
                return hx, hy
            dist_multiplier = 0.5 if agent_obj.current_offset_zone <= 8 else 1.0
            angle_idx = (agent_obj.current_offset_zone - 1) % 8
            angle = angle_idx * (np.pi / 4)
            radius = agent_obj.maximum_distance * dist_multiplier
            return hx + radius * np.cos(angle), hy + radius * np.sin(angle)

    def _compute_observations_and_masks(self) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        obs = {}
        infos = {}
        total_users = self.sim_adapter.num_users
        MAX_ZONE = 16.0  # FBS zones 0–16

        for i, agent_id in enumerate(self.agents):
            agent_obj, is_vbs = self._get_agent_obj(agent_id)
            x, y = self._calculate_world_coords(agent_obj, is_vbs)

            norm_x = np.clip(x / self.map_dim[0], 0.0, 1.0)
            norm_y = np.clip(y / self.map_dim[1], 0.0, 1.0)

            # Coverage fraction — unchanged from original
            if (self.last_coverage_matrix is not None
                    and i < len(self.last_coverage_matrix)):
                raw_coverage_frac = np.clip(
                    float(self.last_coverage_matrix[i].sum()) / float(max(total_users, 1)),
                    0.0, 1.0
                )
            else:
                raw_coverage_frac = 0.0

            if is_vbs:
                # How far along the chosen branch (0.0 = center, 1.0 = terminus)?
                norm_slot = agent_obj.current_slot_index / self.max_slot_per_branch

                # One-hot encode the active branch.
                # CRITICAL: only encode branch when ACTUALLY on an edge (slot > 0).
                # At slot 0 (center) branch_id may be stale from previous backtrack;
                # all-zeros correctly signals "at center, no active commitment."
                branch_hot = np.zeros(3, dtype=np.float32)
                if agent_obj.current_slot_index > 0 and 1 <= agent_obj.current_branch_id <= 3:
                    branch_hot[agent_obj.current_branch_id - 1] = 1.0

                obs[agent_id] = np.array(
                    [norm_x, norm_y, raw_coverage_frac,
                     norm_slot,
                     branch_hot[0], branch_hot[1], branch_hot[2]],
                    dtype=np.float32
                )
            else:
                # FBS: encode current polar zone so network knows its half/full-dist ring
                norm_zone = agent_obj.current_offset_zone / MAX_ZONE
                obs[agent_id] = np.array(
                    [norm_x, norm_y, raw_coverage_frac, norm_zone],
                    dtype=np.float32
                )

            # Action mask — prevent invalid forward move at terminus only.
            # Backtracking remains legal at all slots (agent can choose to reroute).
            mask = np.ones(self.action_space(agent_id).n, dtype=np.int8)
            if is_vbs and agent_obj.current_slot_index >= 10:
                mask[agent_obj.current_branch_id - 1] = 0  # Block moving past terminus

            infos[agent_id] = {"action_mask": mask}

        return obs, infos

    def _get_raw_id(self, agent_string: str) -> int:
        return int(agent_string.split("_")[1])

    def _get_agent_obj(self, agent_string: str):
        agent_id = self._get_raw_id(agent_string)
        is_vbs = "vbs" in agent_string
        obj = (self.agent_manager.vbs_registry[agent_id]
               if is_vbs else self.agent_manager.fbs_registry[agent_id])
        return obj, is_vbs