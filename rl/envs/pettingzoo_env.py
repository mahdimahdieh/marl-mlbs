import functools
import numpy as np
from typing import Dict, Any, Tuple, List
from pettingzoo import ParallelEnv
from gymnasium import spaces

from core.entities.agents import AgentManager, VehicleBaseStation, FlyingBaseStation
from infrastructure.graph.networkx_engine import NetworkXRoadEngine
from infrastructure.simulation.pywisim_adapter import PyWiSimAdapter
from rl.envs.reward_normalizer import StationaryScaler


class CoverageParallelEnv(ParallelEnv):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "vbs_fbs_coverage_v1"
    }

    # Branches incident to the center node; TODO(scope): derive from
    # graph_engine once variable-topology support is tasked.
    NUM_VBS_BRANCHES = 3

    # Local sensing radius is bounded and scaled off each agent's own
    # coverage_radius (configurable via "sensing_radius_multiplier").
    DEFAULT_SENSING_RADIUS_MULTIPLIER = 2.5
    # One-time bonus on the terminating step, scaled by steps saved vs max_cycles.
    TERMINAL_SPEED_BONUS = 5.0
    # Local sensing uses an 8-sector angular histogram (not a vector mean,
    # which cancels to zero between symmetric uncovered-user clusters).
    NUM_LOCAL_SECTOR_BINS = 8

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        # 1. Dependency Injection of High-Performance Adapters
        self.agent_manager: AgentManager = config["agent_manager"]
        self.graph_engine: NetworkXRoadEngine = config["graph_engine"]
        self.sim_adapter: PyWiSimAdapter = config["sim_adapter"]

        self.termination_goal = config.get("termination_goal", 0.95)
        self.max_cycles = config.get("max_cycles", 100)
        self.map_dim = self.graph_engine.get_map_dimension()
        self.max_slot_per_branch = float(config.get("max_slot_per_branch", 10))
        self.sensing_radius_multiplier = float(
            config.get("sensing_radius_multiplier", self.DEFAULT_SENSING_RADIUS_MULTIPLIER)
        )
        # Static hyperparameter (Task 5): ramping a penalty inside step() would
        # mutate the transition dynamics mid-training (MDP non-stationarity).
        self.overlap_penalty_weight = float(config.get("overlap_penalty_weight", 0.20))

        self.possible_agents = (
            [f"vbs_{v.id}" for v in self.agent_manager.vbs_registry.values()] +
            [f"fbs_{f.id}" for f in self.agent_manager.fbs_registry.values()]
        )
        self.agents = self.possible_agents[:]
        self.step_count = 0

        # --- TRUE COVERAGE STATE ---
        # last_true_coverage: unique users covered / total users (set-union).
        # last_coverage_matrix: (N_agents, N_users) bool snapshot from the
        # previous step, reused for the observation coverage dimension.
        self.last_true_coverage: float = 0.0
        self.last_coverage_matrix: np.ndarray = None

        # Task 1: bounded [0, 1] reward metrics pass through a fixed,
        # stationary StationaryScaler — no running mean/std, no moving goalposts.
        self.team_scaler = StationaryScaler(scale=1.0)
        self.marginal_scaler = StationaryScaler(scale=1.0)

        self.uncovered_grid_size = 4
        self.global_extra_dim = 1 + self.uncovered_grid_size ** 2  # [true_coverage] + flattened density grid
        self.last_uncovered_grid = np.zeros(
            (self.uncovered_grid_size, self.uncovered_grid_size), dtype=np.float32
        )

        self.n_vbs = len(self.agent_manager.vbs_registry)
        self.n_fbs = len(self.agent_manager.fbs_registry)
        self._last_obs: Dict[str, np.ndarray] = {}

        # Per-type fixed feature widths; keep in lock-step with the
        # concatenation order in _compute_observations_and_masks.
        self.vbs_fixed_obs_dim = 10 + self.NUM_LOCAL_SECTOR_BINS + 1
        self.fbs_fixed_obs_dim = 13 + self.NUM_LOCAL_SECTOR_BINS + 1

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: str) -> spaces.Box:
        if "vbs" in agent:
            # [norm_x, norm_y, coverage_frac, norm_slot, branch_hot(3),
            #  home_branch_hot(3), local_sector_fracs(8), local_uncovered_presence(1)]
            return spaces.Box(low=-1.0, high=1.0, shape=(self.vbs_fixed_obs_dim + self.n_vbs,), dtype=np.float32)
        else:
            # [norm_x, norm_y, coverage_frac, r_frac, cos_t, sin_t,
            #  host_branch_hot(3), ema_xy_norm(2) — overwritten with the host's
            #  next-position preview by augment_fbs_obs, host_true_xy_norm(2),
            #  local_sector_fracs(8), local_uncovered_presence(1)]
            return spaces.Box(low=-1.0, high=1.0, shape=(self.fbs_fixed_obs_dim + self.n_fbs,), dtype=np.float32)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: str) -> spaces.Discrete:
        if "vbs" in agent:
            # Absolute, factored (branch, slot) selection flattened into a
            # single Discrete head; slots_per_branch includes slot 0 (center).
            slots_per_branch = int(self.max_slot_per_branch) + 1
            return spaces.Discrete(self.NUM_VBS_BRANCHES * slots_per_branch)
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
        # PHASE 1: PHYSICS & MOVEMENT                                         #
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

        # Raw per-station counts for diagnostics / visualisation only — these
        # double-count overlapping users; never use them for reward/termination.
        coverage_counts = coverage_matrix.sum(axis=1, dtype=np.int32)
        for obj, count in zip(agent_mapping, coverage_counts):
            obj.current_coverage_count = int(count)

        # Snapshot matrix for observation computation (index-aligned with self.agents).
        self.last_coverage_matrix = coverage_matrix

        # ------------------------------------------------------------------ #
        # PHASE 3: TRUE NETWORK COVERAGE EFFICIENCY                           #
        # ------------------------------------------------------------------ #
        total_users = self.sim_adapter.num_users
        n_agents = len(self.agents)

        # SET-UNION semantics: a user is counted ONCE regardless of how many
        # agents cover them — the real-world network coverage metric.
        any_covered_mask = np.any(coverage_matrix, axis=0)    # (M,) bool
        total_covered = int(any_covered_mask.sum())

        # ∈ [0.0, 1.0] — this is the RL objective.
        true_coverage_efficiency = float(total_covered) / float(max(total_users, 1))
        self.last_true_coverage = true_coverage_efficiency

        # ------------------------------------------------------------------ #
        # PHASE 4: ENGINEERED REWARD SIGNAL                                   #
        # ------------------------------------------------------------------ #
        # Stationary three-component reward: (a) counterfactual individual
        # credit (anti-free-rider), (b) shared cooperative gradient, (c) spatial
        # redundancy penalty. Both positive terms are raw, stationary-scaled
        # [0, 1] values, so reward stays positive and monotonic in coverage.
        # Tune the weights as hyperparameters if the graph layout changes.
        REWARD_SCALE = 1.0          # Scales reward to [≈-7.3, ≈10.4] — stable for PPO clip=0.2
        MARGINAL_WEIGHT = 0.65      # Individual Shapley-value approximation
        TEAM_WEIGHT = 0.15          # Shared cooperative gradient

        rewards = {}
        for i, agent_id in enumerate(self.agents):
            # Agent i's raw boolean coverage vector
            agent_i_vec = coverage_matrix[i]          # (M,) bool
            agent_i_count = int(agent_i_vec.sum())

            if n_agents > 1:
                # Union of coverage from all agents except i
                others_mask = np.ones(n_agents, dtype=bool)
                others_mask[i] = False
                others_union = np.any(coverage_matrix[others_mask], axis=0)  # (M,) bool
                covered_without_i = int(others_union.sum())

                # Fraction of agent i's covered users already covered by the
                # rest of the team: 0.0 = fully unique, 1.0 = fully redundant.
                if agent_i_count > 0:
                    overlap_ratio = float(
                        np.count_nonzero(agent_i_vec & others_union)
                    ) / float(agent_i_count)
                else:
                    # Agent covers nobody → no redundancy penalty, no credit either
                    overlap_ratio = 0.0
            else:
                covered_without_i = 0
                overlap_ratio = 0.0

            # Counterfactual marginal contribution ∈ [0.0, 1.0]: unique users
            # agent i brings to the team union / total population.
            marginal_contribution = float(
                total_covered - covered_without_i
            ) / float(max(total_users, 1))

            # The overlap penalty is separate from the marginal term: marginal
            # only reduces the reward to 0, while this actively pushes redundant
            # agents negative, giving a gradient even when marginal and team
            # terms are still positive.
            rewards[agent_id] = REWARD_SCALE * (
                    MARGINAL_WEIGHT * self.marginal_scaler(marginal_contribution)
                    + TEAM_WEIGHT * self.team_scaler(true_coverage_efficiency)
                    - self.overlap_penalty_weight * overlap_ratio
            )

        # ------------------------------------------------------------------ #
        # PHASE 5: TERMINATION                                                #
        # ------------------------------------------------------------------ #
        self.step_count += 1
        env_truncation = self.step_count >= self.max_cycles

        # Terminate when termination_goal fraction of USERS are uniquely covered
        # (the config value is a geometrically reachable threshold, e.g. 0.90).
        env_termination = true_coverage_efficiency >= self.termination_goal

        # Speed bonus: flat, identical for every agent, scaled by the fraction
        # of the step budget saved by solving early.
        if env_termination:
            speed_bonus = self.TERMINAL_SPEED_BONUS * (self.max_cycles - self.step_count) / self.max_cycles
            for agent_id in rewards:
                rewards[agent_id] += speed_bonus

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

    def _decode_vbs_action(self, action: int) -> Tuple[int, int]:
        """Flattened Discrete(branches * (slots+1)) -> (branch_id, slot_index).

        branch_id is 1-indexed; slot_index in [0, max_slot_per_branch].
        """
        slots_per_branch = int(self.max_slot_per_branch) + 1
        branch_id = action // slots_per_branch + 1
        slot_index = action % slots_per_branch
        return branch_id, slot_index

    def _apply_actions(self, actions: Dict[str, int]):
        # VBS actions are an absolute, factored (branch, slot) selection with
        # no dependency on the previous state (replacing the old relative
        # increment/decrement deltas, which had no absorbing target state).
        for agent_id, action in actions.items():
            agent_obj, is_vbs = self._get_agent_obj(agent_id)
            if is_vbs:
                branch_id, slot_index = self._decode_vbs_action(int(action))
                agent_obj.current_branch_id = branch_id
                agent_obj.current_slot_index = slot_index
            else:
                agent_obj.current_offset_zone = action

    def _calculate_world_coords(self, agent_obj, is_vbs) -> Tuple[float, float]:
        if is_vbs:
            if agent_obj.current_slot_index == 0:
                return self.graph_engine.get_edge_coordinates(0, 1, 0.0)
            else:
                traveled = agent_obj.current_slot_index / self.max_slot_per_branch
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
            x, y = hx + radius * np.cos(angle), hy + radius * np.sin(angle)
            # Clip physical coords, not just obs, so overshoot is visible to reward
            return float(np.clip(x, 0.0, self.map_dim[0])), float(np.clip(y, 0.0, self.map_dim[1]))

    def _compute_observations_and_masks(self) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        obs = {}
        infos = {}
        total_users = self.sim_adapter.num_users
        NUM_BRANCHES = 3

        # Observation locality: no global team state (branch_occupancy) or
        # global uncovered-user mean is broadcast into per-agent observations —
        # every directional cue below is computed from each agent's own (x, y)
        # and sensing_radius. The uncovered-user mask here only builds the
        # candidate pool each agent filters to its own sensing radius.

        if self.last_coverage_matrix is not None:
            any_covered_mask = np.any(self.last_coverage_matrix, axis=0)
            self.last_uncovered_grid = self.sim_adapter.compute_uncovered_density_grid(
                any_covered_mask, grid_size=self.uncovered_grid_size
            )
            uncovered_coords = self.sim_adapter.user_coords[~any_covered_mask]
        else:
            self.last_uncovered_grid = np.zeros(
                (self.uncovered_grid_size, self.uncovered_grid_size), dtype=np.float32
            )
            uncovered_coords = np.empty((0, 2), dtype=np.float32)

        # VBS EMA update pass, decoupled from iteration order for robustness
        # (don't rely on VBS happening to precede FBS in self.agents).
        for a in self.agents:
            obj, is_vbs = self._get_agent_obj(a)
            if is_vbs:
                vx, vy = self._calculate_world_coords(obj, True)
                obj.update_ema(vx, vy)

        for i, agent_id in enumerate(self.agents):
            agent_obj, is_vbs = self._get_agent_obj(agent_id)
            x, y = self._calculate_world_coords(agent_obj, is_vbs)

            norm_x = np.clip(x / self.map_dim[0], 0.0, 1.0)
            norm_y = np.clip(y / self.map_dim[1], 0.0, 1.0)

            if (self.last_coverage_matrix is not None
                    and i < len(self.last_coverage_matrix)):
                raw_coverage_frac = np.clip(
                    float(self.last_coverage_matrix[i].sum()) / float(max(total_users, 1)),
                    0.0, 1.0
                )
            else:
                raw_coverage_frac = 0.0

            # Local, bounded directional sensing cue for both agent types.
            sensing_radius = agent_obj.coverage_radius * self.sensing_radius_multiplier
            local_sector_fracs, uncovered_presence = self._local_sensing_features(
                x, y, sensing_radius, uncovered_coords
            )

            if is_vbs:
                agent_obj.update_ema(x, y)

                norm_slot = agent_obj.current_slot_index / self.max_slot_per_branch

                branch_hot = np.zeros(NUM_BRANCHES, dtype=np.float32)
                if agent_obj.current_slot_index > 0 and 1 <= agent_obj.current_branch_id <= NUM_BRANCHES:
                    branch_hot[agent_obj.current_branch_id - 1] = 1.0

                home_hot = np.zeros(NUM_BRANCHES, dtype=np.float32)
                if 1 <= agent_obj.home_branch_id <= NUM_BRANCHES:
                    home_hot[agent_obj.home_branch_id - 1] = 1.0

                identity_hot = np.zeros(self.n_vbs, dtype=np.float32)
                identity_hot[agent_obj.identity_index] = 1.0

                obs[agent_id] = np.concatenate([
                    np.array([norm_x, norm_y, raw_coverage_frac, norm_slot,
                              branch_hot[0], branch_hot[1], branch_hot[2],
                              home_hot[0], home_hot[1], home_hot[2]], dtype=np.float32),
                    local_sector_fracs,
                    np.array([uncovered_presence], dtype=np.float32),
                    identity_hot
                ])  # (10 + NUM_LOCAL_SECTOR_BINS + 1) + n_vbs dims

            else:
                # Polar decomposition of the FBS's own action, in the same
                # geometry the action space already uses.
                if agent_obj.current_offset_zone == 0:
                    r_frac, cos_t, sin_t = 0.0, 1.0, 0.0
                else:
                    dist_multiplier = 0.5 if agent_obj.current_offset_zone <= 8 else 1.0
                    angle_idx = (agent_obj.current_offset_zone - 1) % 8
                    angle = angle_idx * (np.pi / 4)
                    r_frac = dist_multiplier
                    cos_t, sin_t = float(np.cos(angle)), float(np.sin(angle))

                host_vbs = self.agent_manager.vbs_registry[agent_obj.host_vbs_id]
                host_branch_hot = np.zeros(NUM_BRANCHES, dtype=np.float32)
                if host_vbs.current_slot_index > 0 and 1 <= host_vbs.current_branch_id <= NUM_BRANCHES:
                    host_branch_hot[host_vbs.current_branch_id - 1] = 1.0

                ema_x_norm = np.clip((host_vbs.ema_x if host_vbs.ema_x is not None else x) / self.map_dim[0], 0.0, 1.0)
                ema_y_norm = np.clip((host_vbs.ema_y if host_vbs.ema_y is not None else y) / self.map_dim[1], 0.0, 1.0)

                # Un-smoothed, this-instant host position (closes the EMA lag).
                host_true_x, host_true_y = self._calculate_world_coords(host_vbs, True)
                host_true_x_norm = np.clip(host_true_x / self.map_dim[0], 0.0, 1.0)
                host_true_y_norm = np.clip(host_true_y / self.map_dim[1], 0.0, 1.0)

                identity_hot = np.zeros(self.n_fbs, dtype=np.float32)
                identity_hot[agent_obj.identity_index] = 1.0

                obs[agent_id] = np.concatenate([
                    np.array([norm_x, norm_y, raw_coverage_frac,
                              r_frac, cos_t, sin_t,
                              host_branch_hot[0], host_branch_hot[1], host_branch_hot[2],
                              ema_x_norm, ema_y_norm,
                              host_true_x_norm, host_true_y_norm], dtype=np.float32),
                    local_sector_fracs,
                    np.array([uncovered_presence], dtype=np.float32),
                    identity_hot
                ])  # (13 + NUM_LOCAL_SECTOR_BINS + 1) + n_fbs dims

            # Every absolute (branch, slot) action decodes to a valid in-range
            # state, so no VBS masking is needed; FBS actions are always valid.
            mask = np.ones(self.action_space(agent_id).n, dtype=np.int8) if not is_vbs \
                else self.graph_engine.get_action_mask(self.action_space(agent_id).n)

            infos[agent_id] = {"action_mask": mask}

        self._last_obs = obs
        return obs, infos

    def _local_sensing_features(
        self, x: float, y: float, sensing_radius: float, uncovered_coords: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """Directional sector histogram of uncovered users within sensing radius.

        Returns (sector_fracs, presence): sector_fracs is the fraction of
        locally detected uncovered users per 45° sector (0 = due east,
        counter-clockwise), summing to 1.0 iff anything is detected; presence
        is 1.0 iff any uncovered user is within sensing_radius.
        """
        n_bins = self.NUM_LOCAL_SECTOR_BINS
        if len(uncovered_coords) > 0:
            local_dists = np.linalg.norm(
                uncovered_coords - np.array([x, y], dtype=np.float32), axis=1
            )
            local_mask = local_dists <= sensing_radius
        else:
            local_mask = np.zeros(0, dtype=bool)

        if np.any(local_mask):
            deltas = uncovered_coords[local_mask] - np.array([x, y], dtype=np.float32)
            angles = np.arctan2(deltas[:, 1], deltas[:, 0])  # [-pi, pi], 0 = due east
            bin_idx = np.floor((angles + np.pi) / (2.0 * np.pi) * n_bins).astype(np.int64) % n_bins
            counts = np.bincount(bin_idx, minlength=n_bins).astype(np.float32)
            sector_fracs = counts / counts.sum()
            presence = 1.0
        else:
            sector_fracs = np.zeros(n_bins, dtype=np.float32)
            presence = 0.0
        return sector_fracs, presence

    def _get_raw_id(self, agent_string: str) -> int:
        return int(agent_string.split("_")[1])

    def _get_agent_obj(self, agent_string: str):
        agent_id = self._get_raw_id(agent_string)
        is_vbs = "vbs" in agent_string
        obj = (self.agent_manager.vbs_registry[agent_id]
               if is_vbs else self.agent_manager.fbs_registry[agent_id])
        return obj, is_vbs

    def get_global_state(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Keys off _last_obs (not self.agents, which is cleared post-terminal)
        # so main.py's truncation bootstrap still sees the real last snapshot.
        vbs_ids = [a for a in self._last_obs if "vbs" in a]
        fbs_ids = [a for a in self._last_obs if "fbs" in a]

        vbs_feats = np.stack([self._last_obs[a] for a in vbs_ids]) \
            if vbs_ids else np.zeros((1, self.vbs_fixed_obs_dim + self.n_vbs), dtype=np.float32)
        fbs_feats = np.stack([self._last_obs[a] for a in fbs_ids]) \
            if fbs_ids else np.zeros((1, self.fbs_fixed_obs_dim + self.n_fbs), dtype=np.float32)

        global_extra = np.concatenate([[self.last_true_coverage], self.last_uncovered_grid.flatten()])
        assert global_extra.shape[0] == self.global_extra_dim, "global_extra drifted from declared schema"
        return vbs_feats, fbs_feats, global_extra

    def get_fbs_host_vbs_indices(self) -> List[int]:
        """Relational topology cue for the CentralizedCritic (Task 4): for
        each FBS row in get_global_state()'s fbs_feats, the row index of that
        FBS's host VBS in the vbs_feats output (same _last_obs ordering)."""
        vbs_ids = [a for a in self._last_obs if "vbs" in a]
        fbs_ids = [a for a in self._last_obs if "fbs" in a]
        vbs_row_by_id = {self._get_raw_id(a): i for i, a in enumerate(vbs_ids)}
        return [
            vbs_row_by_id[self.agent_manager.fbs_registry[self._get_raw_id(a)].host_vbs_id]
            for a in fbs_ids
        ]

    def preview_vbs_world_coords(self, vbs_agent_id: str, action: int) -> Tuple[float, float]:
        """PURE preview — no state mutation. Uses the same decode +
        get_edge_coordinates path step() applies, so it can never drift from
        the actual physics. Lets the FBS observe its host's COMMITTED action
        for this step before choosing its own."""
        branch_id, slot_index = self._decode_vbs_action(int(action))
        if slot_index == 0:
            return self.graph_engine.get_edge_coordinates(0, 1, 0.0)
        traveled = slot_index / self.max_slot_per_branch
        return self.graph_engine.get_edge_coordinates(0, branch_id, traveled)

    def augment_fbs_obs(self, obs_row: np.ndarray, next_x: float, next_y: float) -> np.ndarray:
        """Overwrites the FBS's next-host-position slots (indices 9, 10 in the
        fbs obs layout) with the normalized PURE preview of the host VBS's
        NEXT position. The local sector / presence features live after index
        12, so they are untouched."""
        next_x_norm = np.clip(next_x / self.map_dim[0], 0.0, 1.0)
        next_y_norm = np.clip(next_y / self.map_dim[1], 0.0, 1.0)
        out = obs_row.copy()
        out[9] = next_x_norm
        out[10] = next_y_norm
        return out
