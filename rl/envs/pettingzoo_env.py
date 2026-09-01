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

    # Branches incident to the center node. Hardcoded per current task scope —
    # graph-topology generalization (variable branch counts) is out of scope.
    # TODO(scope): derive this from graph_engine once variable-topology support
    # is tasked; currently duplicated as a literal `NUM_BRANCHES = 3` in
    # _compute_observations_and_masks for observation/mask feature widths.
    NUM_VBS_BRANCHES = 3

    # FIXED: sensing_radius used to be an implicit, unbounded global broadcast —
    # every agent's dx/dy gradient cue was derived from the mean of ALL uncovered
    # users on the map, regardless of distance. That's both an observation-locality
    # violation (no physical sensor gives an agent that global mean) and a bad
    # reward-shaping heuristic (a single shared attractor point pulls every agent
    # toward the same location, fighting the marginal-contribution reward's actual
    # spread-out objective). Each agent's local sensing radius is now bounded and
    # scaled off its OWN coverage_radius (a physically plausible sensor-range
    # relationship: bigger radio footprint == bigger detection footprint),
    # configurable via the "sensing_radius_multiplier" config key so this isn't a
    # silent magic number.
    DEFAULT_SENSING_RADIUS_MULTIPLIER = 2.5
    # FIXED: no incentive to solve fast vs. linger. One-time bonus on the
    # terminating step, scaled by steps saved vs max_cycles.
    TERMINAL_SPEED_BONUS = 5.0

    # FIXED (Task 3): the local sensing cue used to be a single vector mean
    # (dx, dy) of uncovered users within sensing radius. When an agent sat
    # between two symmetric clusters of uncovered users the vector sum
    # cancelled to (0, 0), falsely signalling "target is directly underneath
    # me" (the local sensing dead-zone). Replaced with a directional sector
    # histogram: 8 radial bins (N, NE, E, ...) around the agent, each holding
    # the fraction of locally-detected uncovered users in that angular sector,
    # plus a presence bit. Symmetric clusters now produce two distinct peaks
    # instead of a degenerate zero vector.
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
        # FIXED (Task 5): overlap penalty is now a STATIC hyperparameter loaded
        # from config. It used to ramp in as
        #   0.20 * min(episode_count / warmup_episodes, 1.0)
        # which mutated the transition dynamics mid-training (violating MDP
        # stationarity — the reward function silently changed between episodes).
        # Any curriculum over the penalty must be driven externally between
        # rollouts, never inside env.step().
        self.overlap_penalty_weight = float(config.get("overlap_penalty_weight", 0.20))

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

        # FIXED (Task 1): RunningNorm was removed from the bounded reward metrics
        # (true_coverage_efficiency, marginal_contribution). Both are native
        # [0.0, 1.0] quantities, so they are now passed through a fixed,
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

        # Per-type fixed feature widths. Keep these in lock-step with the
        # concatenation order in _compute_observations_and_masks and the
        # observation_space() shapes below.
        self.vbs_fixed_obs_dim = 10 + self.NUM_LOCAL_SECTOR_BINS + 1
        self.fbs_fixed_obs_dim = 13 + self.NUM_LOCAL_SECTOR_BINS + 1

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: str) -> spaces.Box:
        if "vbs" in agent:
            # FIXED: branch_occupancy(3) (global broadcast across ALL VBS, no VBS has
            # a sensor for this) removed per observation-locality fix; the shared
            # global uncovered_centroid_dx_dy(2) is replaced with a per-agent,
            # sensing_radius-bounded directional sector histogram.
            # [norm_x, norm_y, coverage_frac, norm_slot, branch_hot(3),
            #  home_branch_hot(3), local_sector_fracs(8), local_uncovered_presence(1)]
            return spaces.Box(low=-1.0, high=1.0, shape=(self.vbs_fixed_obs_dim + self.n_vbs,), dtype=np.float32)
        else:
            # [norm_x, norm_y, coverage_frac, r_frac, cos_t, sin_t,
            #  host_branch_hot(3), ema_xy_norm(2), host_true_xy_norm(2),
            #  local_sector_fracs(8), local_uncovered_presence(1)]
            return spaces.Box(low=-1.0, high=1.0, shape=(self.fbs_fixed_obs_dim + self.n_fbs,), dtype=np.float32)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: str) -> spaces.Discrete:
        if "vbs" in agent:
            # Absolute, factored (branch, slot) selection flattened into a single
            # Discrete head — see BUG LEDGER in _apply_actions. slots_per_branch
            # includes slot 0 (center), hence +1.
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
        # FIXED (Task 1): all three components are now STATIONARY. true_coverage
        # and marginal_contribution are native [0.0, 1.0] metrics, so they are
        # passed through fixed StationaryScaler()s — no running mean/std that
        # could chase the policy to a moving goalpost. Previously, once coverage
        # converged to ~0.95 the running mean tracked to 0.95 and normalize()
        # returned ~0.0, erasing both positive terms exactly as the team neared
        # the goal; only the (negative) overlap penalty survived, collapsing the
        # episodic reward to -160 while coverage sat at 98%.
        #
        # Reward range at extreme states (400 users, 10 agents, radius-45 FBS):
        #   All bunched at center: ~(-0.73) per step  ← negative signal to spread
        #   Optimal spread (~40% unique): ~(1.04) per step  ← positive convergence target
        #
        # Tune these three weights as hyperparameters if the graph layout changes:
        REWARD_SCALE = 1.0          # Scales reward to [≈-7.3, ≈10.4] — stable for PPO clip=0.2
        MARGINAL_WEIGHT = 0.65        # Individual Shapley-value approximation
        TEAM_WEIGHT = 0.15            # Shared cooperative gradient
        # FIXED (Task 5): overlap penalty weight is a STATIC hyperparameter
        # (self.overlap_penalty_weight, loaded from config). The old
        # 0.20 * min(episode/warmup, 1.0) ramp mutated the reward function
        # between episodes, violating MDP stationarity.

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
            # Both positive terms are raw, stationary-scaled [0,1] values, so the
            # reward stays positive and monotonic in coverage at every episode.
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

        # FIX: terminate when `termination_goal` fraction of USERS are uniquely
        # covered, not when stations are full. termination_goal is set in
        # config/simulation_config.json to a GEOMETRICALLY REACHABLE threshold
        # (0.90) for the given fixed agent positions/radii — 1.0 was impossible
        # for arbitrary uniform user distributions, so episodes never terminated
        # early and TERMINAL_SPEED_BONUS never fired (see Task 2).
        env_termination = true_coverage_efficiency >= self.termination_goal

        # FIXED: reward positive shaped reward per step, so nothing pushed
        # the policy to finish early vs. hover near the goal. Flat bonus,
        # same for every agent, scaled by fraction of budget saved.
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

        branch_id is 1-indexed (matches agent_obj.current_branch_id convention).
        slot_index in [0, max_slot_per_branch].
        """
        slots_per_branch = int(self.max_slot_per_branch) + 1
        branch_id = action // slots_per_branch + 1
        slot_index = action % slots_per_branch
        return branch_id, slot_index

    def _apply_actions(self, actions: Dict[str, int]):
        # BUG LEDGER — FIXED: VBS actions used to be relative deltas: a chosen
        # action only incremented current_slot_index if it matched
        # current_branch_id, and decremented it otherwise. Reaching any target
        # position therefore required a consistent multi-step action sequence.
        # Because the marginal-contribution reward is recomputed every step as
        # other agents move, the locally-perceived advantage sign could flip
        # mid-transit, producing a stable 2-state limit cycle with no absorbing
        # target state. VBS actions are now an absolute, factored (branch, slot)
        # selection — flattened into a single Discrete head and decoded via
        # _decode_vbs_action — mirroring the FBS 17-point discretized
        # offset-zone fix. current_branch_id/current_slot_index are assigned
        # directly from the decode with no dependency on the previous state.
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
            # FIXED: only obs was clipped before, not physical coords -> overshoot invisible to reward
            return float(np.clip(x, 0.0, self.map_dim[0])), float(np.clip(y, 0.0, self.map_dim[1]))

    def _compute_observations_and_masks(self) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        obs = {}
        infos = {}
        total_users = self.sim_adapter.num_users
        NUM_BRANCHES = 3

        # BUG LEDGER — FIXED (observation-locality violation, 2 instances):
        #
        # 1. branch_occupancy used to be a global fraction ("what % of ALL VBS sit
        #    on branch k") computed once and broadcast identically into every VBS's
        #    local observation. No individual VBS has a sensor for this — it's
        #    privileged, critic-only information. Removed from the per-agent actor
        #    observation entirely (option (a) from the task): the CentralizedCritic
        #    already has implicit access to every agent's state via its pooled
        #    vbs_feats/fbs_feats input, so branch-selection coordination is learned
        #    through the critic's baseline rather than leaked into the actor's local
        #    obs. Preferred over option (b) (a locally-sensed "is MY branch occupied"
        #    bit) because that would still require an ungrounded short-range-sensing
        #    assumption for a feature the critic already sees for free — removing it
        #    is strictly simpler and doesn't touch the critic (out of scope, Task 3).
        #
        # 2. uncovered_centroid used to be the mean of ALL uncovered users on the
        #    map, computed once and broadcast into every agent's dx/dy. Beyond being
        #    a locality violation, it was also a degenerate reward-shaping heuristic:
        #    a single global mean is one shared attractor point that pulls every
        #    agent toward the same location, fighting the marginal-contribution
        #    reward's actual spread-out/minimize-overlap objective. First replaced
        #    with a per-agent, sensing_radius-bounded local centroid (computed
        #    per-agent below, from each agent's own (x, y) — a legitimate local
        #    computation) plus a presence bit, and then (Task 3) upgraded to a
        #    directional sector histogram (_local_sensing_features) that eliminates
        #    the vector-mean dead-zone where symmetric clusters cancelled to (0, 0).

        # Uncovered-user mask — still a one-time global computation, but it is only
        # used below to build a candidate pool that each agent then filters down to
        # its own sensing_radius. The filtering step (not this precursor) is what
        # makes the resulting sector/presence feature a legitimate per-agent quantity.
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

        # --- VBS EMA update pass, decoupled from iteration order for robustness ---
        # Explicit pre-pass rather than relying on "VBS happen to precede FBS in
        # self.agents" — that ordering holds today (possible_agents concatenates
        # VBS then FBS) but shouldn't be a silent dependency of correctness.
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

            # Local, bounded directional sensing cue — identical derivation for both
            # agent types, but now computed from THIS agent's own (x, y) and its own
            # sensing_radius (scaled off its own coverage_radius), not a global mean.
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
                # Polar decomposition of the FBS's own action, expressed in the same
                # geometry the action space already uses — no implicit cartesian→polar
                # re-derivation left for the network to learn.
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

                # NEW: un-smoothed, this-instant host position — closes the ~10-step EMA lag
                # the FBS previously had to perceive its own anchor through.
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

            # FIXED: the old relative VBS action scheme masked the "advance" action
            # on the current branch once current_slot_index hit the max, to prevent
            # overshoot past the branch end. Under the absolute (branch, slot)
            # selection every action decodes to a valid, in-range absolute state
            # (see _decode_vbs_action), so overshoot is structurally impossible and
            # no VBS masking is needed here anymore.
            # Delegates to GraphEngineABC contract now implemented by NetworkXRoadEngine.
            mask = np.ones(self.action_space(agent_id).n, dtype=np.int8) if not is_vbs \
                else self.graph_engine.get_action_mask(self.action_space(agent_id).n)

            infos[agent_id] = {"action_mask": mask}

        self._last_obs = obs
        return obs, infos

    def _local_sensing_features(
        self, x: float, y: float, sensing_radius: float, uncovered_coords: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """Directional sector histogram of uncovered users within sensing radius.

        FIXED (Task 3 — local sensing dead-zone): the previous cue was the mean
        (dx, dy) of all uncovered users within sensing radius. An agent placed
        between two symmetric clusters saw the vectors cancel to (dx=0, dy=0),
        falsely indicating the target sat directly underneath it. The sector
        histogram instead bins locally-detected uncovered users into
        NUM_LOCAL_SECTOR_BINS angular sectors around the agent, so symmetric
        clusters produce two distinct non-zero peaks.

        Returns:
            sector_fracs: (NUM_LOCAL_SECTOR_BINS,) float32 — fraction of locally
                detected uncovered users per 45° sector (0 = due east, increasing
                counter-clockwise). Sums to 1.0 iff at least one user is detected,
                else all-zeros.
            presence: 1.0 if any uncovered user is within sensing_radius else 0.0.
                Distinguishes "nothing detected" (all-zero sectors, presence=0)
                from "uncovered users at/near my position" (presence=1).
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
        # FIXED: filtering by self.agents broke post-terminal-step callers
        # (main.py's truncation bootstrap) since step() clears self.agents
        # before returning. _last_obs.keys() reflects the real last snapshot.
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
        """Relational topology cue for the CentralizedCritic (Task 4).

        Returns, for each FBS row in get_global_state()'s fbs_feats output, the
        row index of that FBS's host VBS in the vbs_feats output. Row ordering is
        identical because both get_global_state() and this helper key off
        _last_obs (vbs first, then fbs, both in ascending agent-id order).

        This is the information the old symmetric mean/max Deep-Sets pooling
        destroyed: which FBS is tethered to which host VBS. Feeding it to the
        critic lets each FBS's value be conditioned on its own host's encoding
        before any permutation-invariant pooling over the (VBS, FBS-group) pairs.
        """
        vbs_ids = [a for a in self._last_obs if "vbs" in a]
        fbs_ids = [a for a in self._last_obs if "fbs" in a]
        vbs_row_by_id = {self._get_raw_id(a): i for i, a in enumerate(vbs_ids)}
        return [
            vbs_row_by_id[self.agent_manager.fbs_registry[self._get_raw_id(a)].host_vbs_id]
            for a in fbs_ids
        ]

    def preview_vbs_world_coords(self, vbs_agent_id: str, action: int) -> Tuple[float, float]:
        """PURE preview — no state mutation. Reuses _decode_vbs_action +
        get_edge_coordinates, the same path _apply_actions/_calculate_world_coords
        use, so the preview can never drift from the physics actually applied in
        Phase 1 of step(). Used by the rollout loop (main.py/inference.py) to let
        FBS observe its host's COMMITTED action for this step before choosing its
        own — see BUG LEDGER: FBS action-vs-observation causality gap (FBS used to
        choose its action from a VBS position that was one full step stale)."""
        branch_id, slot_index = self._decode_vbs_action(int(action))
        if slot_index == 0:
            return self.graph_engine.get_edge_coordinates(0, 1, 0.0)
        traveled = slot_index / self.max_slot_per_branch
        return self.graph_engine.get_edge_coordinates(0, branch_id, traveled)

    def augment_fbs_obs(self, obs_row: np.ndarray, next_x: float, next_y: float) -> np.ndarray:
        """Replaces the FBS's ema_x_norm/ema_y_norm slots (indices 9,10 in the
        fbs_fixed_obs_dim + n_fbs layout — see observation_space()) with the
        normalized, PURE preview of the host VBS's NEXT position (this step's
        committed action, not last step's realized position). FIXED: closes the
        causality gap — EMA's lagged-trend info is now strictly dominated by
        exact current (host_true_x/y_norm) + exact next (this) position, so EMA
        is dropped rather than appended, keeping obs width unchanged. The local
        sector / presence features live after index 12, so they are untouched."""
        next_x_norm = np.clip(next_x / self.map_dim[0], 0.0, 1.0)
        next_y_norm = np.clip(next_y / self.map_dim[1], 0.0, 1.0)
        out = obs_row.copy()
        out[9] = next_x_norm  # was ema_x_norm
        out[10] = next_y_norm  # was ema_y_norm
        return out
