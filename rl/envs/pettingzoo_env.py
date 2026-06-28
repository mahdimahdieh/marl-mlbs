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
        # 1. Dependency Injection of our High-Performance Adapters
        self.agent_manager: AgentManager = config["agent_manager"]
        self.graph_engine: NetworkXRoadEngine = config["graph_engine"]
        self.sim_adapter: PyWiSimAdapter = config["sim_adapter"]

        self.max_cycles = config.get("max_cycles", 100)
        self.map_dim = self.graph_engine.get_map_dimension()

        # 2. Strict PettingZoo Agent Tracking
        self.possible_agents = [f"vbs_{v.id}" for v in self.agent_manager.vbs_registry.values()] + \
                               [f"fbs_{f.id}" for f in self.agent_manager.fbs_registry.values()]
        self.agents = self.possible_agents[:]

        self.step_count = 0

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: str) -> spaces.Box:
        # STRICT RL REQUIREMENT: Bound observations to [0.0, 1.0] to prevent gradient explosion.
        # Format: [norm_x, norm_y, norm_capacity_ratio]
        return spaces.Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: str) -> spaces.Discrete:
        if "vbs" in agent:
            # 0: Branch 0, 1: Branch 1, 2: Branch 2 (or move towards center)
            return spaces.Discrete(3)
        else:
            # 0: Hover, 1-8: Half-Distance (Polar), 9-16: Full-Distance (Polar)
            return spaces.Discrete(17)

    def reset(self, seed: int = None, options: Dict = None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        self.agents = self.possible_agents[:]
        self.step_count = 0

        # 1. Reset Core Logic
        self.agent_manager.reset_all_agents()
        self.sim_adapter.reset_spatial_distribution(seed=seed)

        # 2. Generate Initial Tensors
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
        # Build aligned arrays; index i corresponds to self.agents[i]         #
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

        np_coords = np.array(agent_coords, dtype=np.float32)  # (N, 2)
        np_radii = np.array(coverage_radii, dtype=np.float32)  # (N,)

        # Request the full (N_agents, N_users) boolean matrix.
        # DO NOT collapse to counts here — Phase 3 needs user-level resolution.
        coverage_matrix = self.sim_adapter.compute_coverage_matrix(
            np_coords, np_radii
        )  # shape: (N, M) bool

        # Derive per-agent counts from the matrix for AgentManager state tracking
        coverage_counts = coverage_matrix.sum(axis=1, dtype=np.int32)
        for obj, count in zip(agent_mapping, coverage_counts):
            obj.current_coverage_count = int(count)

        # ------------------------------------------------------------------ #
        # PHASE 3: DIFFERENTIAL REWARD ENGINEERING                            #
        #                                                                      #
        # True marginal contribution:                                          #
        #   reward_i = P(user covered | ALL agents)                           #
        #            - P(user covered | all agents EXCEPT i)                  #
        #                                                                      #
        # This requires the (N, M) matrix — aggregate counts are insufficient. #
        # ------------------------------------------------------------------ #
        total_users = self.sim_adapter.num_users
        n_agents = len(self.agents)

        # Union of all agent coverages — which users are covered at all?
        # any_covered shape: (M,) bool
        any_covered = np.any(coverage_matrix, axis=0)
        total_covered = int(any_covered.sum())

        rewards = {}
        for i, agent_id in enumerate(self.agents):
            if n_agents > 1:
                # Counterfactual: boolean mask excluding agent i
                others_mask = np.ones(n_agents, dtype=bool)
                others_mask[i] = False
                # Which users would still be covered without agent i?
                covered_without_i = int(
                    np.any(coverage_matrix[others_mask], axis=0).sum()
                )
            else:
                # Solo agent: full credit for all coverage it provides
                covered_without_i = 0

            # Marginal contribution ∈ [0.0, 1.0]
            # Numerically stable: no NaN risk since total_users > 0 is enforced
            rewards[agent_id] = float(
                (total_covered - covered_without_i) / max(total_users, 1)
            )

        # ------------------------------------------------------------------ #
        # PHASE 4: HORIZON & TERMINATION                                       #
        # ------------------------------------------------------------------ #
        self.step_count += 1
        env_truncation = self.step_count >= self.max_cycles
        env_termination = self.agent_manager.get_total_efficiency() >= 0.95

        terminations = {agent: env_termination for agent in self.agents}
        truncations = {agent: env_truncation for agent in self.agents}

        # ------------------------------------------------------------------ #
        # PHASE 5: OBSERVATIONS & MASKS                                        #
        # ------------------------------------------------------------------ #
        obs, infos = self._compute_observations_and_masks()

        if env_termination or env_truncation:
            self.agents = []

        return obs, rewards, terminations, truncations, infos

    # --- INTERNAL MECHANICS ---

    def _apply_actions(self, actions: Dict[str, int]):
        """Translates discrete neural network outputs into topological/polar states."""
        for agent_id, action in actions.items():
            agent_obj, is_vbs = self._get_agent_obj(agent_id)
            if is_vbs:
                # Graph navigation logic (Simplified for MVP bounds)
                if agent_obj.current_slot_index == 0:  # At center node 0
                    agent_obj.current_branch_id = action + 1  # Branches 1, 2, 3
                    agent_obj.current_slot_index += 1
                else:
                    if action == agent_obj.current_branch_id - 1:
                        agent_obj.current_slot_index = min(10, agent_obj.current_slot_index + 1)  # Deeper
                    else:
                        agent_obj.current_slot_index = max(0, agent_obj.current_slot_index - 1)  # Back to center
            else:
                # Polar navigation logic
                agent_obj.current_offset_zone = action

    def _calculate_world_coords(self, agent_obj, is_vbs) -> Tuple[float, float]:
        """Maps discrete indices into physical (x,y) space for the Simulator."""
        if is_vbs:
            if agent_obj.current_slot_index == 0:
                return self.graph_engine.get_edge_coordinates(0, 1, 0.0)  # Center
            else:
                # Interpolate traveled distance (assuming 10 slots per branch)
                traveled = agent_obj.current_slot_index / 10.0
                return self.graph_engine.get_edge_coordinates(0, agent_obj.current_branch_id, traveled)
        else:
            host_vbs = self.agent_manager.vbs_registry[agent_obj.host_vbs_id]
            hx, hy = self._calculate_world_coords(host_vbs, True)
            if agent_obj.current_offset_zone == 0:
                return hx, hy

            # Translate action 1-16 into polar offsets
            dist_multiplier = 0.5 if agent_obj.current_offset_zone <= 8 else 1.0
            angle_idx = (agent_obj.current_offset_zone - 1) % 8
            angle = angle_idx * (np.pi / 4)  # 45 degrees step

            radius = agent_obj.coverage_radius * dist_multiplier  # Max tether distance
            return hx + radius * np.cos(angle), hy + radius * np.sin(angle)

    def _compute_observations_and_masks(self) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Constructs aligned observation tensors and logit action masks."""
        obs = {}
        infos = {}
        for agent_id in self.agents:
            agent_obj, is_vbs = self._get_agent_obj(agent_id)
            x, y = self._calculate_world_coords(agent_obj, is_vbs)

            # STRICT FIX: Normalize observations to [0, 1] using map dimensions
            norm_x = np.clip(x / self.map_dim[0], 0.0, 1.0)
            norm_y = np.clip(y / self.map_dim[1], 0.0, 1.0)
            cap_ratio = min(agent_obj.current_coverage_count,
                            agent_obj.capacity) / agent_obj.capacity if agent_obj.capacity > 0 else 0.0

            obs[agent_id] = np.array([norm_x, norm_y, cap_ratio], dtype=np.float32)

            # Inject structural Action Masking for PPO directly into the `info` dict
            mask = np.ones(self.action_space(agent_id).n, dtype=np.int8)
            if is_vbs:
                # Graph boundary masking: If at slot 10 (edge tip), mask out moving deeper
                if agent_obj.current_slot_index >= 10:
                    mask[agent_obj.current_branch_id - 1] = 0

            infos[agent_id] = {"action_mask": mask}

        return obs, infos

    def _get_raw_id(self, agent_string: str) -> int:
        return int(agent_string.split("_")[1])

    def _get_agent_obj(self, agent_string: str):
        agent_id = self._get_raw_id(agent_string)
        is_vbs = "vbs" in agent_string
        obj = self.agent_manager.vbs_registry[agent_id] if is_vbs else self.agent_manager.fbs_registry[agent_id]
        return obj, is_vbs