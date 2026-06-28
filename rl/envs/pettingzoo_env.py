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

        # FIXED: Read graph topology parameters from injected config dict instead of
        # hardcoding them as literals scattered across 3+ methods.
        self.center_node_id: int = config.get("center_node_id", 0)
        self.max_slots_per_branch: int = config.get("max_slots_per_branch", 10)

        # Pre-compute the canonical branch node ID list ONCE at init.
        # sorted() guarantees a deterministic, stable action→node mapping across runs:
        #   action index 0 → _branch_node_ids[0]
        #   action index 1 → _branch_node_ids[1]  ... etc.
        # This is the single source of truth consumed by action_space, _apply_actions,
        # _calculate_world_coords, and _compute_observations_and_masks.
        self._branch_node_ids: list = sorted(
            self.graph_engine.get_neighbors(self.center_node_id)
        )

        # 2. Strict PettingZoo Agent Tracking
        self.possible_agents = (
            [f"vbs_{v.id}" for v in self.agent_manager.vbs_registry.values()] +
            [f"fbs_{f.id}" for f in self.agent_manager.fbs_registry.values()]
        )
        self.agents = self.possible_agents[:]
        self.step_count = 0

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: str) -> spaces.Box:
        return spaces.Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: str) -> spaces.Discrete:
        if "vbs" in agent:
            # FIXED: Derive action count from live graph topology, not a literal.
            # Automatically updates when graph_map.json gains or loses branches.
            return spaces.Discrete(len(self._branch_node_ids))
        else:
            # FBS: 0=Hover + 8 half-dist + 8 full-dist = 17 total
            return spaces.Discrete(1 + 8 * 2)

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
        coverage_matrix = self.sim_adapter.compute_coverage_matrix(
            np_coords, np_radii
        )  # shape: (N, M) bool

        # Derive per-agent counts from the matrix for AgentManager state tracking
        coverage_counts = coverage_matrix.sum(axis=1, dtype=np.int32)
        for obj, count in zip(agent_mapping, coverage_counts):
            obj.current_coverage_count = int(count)

        # ------------------------------------------------------------------ #
        # PHASE 3: REVISED TEAM-ALIGNED REWARD ENGINEERING                  #
        # ------------------------------------------------------------------ #
        total_users = self.sim_adapter.num_users
        n_agents = len(self.agents)

        # Union of all agent coverages
        any_covered = np.any(coverage_matrix, axis=0)
        total_covered = int(any_covered.sum())

        # Pull global efficiency directly into the step reward calculation
        global_efficiency = float(self.agent_manager.get_total_efficiency())

        rewards = {}
        for i, agent_id in enumerate(self.agents):
            if n_agents > 1:
                # Counterfactual: mask excluding agent i
                others_mask = np.ones(n_agents, dtype=bool)
                others_mask[i] = False
                covered_without_i = int(
                    np.any(coverage_matrix[others_mask], axis=0).sum()
                )
            else:
                covered_without_i = 0

            # Marginal contribution component ∈ [0.0, 1.0]
            marginal_contribution = float(
                (total_covered - covered_without_i) / max(total_users, 1)
            )

            # --- COOPERATIVE BLEND REWARD ---
            # 40% Marginal Credit Assignment + 60% Global Objective Alignment
            # This eliminates reward hacking by heavily penalizing low global efficiency.
            rewards[agent_id] = (0.4 * marginal_contribution) + (0.6 * global_efficiency)

        # ------------------------------------------------------------------ #
        # PHASE 4: HORIZON & TERMINATION                                       #
        # ------------------------------------------------------------------ #
        self.step_count += 1
        env_truncation = self.step_count >= self.max_cycles
        env_termination = global_efficiency >= 0.95

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
                if agent_obj.current_slot_index == 0:  # At center node
                    # FIXED: Map action index → actual graph node ID via the canonical list.
                    # Old: agent_obj.current_branch_id = action + 1
                    #   → hardcoded offset; crashes with KeyError if branches aren't nodes {1,2,3}.
                    # New: generalizes to any graph topology (e.g., branches at {2,5,7}).
                    agent_obj.current_branch_id = self._branch_node_ids[action]
                    agent_obj.current_slot_index += 1
                else:
                    # Find which action index means "continue on the current branch"
                    current_branch_action_idx = self._branch_node_ids.index(
                        agent_obj.current_branch_id
                    )
                    if action == current_branch_action_idx:
                        # FIXED: cap at max_slots_per_branch from config, not hardcoded 10
                        agent_obj.current_slot_index = min(
                            self.max_slots_per_branch,
                            agent_obj.current_slot_index + 1
                        )
                    else:
                        agent_obj.current_slot_index = max(0, agent_obj.current_slot_index - 1)
            else:
                agent_obj.current_offset_zone = action


    def _calculate_world_coords(self, agent_obj, is_vbs) -> Tuple[float, float]:
        """Maps discrete indices into physical (x,y) space for the Simulator."""
        if is_vbs:
            if agent_obj.current_slot_index == 0:
                # FIXED: Read center node coordinates directly from the graph attribute dict.
                # Old: get_edge_coordinates(0, 1, 0.0)
                #   → hardcodes edge (0→1); raises KeyError if that edge was removed in the
                #   graph update (e.g., branch node IDs changed or center node moved).
                # New: direct node attribute lookup; immune to any edge-level topology change.
                node_data = self.graph_engine.graph.nodes[self.center_node_id]
                return float(node_data['x']), float(node_data['y'])
            else:
                # FIXED: divide by max_slots_per_branch from config, not hardcoded 10.0
                traveled = agent_obj.current_slot_index / float(self.max_slots_per_branch)
                return self.graph_engine.get_edge_coordinates(
                    self.center_node_id,
                    agent_obj.current_branch_id,
                    traveled
                )
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
        """Constructs aligned observation tensors and logit action masks."""
        obs = {}
        infos = {}
        for agent_id in self.agents:
            agent_obj, is_vbs = self._get_agent_obj(agent_id)
            x, y = self._calculate_world_coords(agent_obj, is_vbs)

            norm_x = np.clip(x / self.map_dim[0], 0.0, 1.0)
            norm_y = np.clip(y / self.map_dim[1], 0.0, 1.0)
            cap_ratio = (
                min(agent_obj.current_coverage_count, agent_obj.capacity) / agent_obj.capacity
                if agent_obj.capacity > 0 else 0.0
            )
            obs[agent_id] = np.array([norm_x, norm_y, cap_ratio], dtype=np.float32)

            mask = np.ones(self.action_space(agent_id).n, dtype=np.int8)
            if is_vbs:
                # FIXED: use max_slots_per_branch from config, not hardcoded 10.
                # FIXED: compute action index via list lookup, not arithmetic offset.
                # Old: mask[agent_obj.current_branch_id - 1] = 0
                #   → assumes branch IDs are {1,2,3}; silently masks the wrong action
                #   on any other graph (e.g., branches at {2,5,7} gives mask[-1+2=1] wrong).
                if (agent_obj.current_slot_index >= self.max_slots_per_branch
                        and agent_obj.current_branch_id in self._branch_node_ids):
                    action_idx = self._branch_node_ids.index(agent_obj.current_branch_id)
                    mask[action_idx] = 0

            infos[agent_id] = {"action_mask": mask}

        return obs, infos

    def _get_raw_id(self, agent_string: str) -> int:
        return int(agent_string.split("_")[1])

    def _get_agent_obj(self, agent_string: str):
        agent_id = self._get_raw_id(agent_string)
        is_vbs = "vbs" in agent_string
        obj = self.agent_manager.vbs_registry[agent_id] if is_vbs else self.agent_manager.fbs_registry[agent_id]
        return obj, is_vbs