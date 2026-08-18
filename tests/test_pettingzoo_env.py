"""
Canonical CoverageParallelEnv regression suite — the structural regression
gate for the bugs fixed in Tasks 1-3:

  Task 1 (VBS action semantics): action_space/_apply_actions used to be a
  relative accumulator (increment/decrement current_slot_index depending on
  whether the chosen action matched current_branch_id). Fixed to an absolute,
  factored (branch, slot) selection.

  Task 2 (observation locality): uncovered_centroid used to be a single
  global mean broadcast into every agent's dx/dy (a degenerate shared
  attractor); branch_occupancy leaked global team state into the per-agent
  actor observation. Fixed to a per-agent, sensing_radius-bounded local
  centroid + presence bit, with branch_occupancy removed.

  Task 3 (critic granularity): out of scope for this file — see
  tests/test_critic_agent_granularity.py.

This file specifically checks that the env's DECLARED spaces
(observation_space()/action_space()) match what it ACTUALLY produces at
runtime — the exact kind of silent shape drift that would otherwise only
surface as a crash deep inside main.py/inference.py's PPO forward pass.
"""
import os
import numpy as np
import pytest

from core.entities.agents import AgentManager, VehicleBaseStation, FlyingBaseStation
from infrastructure.graph.networkx_engine import NetworkXRoadEngine
from infrastructure.simulation.pywisim_adapter import PyWiSimAdapter
from rl.envs.pettingzoo_env import CoverageParallelEnv

GRAPH_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "graph_map.json")


def make_env(num_vbs=2, num_fbs=1, sensing_radius_multiplier=2.5, coverage_radius=15.0,
             max_slot_per_branch=10, num_users=20):
    graph_engine = NetworkXRoadEngine()
    graph_engine.load_from_json(GRAPH_PATH)

    manager = AgentManager()
    for i in range(num_vbs):
        manager.register_vbs(VehicleBaseStation(id=i, capacity=10, coverage_radius=coverage_radius))
    manager.assign_home_branches(num_branches=3)

    for j in range(num_fbs):
        manager.register_fbs(FlyingBaseStation(
            id=num_vbs + j, host_vbs_id=j % max(num_vbs, 1),
            capacity=10, coverage_radius=coverage_radius + 5.0, maximum_distance=15.0,
        ))
    manager.assign_identity_indices()

    sim_adapter = PyWiSimAdapter(num_users=num_users, map_dimensions=graph_engine.get_map_dimension())

    config = {
        "agent_manager": manager,
        "graph_engine": graph_engine,
        "sim_adapter": sim_adapter,
        "max_cycles": 20,
        "termination_goal": 0.999,
        "max_slot_per_branch": max_slot_per_branch,
        "sensing_radius_multiplier": sensing_radius_multiplier,
    }
    env = CoverageParallelEnv(config)
    return env, manager, sim_adapter


# --------------------------------------------------------------------------- #
# 1a. Declared spaces vs. actually-produced arrays (shape-drift regression)   #
# --------------------------------------------------------------------------- #

def test_vbs_action_space_matches_actual_mask_shape():
    env, _, _ = make_env(num_vbs=2)
    obs_dict, infos_dict = env.reset(seed=0)

    declared_n = env.action_space("vbs_0").n
    assert declared_n == 3 * (10 + 1)  # NUM_VBS_BRANCHES * (max_slot_per_branch + 1)

    actual_mask = infos_dict["vbs_0"]["action_mask"]
    assert actual_mask.shape == (declared_n,), (
        "action_space().n has drifted from the actual action_mask width returned "
        "by _compute_observations_and_masks — this is exactly the kind of drift "
        "that crashes main.py/inference.py's PPO forward pass silently."
    )


def test_fbs_action_space_matches_actual_mask_shape():
    env, _, _ = make_env(num_vbs=1, num_fbs=1)
    obs_dict, infos_dict = env.reset(seed=0)

    declared_n = env.action_space("fbs_1").n
    assert declared_n == 17
    actual_mask = infos_dict["fbs_1"]["action_mask"]
    assert actual_mask.shape == (declared_n,)


def test_vbs_observation_space_matches_actual_obs_shape():
    env, _, _ = make_env(num_vbs=3)
    obs_dict, _ = env.reset(seed=0)

    declared_shape = env.observation_space("vbs_0").shape
    assert declared_shape == (13 + env.n_vbs,)
    assert obs_dict["vbs_0"].shape == declared_shape


def test_fbs_observation_space_matches_actual_obs_shape():
    env, _, _ = make_env(num_vbs=2, num_fbs=2)
    obs_dict, _ = env.reset(seed=0)

    declared_shape = env.observation_space("fbs_2").shape
    assert declared_shape == (16 + env.n_fbs,)
    assert obs_dict["fbs_2"].shape == declared_shape


def test_global_state_feature_widths_match_declared_obs_dims():
    """get_global_state()'s stacked rows must have the same per-agent width as
    observation_space() declares — this is the exact field that silently
    drifted (the "15 + n" -> "13 + n" / "16 + n" stale fallback bug) during
    Task 2 and was caught only by manual review, not a test."""
    env, _, _ = make_env(num_vbs=2, num_fbs=1)
    env.reset(seed=0)
    vbs_feats, fbs_feats, global_extra = env.get_global_state()

    assert vbs_feats.shape[1] == env.observation_space("vbs_0").shape[0]
    assert fbs_feats.shape[1] == env.observation_space("fbs_2").shape[0]
    assert global_extra.shape[0] == env.global_extra_dim


# --------------------------------------------------------------------------- #
# 1b. Task 1 regression: absolute action landing, no multi-step drift         #
# --------------------------------------------------------------------------- #

def test_vbs_absolute_action_lands_exactly_regardless_of_prior_state():
    """The original bug: a chosen action only advanced current_slot_index if it
    matched current_branch_id, else it retreated by one slot -- multi-step
    drift, with no absorbing target state. Post-fix, EVERY single action call
    must land EXACTLY on its decoded (branch, slot) target in one step, with
    zero dependency on whatever state the agent was previously in."""
    env, manager, _ = make_env(num_vbs=1)
    env.reset(seed=0)
    vbs = manager.vbs_registry[0]

    # slots_per_branch = 11 (0..10). action = (branch-1)*11 + slot.
    sequence = [
        (0, (1, 0)),      # branch 1, slot 0 (home)
        (32, (3, 10)),    # branch 3, slot 10 (opposite corner) -- large jump
        (11, (2, 0)),     # branch 2, slot 0 -- jump again, no residual drift
        (5, (1, 5)),      # branch 1, slot 5 -- back to branch 1 mid-slot
        (21, (2, 10)),    # branch 2, slot 10
    ]
    for action, (expected_branch, expected_slot) in sequence:
        env._apply_actions({"vbs_0": action})
        assert vbs.current_branch_id == expected_branch, (
            f"action {action}: expected branch {expected_branch}, got {vbs.current_branch_id} "
            "-- looks like a reintroduced relative/incremental dependency on prior state"
        )
        assert vbs.current_slot_index == expected_slot, (
            f"action {action}: expected slot {expected_slot}, got {vbs.current_slot_index} "
            "-- looks like a reintroduced relative/incremental dependency on prior state"
        )


def test_vbs_repeated_identical_action_is_idempotent():
    """A structural signature check for the old bug: repeatedly issuing the
    SAME action must always land on the SAME state (never oscillate), since
    the decode is now a pure function of the action alone."""
    env, manager, _ = make_env(num_vbs=1)
    env.reset(seed=0)
    vbs = manager.vbs_registry[0]

    for _ in range(5):
        env._apply_actions({"vbs_0": 17})  # branch 2, slot 6
        assert (vbs.current_branch_id, vbs.current_slot_index) == (2, 6)


# --------------------------------------------------------------------------- #
# 1c. Task 2 regression: local sensing zero-vector / agent-specific signal    #
# --------------------------------------------------------------------------- #

def _extract_local_signal(obs_row, n_vbs):
    """VBS obs layout: [..., dx, dy, presence, identity_hot(n_vbs)] — see
    observation_space()'s documented feature order."""
    dx = obs_row[-3 - n_vbs]
    dy = obs_row[-2 - n_vbs]
    presence = obs_row[-1 - n_vbs]
    return float(dx), float(dy), float(presence)


def test_local_sensing_returns_zero_vector_and_zero_presence_when_nothing_in_range():
    env, manager, sim_adapter = make_env(num_vbs=1, coverage_radius=2.0, sensing_radius_multiplier=1.0)
    env.reset(seed=0)
    vbs0 = manager.vbs_registry[0]
    env._apply_actions({"vbs_0": 0})
    x0, y0 = env._calculate_world_coords(vbs0, True)

    # Every "uncovered" user is placed far outside vbs_0's sensing radius.
    sim_adapter.user_coords[:] = [
        min(x0 + 1000.0, env.map_dim[0]),
        min(y0 + 1000.0, env.map_dim[1]),
    ]
    env.last_coverage_matrix = np.zeros((len(env.agents), sim_adapter.num_users), dtype=bool)

    obs, _ = env._compute_observations_and_masks()
    dx, dy, presence = _extract_local_signal(obs["vbs_0"], env.n_vbs)

    assert presence == 0.0
    assert dx == 0.0
    assert dy == 0.0


def test_local_sensing_nonzero_and_agent_specific_with_asymmetric_ground_truth():
    """Direct regression test for the global-centroid mode-collapse bug: place
    a single uncovered user near vbs_0 ONLY. vbs_0 must detect it (presence=1,
    nonzero dx/dy); vbs_1, far away with nothing nearby, must NOT (presence=0,
    dx=dy=0). If a future regression reintroduces a global mean, both agents
    would receive the SAME nonzero dx/dy regardless of their own position —
    exactly what this test forbids."""
    env, manager, sim_adapter = make_env(num_vbs=2, coverage_radius=5.0, sensing_radius_multiplier=2.0)
    env.reset(seed=0)
    vbs0, vbs1 = manager.vbs_registry[0], manager.vbs_registry[1]

    # Put vbs_0 and vbs_1 on different branches so they are spatially distinct.
    env._apply_actions({"vbs_0": 10, "vbs_1": 0})  # branch 1 slot 10; branch 1 slot 0
    x0, y0 = env._calculate_world_coords(vbs0, True)
    x1, y1 = env._calculate_world_coords(vbs1, True)
    assert (x0, y0) != (x1, y1), "test fixture requires spatially distinct agents"

    sim_adapter.user_coords[0] = [x0 + 1.0, y0 + 1.0]  # right next to vbs_0
    sim_adapter.user_coords[1:] = [x1 + 500.0, y1 + 500.0]  # far from both agents' sensing radii
    env.last_coverage_matrix = np.zeros((len(env.agents), sim_adapter.num_users), dtype=bool)

    obs, _ = env._compute_observations_and_masks()
    dx0, dy0, presence0 = _extract_local_signal(obs["vbs_0"], env.n_vbs)
    dx1, dy1, presence1 = _extract_local_signal(obs["vbs_1"], env.n_vbs)

    assert presence0 == 1.0, "vbs_0 has an uncovered user well within its sensing radius"
    assert presence1 == 0.0, "vbs_1 has nothing within its sensing radius"
    assert (dx0, dy0) != (dx1, dy1), (
        "vbs_0 and vbs_1 received identical dx/dy — this is the global-centroid "
        "mode-collapse bug: a single shared mean broadcast into every agent's obs"
    )
    assert dx1 == 0.0 and dy1 == 0.0
