"""
Canonical CoverageParallelEnv regression suite: checks that DECLARED spaces
(observation_space()/action_space()) match runtime output, that VBS actions
are absolute (branch, slot) selections, and that the per-agent local sector
sensing (Task 2/3) is bounded, agent-specific and free of the symmetric-
cluster dead-zone. Critic granularity is covered by
tests/test_critic_agent_granularity.py.
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
        "action_space().n has drifted from the actual action_mask width — "
        "this silently crashes main.py/inference.py's PPO forward pass."
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
    # 10 fixed (pos/coverage/slot/branch_hot/home_hot) + 8 sector bins + 1 presence
    assert declared_shape == (10 + env.NUM_LOCAL_SECTOR_BINS + 1 + env.n_vbs,)
    assert obs_dict["vbs_0"].shape == declared_shape


def test_fbs_observation_space_matches_actual_obs_shape():
    env, _, _ = make_env(num_vbs=2, num_fbs=2)
    obs_dict, _ = env.reset(seed=0)

    declared_shape = env.observation_space("fbs_2").shape
    # 13 fixed (pos/coverage/polar/host_branch/ema/host_true) + 8 sector bins + 1 presence
    assert declared_shape == (13 + env.NUM_LOCAL_SECTOR_BINS + 1 + env.n_fbs,)
    assert obs_dict["fbs_2"].shape == declared_shape


def test_global_state_feature_widths_match_declared_obs_dims():
    """get_global_state()'s stacked rows must match observation_space() widths
    (guards against silent per-agent feature-width drift)."""
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
    """Every single action call must land EXACTLY on its decoded (branch, slot)
    target in one step, with zero dependency on prior state (the old relative
    accumulator drifted instead)."""
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
    """Repeatedly issuing the SAME action must always land on the SAME state —
    the decode is a pure function of the action alone."""
    env, manager, _ = make_env(num_vbs=1)
    env.reset(seed=0)
    vbs = manager.vbs_registry[0]

    for _ in range(5):
        env._apply_actions({"vbs_0": 17})  # branch 2, slot 6
        assert (vbs.current_branch_id, vbs.current_slot_index) == (2, 6)


# --------------------------------------------------------------------------- #
# 1c. Task 2/3 regression: local sensing sector histogram / agent-specific    #
# --------------------------------------------------------------------------- #

def _extract_local_signal(obs_row, n_identities):
    """Obs layout tail (both agent types): [local_sector_fracs(8),
    local_uncovered_presence(1), identity_hot(n)]."""
    n_bins = CoverageParallelEnv.NUM_LOCAL_SECTOR_BINS
    presence = float(obs_row[-1 - n_identities])
    sector_fracs = obs_row[-(1 + n_identities + n_bins): -1 - n_identities]
    return sector_fracs, presence


def test_local_sensing_returns_zero_sectors_and_zero_presence_when_nothing_in_range():
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
    sector_fracs, presence = _extract_local_signal(obs["vbs_0"], env.n_vbs)

    assert presence == 0.0
    assert np.all(sector_fracs == 0.0)


def test_local_sensing_nonzero_and_agent_specific_with_asymmetric_ground_truth():
    """Regression for the global-centroid mode-collapse bug: a single uncovered
    user near vbs_0 only — vbs_0 must detect it, far-away vbs_1 must not."""
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
    sectors0, presence0 = _extract_local_signal(obs["vbs_0"], env.n_vbs)
    sectors1, presence1 = _extract_local_signal(obs["vbs_1"], env.n_vbs)

    assert presence0 == 1.0, "vbs_0 has an uncovered user well within its sensing radius"
    assert sectors0.sum() == pytest.approx(1.0), "detected users must distribute mass 1.0 across sectors"
    assert np.any(sectors0 > 0.0)
    assert presence1 == 0.0, "vbs_1 has nothing within its sensing radius"
    assert np.all(sectors1 == 0.0)
    assert not np.allclose(sectors0, sectors1), (
        "vbs_0 and vbs_1 received identical sector histograms — "
        "a reintroduced global broadcast"
    )


def test_local_sensing_symmetric_clusters_produce_two_peaks_not_a_dead_zone():
    """Task 3 dead-zone regression: an agent exactly between two symmetric
    uncovered clusters must see TWO opposite sector peaks (the old vector-mean
    cue cancelled to (dx=0, dy=0), falsely signalling 'target underneath me')."""
    env, manager, sim_adapter = make_env(num_vbs=1, coverage_radius=5.0, sensing_radius_multiplier=2.0)
    env.reset(seed=0)
    vbs0 = manager.vbs_registry[0]
    env._apply_actions({"vbs_0": 0})
    x0, y0 = env._calculate_world_coords(vbs0, True)

    # Two equal, symmetric uncovered clusters: due east and due west of the agent.
    d = 3.0  # well within sensing radius (5.0 * 2.0)
    sim_adapter.user_coords[:] = [x0, y0]  # placeholder, overwritten below
    sim_adapter.user_coords[0::2] = [x0 + d, y0]  # east cluster
    sim_adapter.user_coords[1::2] = [x0 - d, y0]  # west cluster
    env.last_coverage_matrix = np.zeros((len(env.agents), sim_adapter.num_users), dtype=bool)

    obs, _ = env._compute_observations_and_masks()
    sectors, presence = _extract_local_signal(obs["vbs_0"], env.n_vbs)

    assert presence == 1.0, "two symmetric clusters are firmly within sensing range"
    nonzero = np.nonzero(sectors)[0]
    assert len(nonzero) == 2, (
        f"expected exactly two sector peaks from symmetric clusters, got {len(nonzero)}"
    )
    # The peaks must sit in OPPOSITE sectors and split the mass evenly.
    assert abs(int(nonzero[0]) - int(nonzero[1])) == env.NUM_LOCAL_SECTOR_BINS // 2
    for idx in nonzero:
        assert sectors[idx] == pytest.approx(0.5)
    assert sectors.sum() == pytest.approx(1.0)
