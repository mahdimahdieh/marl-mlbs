"""
Regression coverage for observation locality: no global ground truth
(branch_occupancy, global uncovered-user mean) may be broadcast into
per-agent observations. The local directional cue (8 sector fractions +
presence) must be sensing_radius-bounded, genuinely per-agent, and must not
cancel to a dead-zone zero signal between symmetric uncovered clusters.
"""
import os
import numpy as np
import pytest

from core.entities.agents import AgentManager, VehicleBaseStation, FlyingBaseStation
from infrastructure.graph.networkx_engine import NetworkXRoadEngine
from infrastructure.simulation.pywisim_adapter import PyWiSimAdapter
from rl.envs.pettingzoo_env import CoverageParallelEnv

GRAPH_PATH = os.path.join(
    os.path.dirname(__file__), "..", "config", "graph_map.json"
)


def make_env(num_vbs=2, sensing_radius_multiplier=2.5, coverage_radius=15.0):
    graph_engine = NetworkXRoadEngine()
    graph_engine.load_from_json(GRAPH_PATH)

    manager = AgentManager()
    for i in range(num_vbs):
        vbs = VehicleBaseStation(id=i, capacity=10, coverage_radius=coverage_radius)
        manager.register_vbs(vbs)
    manager.assign_home_branches(num_branches=3)
    manager.assign_identity_indices()

    sim_adapter = PyWiSimAdapter(num_users=20, map_dimensions=graph_engine.get_map_dimension())

    config = {
        "agent_manager": manager,
        "graph_engine": graph_engine,
        "sim_adapter": sim_adapter,
        "max_cycles": 10,
        "termination_goal": 0.95,
        "max_slot_per_branch": 10,
        "sensing_radius_multiplier": sensing_radius_multiplier,
    }
    env = CoverageParallelEnv(config)
    env.reset(seed=0)
    return env, manager, sim_adapter


def _extract_local_signal(obs_row, n_identities):
    """Obs layout tail (both agent types): [local_sector_fracs(8),
    local_uncovered_presence(1), identity_hot(n)]."""
    n_bins = CoverageParallelEnv.NUM_LOCAL_SECTOR_BINS
    presence = float(obs_row[-1 - n_identities])
    sector_fracs = obs_row[-(1 + n_identities + n_bins): -1 - n_identities]
    return sector_fracs, presence


def test_vbs_observation_space_excludes_branch_occupancy():
    env, _, _ = make_env(num_vbs=2)
    # 10 fixed dims (no branch_occupancy) + 8 sector bins + 1 presence + n_vbs identity
    assert env.observation_space("vbs_0").shape == (10 + env.NUM_LOCAL_SECTOR_BINS + 1 + env.n_vbs,)


def test_fbs_observation_space_gained_presence_bit():
    env, manager, _ = make_env(num_vbs=1)
    fbs = FlyingBaseStation(id=99, host_vbs_id=0, capacity=10, coverage_radius=20.0, maximum_distance=15.0)
    manager.register_fbs(fbs)
    manager.assign_identity_indices()
    env.n_fbs = len(manager.fbs_registry)
    assert env.observation_space("fbs_99").shape == (13 + env.NUM_LOCAL_SECTOR_BINS + 1 + env.n_fbs,)


def test_no_branch_occupancy_leak_into_vbs_obs():
    """The old bug broadcast an identical 3-dim branch_occupancy vector into
    every VBS — the fixed obs width has no room for it."""
    env, manager, _ = make_env(num_vbs=2)
    env._apply_actions({"vbs_0": 0, "vbs_1": 21})  # branch 1 slot 0, branch 2 slot 10
    obs, _ = env._compute_observations_and_masks()
    expected = (10 + env.NUM_LOCAL_SECTOR_BINS + 1 + env.n_vbs,)
    assert obs["vbs_0"].shape == expected
    assert obs["vbs_1"].shape == expected


def test_local_uncovered_signal_differs_by_agent_position():
    """Two VBS far apart, with an uncovered user near only one of them, must
    see DIFFERENT sector histograms / presence (no shared global broadcast)."""
    env, manager, sim_adapter = make_env(num_vbs=2, coverage_radius=5.0, sensing_radius_multiplier=2.0)

    vbs0 = manager.vbs_registry[0]
    vbs1 = manager.vbs_registry[1]

    # Place vbs_0 at branch 1 slot 10 (far along branch 1), vbs_1 at home (slot 0).
    env._apply_actions({"vbs_0": 10, "vbs_1": 0})

    x0, y0 = env._calculate_world_coords(vbs0, True)
    x1, y1 = env._calculate_world_coords(vbs1, True)
    assert (x0, y0) != (x1, y1), "test fixture requires spatially distinct agents"

    # Fake an all-uncovered matrix; place one uncovered user very close to
    # vbs_0 and everything else far from both agents.
    sim_adapter.user_coords[0] = [x0 + 1.0, y0 + 1.0]  # right next to vbs_0
    far_x = min(max(x1 + 1000.0, 0.0), env.map_dim[0])
    sim_adapter.user_coords[1:] = [far_x, y1]
    env.last_coverage_matrix = np.zeros((len(env.agents), sim_adapter.num_users), dtype=bool)

    obs, _ = env._compute_observations_and_masks()

    vbs0_sectors, vbs0_presence = _extract_local_signal(obs["vbs_0"], env.n_vbs)
    vbs1_sectors, vbs1_presence = _extract_local_signal(obs["vbs_1"], env.n_vbs)

    assert vbs0_presence == 1.0, "vbs_0 has an uncovered user within its sensing radius"
    assert vbs1_presence == 0.0
    assert not (np.allclose(vbs0_sectors, vbs1_sectors) and vbs0_presence == vbs1_presence), (
        "identical local signals for spatially distinct agents indicate a "
        "reintroduced global broadcast"
    )


def test_no_detection_yields_zero_sectors_and_zero_presence():
    """If nothing is within sensing range, all sectors are 0 AND presence=0
    (not the old map-center fallback, which pointed every agent at center)."""
    env, manager, sim_adapter = make_env(num_vbs=1, coverage_radius=1.0, sensing_radius_multiplier=1.0)
    vbs0 = manager.vbs_registry[0]
    env._apply_actions({"vbs_0": 0})
    x0, y0 = env._calculate_world_coords(vbs0, True)

    # Every user is uncovered but placed far away from vbs_0.
    sim_adapter.user_coords[:] = [
        min(x0 + 1000.0, env.map_dim[0]),
        min(y0 + 1000.0, env.map_dim[1]),
    ]
    env.last_coverage_matrix = np.zeros((len(env.agents), sim_adapter.num_users), dtype=bool)

    obs, _ = env._compute_observations_and_masks()
    sectors, presence = _extract_local_signal(obs["vbs_0"], env.n_vbs)
    assert presence == 0.0
    assert np.all(sectors == 0.0)


def test_symmetric_uncovered_clusters_do_not_cancel_local_signal():
    """Task 3 dead-zone regression: an agent between two symmetric uncovered
    clusters must see two opposite sector peaks, not the cancelled zero cue
    the old vector-mean produced."""
    env, manager, sim_adapter = make_env(num_vbs=1, coverage_radius=5.0, sensing_radius_multiplier=2.0)
    vbs0 = manager.vbs_registry[0]
    env._apply_actions({"vbs_0": 0})
    x0, y0 = env._calculate_world_coords(vbs0, True)

    d = 3.0  # well within sensing radius (5.0 * 2.0)
    sim_adapter.user_coords[:] = [x0, y0]
    sim_adapter.user_coords[0::2] = [x0 + d, y0]  # east cluster
    sim_adapter.user_coords[1::2] = [x0 - d, y0]  # west cluster
    env.last_coverage_matrix = np.zeros((len(env.agents), sim_adapter.num_users), dtype=bool)

    obs, _ = env._compute_observations_and_masks()
    sectors, presence = _extract_local_signal(obs["vbs_0"], env.n_vbs)

    assert presence == 1.0, "symmetric clusters are within sensing range — must not read as 'nothing detected'"
    nonzero = np.nonzero(sectors)[0]
    assert len(nonzero) == 2, (
        f"expected two opposite sector peaks from the symmetric clusters, got {len(nonzero)} — "
        "a mean-based cue would cancel them to the dead-zone zero signal"
    )
    assert abs(int(nonzero[0]) - int(nonzero[1])) == env.NUM_LOCAL_SECTOR_BINS // 2
    assert sectors.sum() == pytest.approx(1.0)


def test_global_extra_dim_unaffected_by_locality_fix():
    """global_extra_dim is derived solely from true_coverage + the uncovered
    density grid — unaffected by the locality fixes."""
    env, _, _ = make_env(num_vbs=2)
    assert env.global_extra_dim == 1 + env.uncovered_grid_size ** 2
