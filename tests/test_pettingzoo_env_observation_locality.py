"""
Regression coverage for the observation-locality fix in
rl/envs/pettingzoo_env.py::_compute_observations_and_masks.

BUG LEDGER context: two per-agent observations used to leak global ground
truth broadcast identically into every agent:
  1. `uncovered_centroid` — mean of ALL uncovered users on the map, injected
     into every agent's dx/dy regardless of distance. Also a degenerate
     reward-shaping heuristic (single shared attractor collapses agents to
     one point).
  2. `branch_occupancy` — fraction of ALL VBS on each branch, broadcast into
     every VBS's local observation with no plausible local sensor.

These tests pin down the fixed contract:
  - branch_occupancy is gone from the VBS observation (dim count reflects
    this: 13 + n_vbs for VBS, 16 + n_fbs for FBS).
  - the uncovered-user gradient cue (dx, dy, presence) is bounded by each
    agent's own sensing_radius (scaled off its own coverage_radius) and is
    genuinely per-agent: two agents far apart with different local
    uncovered-user distributions must see different dx/dy/presence values.
  - presence=0 with dx=dy=0 when nothing is within sensing range, distinct
    from an in-range detection.
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


def test_vbs_observation_space_excludes_branch_occupancy():
    env, _, _ = make_env(num_vbs=2)
    # 13 fixed dims (no branch_occupancy) + n_vbs identity one-hot
    assert env.observation_space("vbs_0").shape == (13 + env.n_vbs,)


def test_fbs_observation_space_gained_presence_bit():
    env, manager, _ = make_env(num_vbs=1)
    fbs = FlyingBaseStation(id=99, host_vbs_id=0, capacity=10, coverage_radius=20.0, maximum_distance=15.0)
    manager.register_fbs(fbs)
    manager.assign_identity_indices()
    env.n_fbs = len(manager.fbs_registry)
    assert env.observation_space("fbs_99").shape == (16 + env.n_fbs,)


def test_no_branch_occupancy_leak_into_vbs_obs():
    """The old bug broadcast an identical 3-dim branch_occupancy vector into
    every VBS. Confirm the fixed obs width has no room for it and that two
    VBS on different branches don't carry any team-wide fraction feature."""
    env, manager, _ = make_env(num_vbs=2)
    env._apply_actions({"vbs_0": 0, "vbs_1": 21})  # branch 1 slot 0, branch 2 slot 10
    obs, _ = env._compute_observations_and_masks()
    assert obs["vbs_0"].shape == (13 + env.n_vbs,)
    assert obs["vbs_1"].shape == (13 + env.n_vbs,)


def test_local_uncovered_signal_differs_by_agent_position():
    """Two VBS far apart, with an uncovered user near only one of them, must
    see DIFFERENT dx/dy/presence — proof the feature is no longer a shared
    global broadcast."""
    env, manager, sim_adapter = make_env(num_vbs=2, coverage_radius=5.0, sensing_radius_multiplier=2.0)

    vbs0 = manager.vbs_registry[0]
    vbs1 = manager.vbs_registry[1]

    # Place vbs_0 at branch 1 slot 10 (far along branch 1), vbs_1 at home (slot 0).
    env._apply_actions({"vbs_0": 10, "vbs_1": 0})

    x0, y0 = env._calculate_world_coords(vbs0, True)
    x1, y1 = env._calculate_world_coords(vbs1, True)
    assert (x0, y0) != (x1, y1), "test fixture requires spatially distinct agents"

    # No coverage history yet -> _compute_observations_and_masks treats all
    # users as "uncovered" once last_coverage_matrix is populated. Fake a
    # coverage matrix marking every user as uncovered (all False) and place
    # exactly one uncovered user very close to vbs_0 and far from vbs_1.
    sim_adapter.user_coords[0] = [x0 + 1.0, y0 + 1.0]  # right next to vbs_0
    far_x = min(max(x1 + 1000.0, 0.0), env.map_dim[0])  # push far outside vbs_1's sensing range if possible
    sim_adapter.user_coords[1:] = [far_x, y1]
    env.last_coverage_matrix = np.zeros((len(env.agents), sim_adapter.num_users), dtype=bool)

    obs, _ = env._compute_observations_and_masks()

    vbs0_dx, vbs0_dy, vbs0_presence = obs["vbs_0"][-3 - env.n_vbs], obs["vbs_0"][-2 - env.n_vbs], obs["vbs_0"][-1 - env.n_vbs]
    vbs1_dx, vbs1_dy, vbs1_presence = obs["vbs_1"][-3 - env.n_vbs], obs["vbs_1"][-2 - env.n_vbs], obs["vbs_1"][-1 - env.n_vbs]

    assert vbs0_presence == 1.0, "vbs_0 has an uncovered user within its sensing radius"
    assert not (vbs0_dx == vbs1_dx and vbs0_dy == vbs1_dy and vbs0_presence == vbs1_presence), (
        "dx/dy/presence must differ between spatially distinct agents; "
        "identical values indicate a reintroduced global broadcast"
    )


def test_no_detection_yields_zero_vector_and_zero_presence():
    """If nothing is within sensing range, dx=dy=0 AND presence=0 (not the old
    map-center fallback, which produced a nonzero dx/dy pointing at the map
    center for every agent)."""
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
    dx, dy, presence = obs["vbs_0"][-3 - env.n_vbs], obs["vbs_0"][-2 - env.n_vbs], obs["vbs_0"][-1 - env.n_vbs]
    assert presence == 0.0
    assert dx == 0.0
    assert dy == 0.0


def test_global_extra_dim_unaffected_by_locality_fix():
    """global_extra_dim is derived solely from true_coverage + the uncovered
    density grid, neither of which involved branch_occupancy or the removed
    global centroid — must be unchanged by this fix."""
    env, _, _ = make_env(num_vbs=2)
    assert env.global_extra_dim == 1 + env.uncovered_grid_size ** 2
