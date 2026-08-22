import os

import numpy as np
import pytest

from core.entities.agents import AgentManager, VehicleBaseStation, FlyingBaseStation
from infrastructure.graph.networkx_engine import NetworkXRoadEngine
from infrastructure.simulation.pywisim_adapter import PyWiSimAdapter
from rl.envs.pettingzoo_env import CoverageParallelEnv

GRAPH_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "graph_map.json")


def make_env(num_vbs=2, num_fbs=1, max_cycles=5, termination_goal=0.999, num_users=20):
    graph_engine = NetworkXRoadEngine()
    graph_engine.load_from_json(GRAPH_PATH)

    manager = AgentManager()
    for i in range(num_vbs):
        manager.register_vbs(VehicleBaseStation(id=i, capacity=10, coverage_radius=15.0))
    manager.assign_home_branches(num_branches=3)

    for j in range(num_fbs):
        manager.register_fbs(FlyingBaseStation(
            id=num_vbs + j, host_vbs_id=j % max(num_vbs, 1),
            capacity=10, coverage_radius=20.0, maximum_distance=15.0,
        ))
    manager.assign_identity_indices()

    sim_adapter = PyWiSimAdapter(num_users=num_users, map_dimensions=graph_engine.get_map_dimension())

    config = {
        "agent_manager": manager,
        "graph_engine": graph_engine,
        "sim_adapter": sim_adapter,
        "max_cycles": max_cycles,
        "termination_goal": termination_goal,
    }
    env = CoverageParallelEnv(config)
    return env, manager, sim_adapter


def _run_random_steps(env, n_steps, seed=0):
    rng = np.random.default_rng(seed)
    env.reset(seed=seed)
    terminations, truncations = {}, {}

    steps_run = 0
    while env.agents and steps_run < n_steps:
        actions = {a: int(rng.integers(0, env.action_space(a).n)) for a in env.agents}
        _, _, terminations, truncations, _ = env.step(actions)
        steps_run += 1

    terminated = bool(terminations) and all(terminations.values())
    truncated = bool(truncations) and all(truncations.values()) and not terminated
    return terminated, truncated


# --- Bug #1: team_norm must actually update -------------------------------- #

def test_team_norm_updates_across_rollout():
    env, _, _ = make_env(max_cycles=5)
    env.reset(seed=0)

    initial_count = env.team_norm.count
    initial_mean = env.team_norm.mean
    assert initial_count == pytest.approx(1e-4)

    for _ in range(4):
        if not env.agents:
            break
        env.step({a: 0 for a in env.agents})

    assert env.team_norm.count > initial_count, (
        "team_norm.count did not increase -- update() is not being called during step()"
    )
    assert env.team_norm.mean != pytest.approx(initial_mean) or env.team_norm.count > 4, (
        "team_norm.mean appears frozen at its cold-start value"
    )


def test_team_norm_updated_once_per_step_not_once_per_agent():
    env, manager, _ = make_env(num_vbs=3, num_fbs=1, max_cycles=5)
    env.reset(seed=0)
    n_agents = len(env.agents)
    assert n_agents == 4

    count_before = env.team_norm.count
    env.step({a: 0 for a in env.agents})
    count_after = env.team_norm.count

    assert count_after == pytest.approx(count_before + 1), (
        f"team_norm.count increased by {count_after - count_before} with {n_agents} agents "
        "-- expected exactly +1 (update() called once per step, not once per agent)"
    )


# --- Bug #2: get_global_state() after a terminal step must not zero out ---- #

def test_get_global_state_after_truncation_returns_real_features_not_zeros():
    env, manager, sim_adapter = make_env(num_vbs=2, num_fbs=1, max_cycles=3, termination_goal=0.999)

    terminated, truncated = _run_random_steps(env, n_steps=3, seed=1)
    assert truncated and not terminated
    assert env.agents == []

    vbs_feats, fbs_feats, global_extra = env.get_global_state()

    assert vbs_feats.shape == (2, env.observation_space("vbs_0").shape[0])
    assert fbs_feats.shape == (1, env.observation_space("fbs_2").shape[0])
    assert not np.allclose(vbs_feats, 0.0), "VBS features fell back to the zero-padding branch"
    assert not np.allclose(fbs_feats, 0.0), "FBS features fell back to the zero-padding branch"


def test_get_global_state_post_terminal_matches_pre_terminal_last_obs():
    env, manager, sim_adapter = make_env(num_vbs=2, num_fbs=1, max_cycles=3, termination_goal=0.999)

    rng = np.random.default_rng(2)
    env.reset(seed=2)
    last_agents_order = None
    while env.agents:
        last_agents_order = list(env.agents)
        actions = {a: int(rng.integers(0, env.action_space(a).n)) for a in env.agents}
        env.step(actions)

    assert env.agents == []
    assert last_agents_order is not None

    vbs_feats, fbs_feats, global_extra = env.get_global_state()

    expected_vbs = np.stack([env._last_obs[a] for a in last_agents_order if "vbs" in a])
    expected_fbs = np.stack([env._last_obs[a] for a in last_agents_order if "fbs" in a])

    np.testing.assert_array_equal(vbs_feats, expected_vbs)
    np.testing.assert_array_equal(fbs_feats, expected_fbs)


def test_get_global_state_still_works_mid_episode():
    env, manager, sim_adapter = make_env(num_vbs=2, num_fbs=1, max_cycles=5, termination_goal=0.999)
    env.reset(seed=3)
    env.step({a: 0 for a in env.agents})

    assert env.agents != []
    vbs_feats, fbs_feats, global_extra = env.get_global_state()

    assert vbs_feats.shape == (2, env.observation_space("vbs_0").shape[0])
    assert fbs_feats.shape == (1, env.observation_space("fbs_2").shape[0])
    assert global_extra.shape[0] == env.global_extra_dim


def test_get_global_state_before_any_reset_uses_empty_fallback():
    """self.agents is populated at __init__ time (from possible_agents), but
    _last_obs has no entries yet -- get_global_state() must not try to key
    into _last_obs using self.agents' IDs (that was the actual bug: the
    active-agent set has to be _last_obs's OWN keys, not a fallback to
    self.agents when _last_obs is empty)."""
    env, _, _ = make_env(num_vbs=2, num_fbs=1)

    vbs_feats, fbs_feats, global_extra = env.get_global_state()

    assert vbs_feats.shape == (1, 13 + env.n_vbs)
    assert fbs_feats.shape == (1, 16 + env.n_fbs)
    assert global_extra.shape[0] == env.global_extra_dim


# --- Integration: bootstrap input reaches the critic non-degenerate -------- #

def test_truncation_bootstrap_value_is_not_degenerate_zero_input_artifact():
    import torch
    from rl.agents.ppo_module import HeterogeneousPPOManager

    env, manager, sim_adapter = make_env(num_vbs=2, num_fbs=1, max_cycles=3, termination_goal=0.999)

    vbs_agent_id = next(a for a in env.possible_agents if "vbs" in a)
    fbs_agent_id = next(a for a in env.possible_agents if "fbs" in a)
    ppo = HeterogeneousPPOManager(
        vbs_obs_dim=env.observation_space(vbs_agent_id).shape[0],
        fbs_obs_dim=env.observation_space(fbs_agent_id).shape[0],
        vbs_action_dim=env.action_space(vbs_agent_id).n,
        fbs_action_dim=env.action_space(fbs_agent_id).n,
        global_extra_dim=env.global_extra_dim,
        device="cpu",
    )

    terminated, truncated = _run_random_steps(env, n_steps=3, seed=4)
    assert truncated and not terminated

    f_vbs, f_fbs, f_extra = env.get_global_state()
    assert not np.allclose(f_vbs, 0.0)
    assert not np.allclose(f_fbs, 0.0)

    bootstrap_value = ppo.get_value(
        torch.tensor(f_vbs, dtype=torch.float32).unsqueeze(0),
        torch.tensor(f_fbs, dtype=torch.float32).unsqueeze(0),
        torch.tensor(f_extra, dtype=torch.float32).unsqueeze(0),
    )
    assert isinstance(bootstrap_value["team"], float)
    assert len(bootstrap_value["vbs"]) == 2
    assert len(bootstrap_value["fbs"]) == 1