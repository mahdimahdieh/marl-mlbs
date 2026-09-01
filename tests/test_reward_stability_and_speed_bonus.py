"""
Regression coverage for reward-signal stability: RunningNorm's std floor /
decay window / clip bounds, and the terminal speed bonus (added exactly once
on solve, zero when solved on the final step).
"""
import os
import pytest

from core.entities.agents import AgentManager, VehicleBaseStation, FlyingBaseStation
from infrastructure.graph.networkx_engine import NetworkXRoadEngine
from infrastructure.simulation.pywisim_adapter import PyWiSimAdapter
from rl.envs.pettingzoo_env import CoverageParallelEnv
from rl.envs.reward_normalizer import RunningNorm

GRAPH_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "graph_map.json")


def make_env(termination_goal, num_vbs=2, num_fbs=1, max_cycles=10, num_users=20):
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
    return CoverageParallelEnv({
        "agent_manager": manager,
        "graph_engine": graph_engine,
        "sim_adapter": sim_adapter,
        "max_cycles": max_cycles,
        "termination_goal": termination_goal,
    })


# --- RunningNorm stability --------------------------------------------------

def test_normalize_output_is_bounded_by_clip():
    norm = RunningNorm(clip=5.0)
    for _ in range(500):
        norm.update(1.0)  # variance collapses, mirroring post-convergence coverage
    out = norm.normalize(1.5)
    assert abs(out) <= 5.0 + 1e-9


def test_min_std_floor_prevents_division_blowup():
    norm = RunningNorm(min_std=0.1, clip=1e9)
    for _ in range(1000):
        norm.update(0.5)
    out = norm.normalize(0.5001)
    assert abs(out) < 1.0


def test_running_norm_tracks_recent_window_not_full_lifetime():
    """The decayed estimator must re-track a shifted regime instead of staying
    anchored near the stale lifetime mean."""
    norm = RunningNorm(decay=0.99)
    for _ in range(2000):
        norm.update(0.0)
    for _ in range(200):
        norm.update(10.0)
    assert norm.mean > 2.0, "mean failed to track the new regime -- window is not decaying"


# --- Terminal speed bonus ---------------------------------------------------

def test_terminal_bonus_added_exactly_once_on_solve():
    """Same config/seed/actions, differing only in termination_goal so one run
    solves immediately and the other never does — the reward diff must equal
    exactly the speed bonus formula."""
    solved_env = make_env(termination_goal=0.0, max_cycles=10)   # terminates at step 1
    unsolved_env = make_env(termination_goal=1.1, max_cycles=10)  # unreachable -> never terminates

    solved_env.reset(seed=0)
    unsolved_env.reset(seed=0)
    actions = {a: 0 for a in solved_env.agents}

    _, solved_rewards, solved_term, _, _ = solved_env.step(actions)
    _, unsolved_rewards, unsolved_term, _, _ = unsolved_env.step(actions)

    assert all(solved_term.values())
    assert not any(unsolved_term.values())

    expected_bonus = (
        solved_env.TERMINAL_SPEED_BONUS
        * (solved_env.max_cycles - solved_env.step_count)
        / solved_env.max_cycles
    )
    assert expected_bonus > 0.0
    for agent_id in solved_rewards:
        assert solved_rewards[agent_id] == pytest.approx(unsolved_rewards[agent_id] + expected_bonus)


def test_terminal_bonus_is_zero_when_solved_on_the_final_step():
    """Solving exactly on the last allowed step saves zero steps -> bonus is
    exactly 0 (no free reward just for terminating vs. truncating)."""
    env = make_env(termination_goal=0.0, max_cycles=1)  # solves on step 1 == max_cycles
    env.reset(seed=0)
    actions = {a: 0 for a in env.agents}
    _, rewards, term, trunc, _ = env.step(actions)

    assert all(term.values())
    assert env.step_count == env.max_cycles
    expected_bonus = env.TERMINAL_SPEED_BONUS * (env.max_cycles - env.step_count) / env.max_cycles
    assert expected_bonus == 0.0