"""
Regression coverage for the VBS action-space fix in rl/envs/pettingzoo_env.py.

BUG LEDGER context: VBS actions used to be relative deltas (increment/decrement
current_slot_index depending on whether the chosen action matched
current_branch_id). These tests pin down the new absolute, factored
(branch, slot) contract so that regression back to relative semantics is
caught immediately:
  - action_space size reflects branches * (slots + 1)
  - a single action call fully determines (current_branch_id, current_slot_index)
    with no dependency on the agent's previous state
  - the action mask no longer needs to forbid any action (no overshoot is
    possible under absolute selection)
"""
import os
import pytest

from core.entities.agents import AgentManager, VehicleBaseStation, FlyingBaseStation
from infrastructure.graph.networkx_engine import NetworkXRoadEngine
from infrastructure.simulation.pywisim_adapter import PyWiSimAdapter
from rl.envs.pettingzoo_env import CoverageParallelEnv

GRAPH_PATH = os.path.join(
    os.path.dirname(__file__), "..", "config", "graph_map.json"
)


@pytest.fixture
def env():
    graph_engine = NetworkXRoadEngine()
    graph_engine.load_from_json(GRAPH_PATH)

    manager = AgentManager()
    vbs = VehicleBaseStation(id=0, capacity=10, coverage_radius=15.0)
    manager.register_vbs(vbs)
    manager.assign_home_branches(num_branches=3)
    manager.assign_identity_indices()

    fbs = FlyingBaseStation(
        id=1, host_vbs_id=0, capacity=10, coverage_radius=20.0, maximum_distance=15.0
    )
    manager.register_fbs(fbs)

    sim_adapter = PyWiSimAdapter(num_users=10, map_dimensions=graph_engine.get_map_dimension())

    config = {
        "agent_manager": manager,
        "graph_engine": graph_engine,
        "sim_adapter": sim_adapter,
        "max_cycles": 10,
        "termination_goal": 0.95,
        "max_slot_per_branch": 10,
    }
    e = CoverageParallelEnv(config)
    e.reset(seed=0)
    return e


def test_vbs_action_space_size_is_branches_times_slots(env):
    # 3 branches * 11 slots (0..10 inclusive) = 33
    assert env.action_space("vbs_0").n == 33


def test_vbs_decode_action_is_absolute(env):
    # action 0 -> branch 1, slot 0
    assert env._decode_vbs_action(0) == (1, 0)
    # action 10 -> branch 1, slot 10 (last slot of branch 1)
    assert env._decode_vbs_action(10) == (1, 10)
    # action 11 -> branch 2, slot 0
    assert env._decode_vbs_action(11) == (2, 0)
    # action 32 -> branch 3, slot 10
    assert env._decode_vbs_action(32) == (3, 10)


def test_apply_actions_sets_absolute_state_regardless_of_prior_state(env):
    vbs = env.agent_manager.vbs_registry[0]

    # Drive to branch 2, slot 7.
    env._apply_actions({"vbs_0": 11 + 7, "fbs_1": 0})
    assert vbs.current_branch_id == 2
    assert vbs.current_slot_index == 7

    # A single action call must fully re-determine state — no multi-step
    # transit required, and no dependency on the previous (branch=2, slot=7).
    env._apply_actions({"vbs_0": 32, "fbs_1": 0})  # branch 3, slot 10
    assert vbs.current_branch_id == 3
    assert vbs.current_slot_index == 10

    # And back down in one step, including to slot 0 on a different branch.
    env._apply_actions({"vbs_0": 0, "fbs_1": 0})  # branch 1, slot 0
    assert vbs.current_branch_id == 1
    assert vbs.current_slot_index == 0


def test_vbs_action_mask_is_never_restricted(env):
    # Push VBS to the max slot on some branch, then confirm no action index
    # is masked out (overshoot is structurally impossible now).
    env._apply_actions({"vbs_0": 10, "fbs_1": 0})  # branch 1, slot 10 (max)
    _, infos = env._compute_observations_and_masks()
    mask = infos["vbs_0"]["action_mask"]
    assert mask.sum() == env.action_space("vbs_0").n
