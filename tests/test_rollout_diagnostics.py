"""Rollout diagnostic for the Task-1 VBS relative-action bug: chosen actions
used to only increment/decrement current_slot_index, trapping agents in a
2-state limit cycle. Drives the real PPO action path and checks the absolute
action contract end-to-end. Standalone: python tests/test_rollout_diagnostics.py
"""
import os
import sys
from typing import Dict, List, Tuple

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.entities.agents import AgentManager, VehicleBaseStation, FlyingBaseStation
from infrastructure.graph.networkx_engine import NetworkXRoadEngine
from infrastructure.simulation.pywisim_adapter import PyWiSimAdapter
from infrastructure.training.determinism import lock_determinism
from rl.envs.pettingzoo_env import CoverageParallelEnv
from rl.agents.ppo_module import HeterogeneousPPOManager

GRAPH_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "graph_map.json")
MAX_SLOT_PER_BRANCH = 10


def detect_2state_cycle(states: List[Tuple[int, int]], min_alternations: int = 4,
                         persistence_frac: float = 0.25):
    """Finds the longest run alternating between exactly two distinct states,
    flagged only when it spans >= persistence_frac of the whole trajectory.
    Short argmax flip-flops of an untrained policy are noise, not trapping.
    Returns (found, details) with details = (state_a, state_b, start, length).
    """
    n = len(states)
    required = max(min_alternations, int(round(persistence_frac * n)))
    best = None
    for start in range(n - 1):
        a, b = states[start], states[start + 1]
        if a == b:
            continue
        run_length = 2
        i = start + 2
        while i < n and states[i] == (a if (i - start) % 2 == 0 else b):
            run_length += 1
            i += 1
        if run_length - 1 >= required and (best is None or run_length > best[3]):
            best = (a, b, start, run_length)
    return (best is not None), best


def run_rollout_diagnostic(max_steps: int = 60, num_vbs: int = 3, num_fbs: int = 2,
                           seed: int = 7) -> Dict[str, List[Tuple[int, int, int]]]:
    """Short headless rollout via the real PPO action-sampling path; records
    each VBS's per-step (action, branch, slot) trajectory."""
    lock_determinism(seed)

    graph_engine = NetworkXRoadEngine()
    graph_engine.load_from_json(GRAPH_PATH)

    manager = AgentManager()
    for i in range(num_vbs):
        manager.register_vbs(VehicleBaseStation(id=i, capacity=10, coverage_radius=15.0))
    manager.assign_home_branches(num_branches=3)
    for j in range(num_fbs):
        manager.register_fbs(FlyingBaseStation(
            id=num_vbs + j, host_vbs_id=j % num_vbs,
            capacity=10, coverage_radius=25.0, maximum_distance=15.0,
        ))
    manager.assign_identity_indices()

    sim_adapter = PyWiSimAdapter(num_users=50, map_dimensions=graph_engine.get_map_dimension())
    env = CoverageParallelEnv({
        "agent_manager": manager,
        "graph_engine": graph_engine,
        "sim_adapter": sim_adapter,
        "max_cycles": max_steps,
        "termination_goal": 0.999,  # unreachable -> run the full horizon
        "max_slot_per_branch": MAX_SLOT_PER_BRANCH,
    })

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

    obs_dict, infos_dict = env.reset(seed=seed)
    vbs_agents = [a for a in env.agents if "vbs" in a]
    trajectories: Dict[str, List[Tuple[int, int, int]]] = {a: [] for a in vbs_agents}

    step = 0
    while env.agents and step < max_steps:
        actions = {}
        for agent_id in env.agents:
            agent_type = "vbs" if "vbs" in agent_id else "fbs"
            t_obs = torch.tensor(obs_dict[agent_id], dtype=torch.float32)
            t_mask = torch.tensor(infos_dict[agent_id]["action_mask"], dtype=torch.float32)
            # Deterministic argmax: the run stays a reproducible function of
            # env dynamics alone, not Categorical sampling noise.
            action = ppo.get_deterministic_action(t_obs, agent_type, action_mask=t_mask)
            actions[agent_id] = action

        obs_dict, rewards_dict, terminations, truncations, infos_dict = env.step(actions)

        for agent_id in vbs_agents:
            vbs_obj, _ = env._get_agent_obj(agent_id)
            trajectories[agent_id].append(
                (actions[agent_id], vbs_obj.current_branch_id, vbs_obj.current_slot_index)
            )
        step += 1

    return trajectories


def test_realized_state_is_pure_function_of_issued_action():
    """Core Task-1 contract: the realized (branch, slot) must equal the decode
    of the action issued that same step, every step. A relative accumulator
    (the old bug) drifts the state away from the decoded target."""
    trajectories = run_rollout_diagnostic()
    slots_per_branch = MAX_SLOT_PER_BRANCH + 1

    for agent_id, steps in trajectories.items():
        assert len(steps) > 0, f"{agent_id} produced no state trajectory"
        for t, (action, branch, slot) in enumerate(steps):
            expected = (action // slots_per_branch + 1, action % slots_per_branch)
            assert (branch, slot) == expected, (
                f"{agent_id} step {t}: action {action} decodes to {expected} but "
                f"state is ({branch}, {slot}) — state is drifting from the issued "
                "action (relative-accumulator regression)"
            )


def test_no_2state_cycle_in_vbs_rollout():
    """No VBS may get trapped alternating between exactly two states for a
    sustained stretch of the rollout (the pre-fix limit-cycle signature)."""
    trajectories = run_rollout_diagnostic()

    for agent_id, steps in trajectories.items():
        states = [(branch, slot) for _, branch, slot in steps]
        assert len(states) > 0, f"{agent_id} produced no state trajectory"
        print(f"{agent_id}: {len(set(states))} distinct (branch, slot) states over {len(states)} steps")

        found, details = detect_2state_cycle(states, min_alternations=4, persistence_frac=0.25)
        assert not found, (
            f"{agent_id} alternates between {details[0]} and {details[1]} for "
            f"{details[3]} consecutive steps from step {details[2]} — "
            "2-state limit cycle regression"
        )


if __name__ == "__main__":
    trajectories = run_rollout_diagnostic()
    print("VBS rollout diagnostic (untrained policy, structural check only):\n")
    for agent_id, steps in trajectories.items():
        states = [(b, s) for _, b, s in steps]
        found, details = detect_2state_cycle(states)
        status = "CYCLE DETECTED" if found else "ok"
        print(f"  {agent_id}: {len(states)} steps, {len(set(states))} distinct "
              f"(branch, slot) states, 2-state-cycle check: {status}")
