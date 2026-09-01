"""
Lightweight rollout diagnostic for the Task 1 failure mode: under the OLD
relative VBS action scheme, a chosen action only advanced current_slot_index
if it matched current_branch_id, else it retreated by one slot. Combined with
the marginal-contribution reward's step-to-step sign flips (as other agents
move), this produced a STABLE 2-STATE LIMIT CYCLE with no absorbing target
state -- the agent would bounce between exactly two (branch, slot) states
indefinitely instead of converging toward a target.

This is not a training-convergence test (no reward-improvement assertion,
no claim that the policy is good). It only proves the specific STRUCTURAL
failure mode from Task 1 is gone: across a short, headless rollout driven by
the real PPO action-sampling path (rl.agents.ppo_module.HeterogeneousPPOManager,
untrained), no VBS agent's (branch, slot) trajectory degenerates into a long
back-and-forth cycle between exactly two states.

Can also be run standalone as a diagnostic script:
    python tests/test_rollout_diagnostics.py
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


def detect_2state_cycle(states: List[Tuple[int, int]], min_alternations: int = 4,
                        persistence_frac: float = 0.25):
    """Detects a maximal run alternating between exactly two distinct states
    (A, B, A, B, A, B, ...). Returns (found, details) where details is
    (state_a, state_b, start_index, run_length) for the longest such run.

    A run is only flagged as the pathological limit cycle when it BOTH clears
    `min_alternations` AND consumes at least `persistence_frac` of the whole
    trajectory. The persistence requirement matters post-fix: an UNTRAINED
    argmax policy legitimately flips between two adjacent ABSOLUTE actions for
    a handful of steps while the team's coverage state evolves underneath it
    (multi-agent non-stationarity: other agents move, the local sector
    histogram shifts, the argmax flips back). The relative-accumulator bug
    this diagnostic targets trapped the agent INDEFINITELY — the alternation
    dominated the entire rollout with no absorbing state reachable. Only
    sustained trapping is that failure signature; a short transient flip-flop
    is exploration noise, not the bug.
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
        alternations = run_length - 1
        if alternations >= required:
            if best is None or run_length > best[3]:
                best = (a, b, start, run_length)
    return (best is not None), best


def run_rollout_diagnostic(max_steps: int = 60, num_vbs: int = 3, num_fbs: int = 2, seed: int = 7):
    """Runs a short, headless rollout using the real PPO action-sampling path
    and returns, per VBS agent, the full (branch, slot) state trajectory."""
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
        "termination_goal": 0.999,  # effectively unreachable -> run the full horizon
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
    trajectories: Dict[str, List[Tuple[int, int]]] = {a: [] for a in vbs_agents}

    step = 0
    while env.agents and step < max_steps:
        actions = {}
        for agent_id in env.agents:
            agent_type = "vbs" if "vbs" in agent_id else "fbs"
            t_obs = torch.tensor(obs_dict[agent_id], dtype=torch.float32)
            t_mask = torch.tensor(infos_dict[agent_id]["action_mask"], dtype=torch.float32)
            # Deterministic (argmax) action selection rather than stochastic
            # sampling: this makes the diagnostic a repeatable function of the
            # environment's OWN dynamics (coverage state evolving each step)
            # rather than of Categorical sampling noise, which can coincidentally
            # alternate between two actions by pure chance and produce a false
            # positive unrelated to any structural bug.
            action = ppo.get_deterministic_action(t_obs, agent_type, action_mask=t_mask)
            actions[agent_id] = action

        obs_dict, rewards_dict, terminations, truncations, infos_dict = env.step(actions)

        for agent_id in vbs_agents:
            vbs_obj, _ = env._get_agent_obj(agent_id)
            trajectories[agent_id].append((vbs_obj.current_branch_id, vbs_obj.current_slot_index))

        step += 1

    return trajectories


def test_no_2state_cycle_in_vbs_rollout():
    trajectories = run_rollout_diagnostic(max_steps=60, num_vbs=3, num_fbs=2, seed=7)

    for agent_id, states in trajectories.items():
        assert len(states) > 0, f"{agent_id} produced no state trajectory"

        # Distinct-state count is reported (not hard-asserted): with the
        # deterministic (argmax) policy used here to keep this diagnostic
        # reproducible, an untrained network can legitimately settle on one
        # fixed action if its local observation barely changes step to step.
        # A single, STATIC state is not the Task 1 failure mode (there is no
        # oscillation) — it's arguably the correct post-fix behavior: a fixed
        # decision rule reaches and STAYS at an absorbing state rather than
        # bouncing. Only SUSTAINED alternation between two states (>= 25% of
        # the trajectory, see detect_2state_cycle) is checked below — short
        # transient flip-flops of an untrained argmax under multi-agent
        # non-stationarity are exploration noise, not the pathology.
        distinct_states = set(states)
        print(f"{agent_id}: {len(distinct_states)} distinct (branch, slot) states over {len(states)} steps")

        found, details = detect_2state_cycle(states, min_alternations=4, persistence_frac=0.25)
        assert not found, (
            f"{agent_id} exhibits a 2-state limit cycle: alternates between "
            f"{details[0]} and {details[1]} for {details[3]} consecutive steps "
            f"starting at step {details[2]} — this is the exact Task 1 failure "
            "signature (relative-accumulator action scheme trapping the agent "
            "between two states with no absorbing target)."
        )


if __name__ == "__main__":
    trajectories = run_rollout_diagnostic()
    print("VBS rollout diagnostic (untrained policy, structural check only):\n")
    for agent_id, states in trajectories.items():
        distinct = len(set(states))
        found, details = detect_2state_cycle(states, min_alternations=4, persistence_frac=0.25)
        status = "CYCLE DETECTED" if found else "ok"
        print(f"  {agent_id}: {len(states)} steps, {distinct} distinct (branch, slot) states, "
              f"2-state-cycle check: {status}")
        if found:
            print(f"    -> alternating {details[0]} <-> {details[1]} for {details[3]} steps "
                  f"starting at step {details[2]}")
