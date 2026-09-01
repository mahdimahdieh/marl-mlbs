"""Greedy-placement ceiling probe.

Hill-climbs the ABSOLUTE placement space (VBS (branch, slot) x FBS offset
zones) — single-agent moves plus joint (host VBS, tethered FBS) pair moves —
with restarts, estimating a LOWER BOUND on the maximum reachable true
coverage for the current config + graph. Exits non-zero when
termination_goal exceeds the probe ceiling (goal provably unreachable).

Usage:
    python tools/ceiling_probe.py [--restarts 3] [--seed 0]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from main import bootstrap_environment  # noqa: E402
from rl.envs.pettingzoo_env import CoverageParallelEnv  # noqa: E402


def _row_for(env, agent_id, action, host_action_override=None, cache=None):
    """Pure (M,) coverage-row preview for one candidate placement. Reuses the
    env's own coordinate physics so the probe can never drift from it."""
    if cache is not None:
        key = (agent_id, int(action),
               int(host_action_override) if host_action_override is not None else None)
        if key in cache:
            return cache[key]

    obj, is_vbs = env._get_agent_obj(agent_id)
    if is_vbs:
        branch_id, slot_index = env._decode_vbs_action(int(action))
        saved = (obj.current_branch_id, obj.current_slot_index)
        obj.current_branch_id, obj.current_slot_index = branch_id, slot_index
        x, y = env._calculate_world_coords(obj, True)
        obj.current_branch_id, obj.current_slot_index = saved
    else:
        host = env.agent_manager.vbs_registry[obj.host_vbs_id]
        saved_host = (host.current_branch_id, host.current_slot_index)
        saved_zone = obj.current_offset_zone
        if host_action_override is not None:
            host.current_branch_id, host.current_slot_index = env._decode_vbs_action(
                int(host_action_override))
        obj.current_offset_zone = int(action)
        x, y = env._calculate_world_coords(obj, False)
        obj.current_offset_zone = saved_zone
        host.current_branch_id, host.current_slot_index = saved_host

    row = env.sim_adapter.compute_coverage_matrix(
        np.array([[x, y]], dtype=np.float32),
        np.array([obj.coverage_radius], dtype=np.float32),
    )[0]

    if cache is not None:
        cache[key] = row
    return row


def _apply_placement(env, placements):
    for agent_id, action in placements.items():
        obj, is_vbs = env._get_agent_obj(agent_id)
        if is_vbs:
            obj.current_branch_id, obj.current_slot_index = env._decode_vbs_action(int(action))
        else:
            obj.current_offset_zone = int(action)


def _current_rows(env, agent_ids, radii):
    coords = []
    for a in agent_ids:
        obj, is_vbs = env._get_agent_obj(a)
        coords.append(env._calculate_world_coords(obj, is_vbs))
    matrix = env.sim_adapter.compute_coverage_matrix(
        np.array(coords, dtype=np.float32), radii)
    return [matrix[i] for i in range(len(agent_ids))]


def _union_count(rows):
    u = np.zeros(rows[0].shape[0], dtype=bool)
    for r in rows:
        u |= r
    return int(np.count_nonzero(u))


def hill_climb(env, init_placements, radii, max_rounds=50, cache=None):
    """Greedy improvement from one initial placement until convergence.

    Phase 1 sweeps single-agent best moves (a VBS move drags its tethered
    FBS, whose coverage rows are recomputed so the union count always
    matches the true resulting coverage). Phase 2 sweeps joint
    (host VBS, FBS zone) pair moves for the tether coupling single-agent
    greedy cannot traverse in one step.
    """
    agent_ids = list(init_placements.keys())
    placements = dict(init_placements)
    _apply_placement(env, placements)
    rows = _current_rows(env, agent_ids, radii)
    num_users = env.sim_adapter.num_users

    hosted_fbs = {}
    for a in agent_ids:
        obj, is_vbs = env._get_agent_obj(a)
        if not is_vbs:
            hosted_fbs.setdefault(f"vbs_{obj.host_vbs_id}", []).append(a)

    def row(agent_id, action, host_override=None):
        return _row_for(env, agent_id, action, host_action_override=host_override, cache=cache)

    for _ in range(max_rounds):
        improved = False

        # Phase 1: single-agent best moves.
        for i, agent_id in enumerate(agent_ids):
            obj, is_vbs = env._get_agent_obj(agent_id)
            n_actions = env.action_space(agent_id).n
            affected = [agent_id] + (hosted_fbs.get(agent_id, []) if is_vbs else [])
            affected_idx = [agent_ids.index(a) for a in affected]

            others_union = np.zeros(num_users, dtype=bool)
            for j, r in enumerate(rows):
                if j not in affected_idx:
                    others_union |= r
            current_union = others_union.copy()
            for j in affected_idx:
                current_union |= rows[j]
            best_count = int(np.count_nonzero(current_union))
            best_action = placements[agent_id]
            best_rows = None

            for action in range(n_actions):
                if action == placements[agent_id]:
                    continue
                cand_rows = [row(agent_id, action)]
                if is_vbs:
                    for f in hosted_fbs.get(agent_id, []):
                        cand_rows.append(row(f, placements[f], host_override=action))
                u = others_union.copy()
                for r in cand_rows:
                    u |= r
                c = int(np.count_nonzero(u))
                if c > best_count:
                    best_count, best_action, best_rows = c, action, cand_rows

            if best_action != placements[agent_id]:
                placements[agent_id] = best_action
                _apply_placement(env, placements)
                for idx, a in enumerate(affected):
                    rows[agent_ids.index(a)] = best_rows[idx]
                improved = True

        # Phase 2: joint (host VBS, tethered FBS) pair moves.
        for vbs_id, fbs_list in hosted_fbs.items():
            if not fbs_list:
                continue
            affected = [vbs_id] + fbs_list
            affected_idx = [agent_ids.index(a) for a in affected]

            others_union = np.zeros(num_users, dtype=bool)
            for j, r in enumerate(rows):
                if j not in affected_idx:
                    others_union |= r
            current_union = others_union.copy()
            for j in affected_idx:
                current_union |= rows[j]
            best_count = int(np.count_nonzero(current_union))
            best_assignment = None

            n_vbs_actions = env.action_space(vbs_id).n
            n_fbs_actions = env.action_space(fbs_list[0]).n

            for v_act in range(n_vbs_actions):
                v_row = row(vbs_id, v_act)
                base_f_rows = {f: row(f, placements[f], host_override=v_act)
                               for f in fbs_list}

                u = others_union | v_row
                for r in base_f_rows.values():
                    u |= r
                c = int(np.count_nonzero(u))
                if c > best_count:
                    best_count = c
                    best_assignment = (v_act, {f: placements[f] for f in fbs_list})

                for f in fbs_list:
                    for zone in range(n_fbs_actions):
                        if zone == placements[f]:
                            continue
                        u = others_union | v_row
                        for f2 in fbs_list:
                            u |= base_f_rows[f2] if f2 != f else row(f, zone, host_override=v_act)
                        c = int(np.count_nonzero(u))
                        if c > best_count:
                            best_count = c
                            zones = {f2: placements[f2] for f2 in fbs_list}
                            zones[f] = zone
                            best_assignment = (v_act, zones)

            if best_assignment is not None:
                v_act, zones = best_assignment
                placements[vbs_id] = v_act
                for f, z in zones.items():
                    placements[f] = z
                _apply_placement(env, placements)
                rows = _current_rows(env, agent_ids, radii)
                improved = True

        if not improved:
            break

    return _union_count(rows) / float(max(num_users, 1))


def estimate_ceiling(env, restarts=3, seed=0, max_rounds=50):
    """Greedy ceiling lower-bound: one deterministic cold-start climb
    (everything at center/hover) plus `restarts` random-init climbs."""
    rng = np.random.default_rng(seed)
    agent_ids = list(env.possible_agents)
    radii = np.array(
        [env._get_agent_obj(a)[0].coverage_radius for a in agent_ids], dtype=np.float32)

    cache = {}
    inits = [{a: 0 for a in agent_ids}]
    for _ in range(restarts):
        inits.append({a: int(rng.integers(env.action_space(a).n)) for a in agent_ids})

    best = 0.0
    for init in inits:
        best = max(best, hill_climb(env, init, radii, max_rounds=max_rounds, cache=cache))
    return best


def main():
    parser = argparse.ArgumentParser(description="Greedy placement ceiling probe")
    parser.add_argument("--config", type=str, default="config/simulation_config.json")
    parser.add_argument("--graph", type=str, default="config/graph_map.json")
    parser.add_argument("--restarts", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    env_config, _, _ = bootstrap_environment(args.config, args.graph)
    env = CoverageParallelEnv(env_config)
    env.reset(seed=args.seed)  # fixes the user distribution

    ceiling = estimate_ceiling(env, restarts=args.restarts, seed=args.seed)
    goal = env.termination_goal
    margin = ceiling - goal

    print(f"Greedy ceiling estimate : {ceiling:.4f}")
    print(f"termination_goal        : {goal:.4f}")
    print(f"Margin                  : {margin:+.4f}")
    if goal <= ceiling:
        print("PASS: goal is within the greedy reach (goal is achievable by construction)")
        return 0
    print("FAIL: termination_goal exceeds the greedy ceiling — lower the goal or "
          "raise agent count/radii")
    return 1


if __name__ == "__main__":
    sys.exit(main())
