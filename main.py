import json
import torch
import argparse
from typing import Dict, List, Any
import os
import datetime
from collections import deque
import numpy as np

# Core & Infrastructure
from core.entities.agents import AgentManager, VehicleBaseStation, FlyingBaseStation
from infrastructure.graph.networkx_engine import NetworkXRoadEngine
from infrastructure.simulation.pywisim_adapter import PyWiSimAdapter
from infrastructure.tracking.tensorboard_tracker import TensorBoardTracker
from infrastructure.training.determinism import lock_determinism

# RL Layers
from rl.envs.pettingzoo_env import CoverageParallelEnv
from rl.agents.ppo_module import HeterogeneousPPOManager


def bootstrap_environment(config_path: str, graph_path: str):
    """Parses static configurations into the instantiated OOP engines."""
    graph_engine = NetworkXRoadEngine()
    graph_engine.load_from_json(graph_path)

    with open(config_path, "r") as f:
        config = json.load(f)

    manager = AgentManager()

    for v_cfg in config["vbs_agents"]:
        vbs = VehicleBaseStation(
            id=v_cfg["id"],
            capacity=v_cfg["capacity"],
            coverage_radius=v_cfg["coverage_radius"]
        )
        manager.register_vbs(vbs)
    manager.assign_home_branches(
        num_branches=3)  # hardcoded,  bind to config["graph_settings"] if branch count becomes configurable
    manager.assign_identity_indices()

    for f_cfg in config["fbs_agents"]:
        fbs = FlyingBaseStation(
            id=f_cfg["id"],
            host_vbs_id=f_cfg["host_vbs_id"],
            capacity=f_cfg["capacity"],
            coverage_radius=f_cfg["coverage_radius"],
            maximum_distance=f_cfg["maximum_distance"],
        )
        manager.register_fbs(fbs)

    sim_adapter = PyWiSimAdapter(
        num_users=config["env_settings"]["num_users"],
        map_dimensions=graph_engine.get_map_dimension()
    )

    env_config = {
        "agent_manager": manager,
        "graph_engine": graph_engine,
        "sim_adapter": sim_adapter,
        "max_cycles": config["env_settings"]["max_cycles"],
        "termination_goal": config["env_settings"]["termination_goal"],
        # FIXED: Forward graph topology settings from config to the env.
        # Without this, CoverageParallelEnv ignores simulation_config.json entirely
        # for these parameters and falls back to hardcoded defaults on every run.
        "center_node_id": config.get("graph_settings", {}).get("center_node_id", 0),
        "max_slot_per_branch": config.get("graph_settings", {}).get("max_slots_per_branch", 10),
    }

    return env_config, config["hyperparameters"], config


def _broadcast_bootstrap(vals: List[float], n_agents: int) -> List[float]:
    """Normalize a per-agent bootstrap-value list to exactly n_agents entries.

    TODO(scope): CoverageParallelEnv.get_global_state() falls back to a single
    zero-padded row whenever self.agents is empty -- which it always is by the
    time the truncated-episode bootstrap call runs, since
    CoverageParallelEnv.step() clears self.agents before returning on the
    terminating step. That means the "bootstrap from the critic's own V(s_T)"
    estimate (bug ledger #5) is actually V(zeros) whenever an episode ends,
    not the real final observed state -- a pre-existing issue, unrelated to
    the granularity mismatch this function guards against, and out of scope
    for this task. This helper only prevents that pre-existing shape quirk
    from crashing the newly per-agent-indexed bootstrap lookup below.
    """
    if len(vals) == n_agents:
        return vals
    if len(vals) == 0:
        return [0.0] * n_agents
    return [vals[0]] * n_agents


def compute_gae(rewards: List[float], values: List[float], next_value: float, gamma: float = 0.99, lam: float = 0.95):
    """Calculates Generalized Advantage Estimation for stable Critic targets.

    `next_value` MUST be 0.0 only for a true terminal (absorbing) state. For a
    time-limit truncation the trajectory is NOT actually over, and next_value
    should be the critic's own V(s_T) estimate -- see the bootstrap_value
    computation in the training loop below. Bootstrapping every rollout with
    a hardcoded 0.0 regardless of termination vs. truncation was bug ledger
    item #5: it systematically biases the value target toward zero for every
    episode that hits max_cycles without solving the task.
    """
    advantages = []
    last_gae_lam = 0

    # Append next_value for bootstrap calculation at terminal state
    values_extended = values + [next_value]

    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values_extended[step + 1] - values_extended[step]
        last_gae_lam = delta + gamma * lam * last_gae_lam
        advantages.insert(0, last_gae_lam)

    returns = [adv + val for adv, val in zip(advantages, values)]
    return advantages, returns


def _save_models(ppo: "HeterogeneousPPOManager", save_dir: str, episode: int) -> None:
    os.makedirs(save_dir, exist_ok=True)

    torch.save(ppo.vbs_actor.state_dict(), os.path.join(save_dir, "vbs_actor.pt"))
    torch.save(ppo.fbs_actor.state_dict(), os.path.join(save_dir, "fbs_actor.pt"))
    torch.save(ppo.critic.state_dict(), os.path.join(save_dir, "critic.pt"))

    torch.save(ppo.vbs_actor.state_dict(), os.path.join(save_dir, f"vbs_actor_ep{episode}.pt"))
    torch.save(ppo.fbs_actor.state_dict(), os.path.join(save_dir, f"fbs_actor_ep{episode}.pt"))
    torch.save(ppo.critic.state_dict(), os.path.join(save_dir, f"critic_ep{episode}.pt"))

    print(f"Checkpoint saved {save_dir}/*.pt  [ep {episode}]")


def main():
    parser = argparse.ArgumentParser(description="Train VBS/FBS Base Stations")
    parser.add_argument("--config", type=str, default="config/simulation_config.json")
    parser.add_argument("--graph", type=str, default="config/graph_map.json")
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--save-dir", type=str, default="models")
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument("--log-every", type=int, default=10,
                        help="Console print cadence. TensorBoard now logs EVERY episode (see bug ledger #3) regardless of this.")
    parser.add_argument("--seed", type=int, default=42,
                        help="seed for full reproducibility / overfitting baseline.")
    parser.add_argument("--overfit", action="store_true",
                        help="freeze the spatial distribution")
    args = parser.parse_args()

    lock_determinism(args.seed)

    env_config, hp, raw_config = bootstrap_environment(args.config, args.graph)
    env = CoverageParallelEnv(env_config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Booting Training Loop on: {device.upper()}")

    # FIXED: Derive ALL network I/O dimensions from the live env at startup.
    # This makes main.py a passive consumer of env's ground truth — if any config
    # change cascades through to CoverageParallelEnv (e.g., new branch makes VBS Discrete(4)),
    # PPO receives the corrected dims automatically on the next run.
    vbs_agent_id = next(a for a in env.possible_agents if "vbs" in a)
    fbs_agent_id = next(a for a in env.possible_agents if "fbs" in a)

    vbs_obs_dim = env.observation_space(vbs_agent_id).shape[0]
    fbs_obs_dim = env.observation_space(fbs_agent_id).shape[0]
    vbs_action_dim = env.action_space(vbs_agent_id).n
    fbs_action_dim = env.action_space(fbs_agent_id).n
    global_extra_dim = env.global_extra_dim

    print(
        f"Network I/O | VBS: obs={vbs_obs_dim} act={vbs_action_dim}"
        f" | FBS: obs={fbs_obs_dim} act={fbs_action_dim} | Global: {global_extra_dim}"
    )

    ppo = HeterogeneousPPOManager(
        vbs_obs_dim=vbs_obs_dim,
        fbs_obs_dim=fbs_obs_dim,
        vbs_action_dim=vbs_action_dim,
        fbs_action_dim=fbs_action_dim,
        global_extra_dim=global_extra_dim,
        lr=hp["learning_rate"],
        device=device
    )

    tracker = TensorBoardTracker(
        project_name="MARL-Network-Sim",
        config=raw_config,
        run_name="PPO-Y-Graph-MVP"
    )

    data_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    save_dir = os.path.join(args.save_dir, data_time)
    os.makedirs(save_dir, exist_ok=True)

    # BUG LEDGER #3 FIX: these used to be seeded with a single 0.0 and never
    # appended to anywhere in the loop, so every "avg100" value ever printed
    # or logged was sum([0.0]) / 1 == 0.0 for the entire run. They now start
    # empty and are appended to unconditionally, every episode, below.
    reward_window = deque(maxlen=100)
    coverage_window = deque(maxlen=100)
    solve_window = deque(maxlen=100)  # 1.0 if the episode reached the true-coverage goal, else 0.0

    # 2. Training Loop
    for episode in range(1, args.episodes + 1):
        distribution_seed = args.seed if args.overfit else episode
        obs_dict, infos_dict = env.reset(seed=distribution_seed)

        # BUG LEDGER — FIXED (baseline/reward-stream granularity mismatch):
        # vbs_values/fbs_values used to be ONE value per type per step (from
        # CentralizedCritic's old type-pooled heads), trained against the MEAN
        # reward across agents of that type, then reused UNMODIFIED as the GAE
        # baseline for EACH individual agent's own distinct reward trajectory
        # (buffers[agent_type][agent_id]["rewards"]). Since the marginal-
        # contribution reward is explicitly designed to differentiate agents by
        # their individual counterfactual contribution, baselining every agent's
        # distinct reward stream with the same type-mean value systematically
        # biased the advantage estimate whenever agents within a type diverge in
        # behavior — the expected, desired outcome of the reward design.
        #
        # FIX (Option A — agent-conditioned value heads, see CentralizedCritic):
        # vbs_head/fbs_head now output one value PER AGENT per step. Each agent's
        # own value trajectory is captured directly into buffers[type][agent_id]
        # ["values"], at the same per-step granularity as its own "rewards" — no
        # separate type-level joint_buffer arrays needed for vbs/fbs anymore.
        # agent order is captured once, up front, because CoverageParallelEnv
        # keeps self.agents (and therefore env.get_global_state()'s stacking
        # order) fixed for the full episode until the final terminating step.
        vbs_agent_order = [a for a in env.agents if "vbs" in a]
        fbs_agent_order = [a for a in env.agents if "fbs" in a]

        # Isolated Buffers to prevent weight contamination
        buffers = {
            "vbs": {agent: {"obs": [], "actions": [], "logprobs": [], "rewards": [], "masks": [], "values": []} for
                    agent in env.agents if "vbs" in agent},
            "fbs": {agent: {"obs": [], "actions": [], "logprobs": [], "rewards": [], "masks": [], "values": []} for
                    agent in env.agents if "fbs" in agent}
        }
        episode_reward = 0.0

        # --- ROLLOUT PHASE ---
        # team_* stays scalar-per-step (team_head is unchanged); vbs/fbs returns
        # and values are now assembled AFTER the rollout from buffers[type][*]["values"]/
        # ["rewards"], at per-agent granularity (see BUG LEDGER above).
        joint_buffer = {
            "vbs_feats": [],
            "fbs_feats": [],
            "global_extra": [],
            "team_values": [],
            "team_rewards": [],
        }
        terminations: Dict[str, bool] = {}
        truncations: Dict[str, bool] = {}

        while env.agents:
            actions = {}
            for agent_id in env.agents:
                agent_type = "vbs" if "vbs" in agent_id else "fbs"
                t_obs = torch.tensor(obs_dict[agent_id], dtype=torch.float32)
                t_mask = torch.tensor(infos_dict[agent_id]["action_mask"], dtype=torch.float32)

                action, logprob = ppo.get_action(t_obs, agent_type, action_mask=t_mask)  # 2-tuple, not 3
                actions[agent_id] = action

                buffers[agent_type][agent_id]["obs"].append(t_obs)
                buffers[agent_type][agent_id]["masks"].append(t_mask)
                buffers[agent_type][agent_id]["actions"].append(action)
                buffers[agent_type][agent_id]["logprobs"].append(logprob)

            vbs_feats, fbs_feats, global_extra = env.get_global_state()
            step_values = ppo.get_value(
                torch.tensor(vbs_feats, dtype=torch.float32).unsqueeze(0),
                torch.tensor(fbs_feats, dtype=torch.float32).unsqueeze(0),
                torch.tensor(global_extra, dtype=torch.float32).unsqueeze(0),
            )
            joint_buffer["vbs_feats"].append(vbs_feats)
            joint_buffer["fbs_feats"].append(fbs_feats)
            joint_buffer["global_extra"].append(global_extra)
            joint_buffer["team_values"].append(step_values["team"])

            # Per-agent value, at the SAME granularity as the per-agent reward
            # each agent will receive this same step (fix core: no more type-mean
            # value standing in for an individual agent's baseline).
            for i, agent_id in enumerate(vbs_agent_order):
                buffers["vbs"][agent_id]["values"].append(step_values["vbs"][i])
            for i, agent_id in enumerate(fbs_agent_order):
                buffers["fbs"][agent_id]["values"].append(step_values["fbs"][i])

            next_obs_dict, rewards_dict, terminations, truncations, next_infos_dict = env.step(actions)

            for agent_id in actions.keys():
                agent_type = "vbs" if "vbs" in agent_id else "fbs"
                buffers[agent_type][agent_id]["rewards"].append(rewards_dict[agent_id])
                episode_reward += rewards_dict[agent_id]

            joint_buffer["team_rewards"].append(float(np.mean(list(rewards_dict.values()))))

            obs_dict = next_obs_dict
            infos_dict = next_infos_dict

        # --- OPTIMIZATION PHASE ---
        # Compile global metrics
        #
        # CRITICAL: env.last_true_coverage is the unique-user set-union metric
        # computed in CoverageParallelEnv.step() Phase 3:
        #     any_covered_mask = np.any(coverage_matrix, axis=0)   # per-USER, not per-station
        #     true_coverage = unique_users_covered / total_users
        #
        # agent_manager.get_total_efficiency() is a DIFFERENT metric — per-station
        # capacity saturation (sum(min(count, capacity)) / sum(capacity)). It
        # double-counts users seen by multiple overlapping stations and clamps
        # each station's contribution at its own capacity, so it can read 100%
        # while only a small fraction of the actual user population is covered.
        # That is what was producing the 20%-covered-but-100%-reported symptom.
        final_efficiency = env.last_true_coverage

        # BUG LEDGER #5 FIX: distinguish a true terminal state from a
        # time-limit truncation. compute_gae used to always bootstrap with
        # next_value=0.0, which is only correct for the terminated case — a
        # truncated episode is not actually over, so we bootstrap it with the
        # critic's own value estimate at the final observed state instead.
        episode_terminated = bool(terminations) and all(terminations.values())
        episode_truncated = bool(truncations) and all(truncations.values()) and not episode_terminated

        if episode_truncated:
            f_vbs, f_fbs, f_extra = env.get_global_state()
            bootstrap_value = ppo.get_value(
                torch.tensor(f_vbs, dtype=torch.float32).unsqueeze(0),
                torch.tensor(f_fbs, dtype=torch.float32).unsqueeze(0),
                torch.tensor(f_extra, dtype=torch.float32).unsqueeze(0),
            )
        else:
            bootstrap_value = {
                "team": 0.0,
                "vbs": [0.0] * len(vbs_agent_order),
                "fbs": [0.0] * len(fbs_agent_order),
            }
        # See _broadcast_bootstrap's TODO(scope) docstring: normalizes whatever
        # shape ppo.get_value()/env.get_global_state() produced back to exactly
        # one bootstrap scalar per agent, defensively (does not fix the
        # pre-existing get_global_state() zero-fallback quirk itself).
        bootstrap_vbs = _broadcast_bootstrap(bootstrap_value["vbs"], len(vbs_agent_order))
        bootstrap_fbs = _broadcast_bootstrap(bootstrap_value["fbs"], len(fbs_agent_order))

        batch_vbs = {"obs": [], "actions": [], "logprobs": [], "advantages": [], "masks": []}
        batch_fbs = {"obs": [], "actions": [], "logprobs": [], "advantages": [], "masks": []}

        # BUG LEDGER — FIXED: baseline (buffers[type][agent]["values"]) and reward
        # stream (buffers[type][agent]["rewards"]) are now the SAME agent's own
        # trajectory at the SAME per-step granularity — no more type-mean value
        # standing in for an individual agent's distinct marginal-contribution
        # reward. Also collects the per-agent (returns, values) needed for the
        # critic update below, transposed into (T, n_agents) matrices aligned
        # with vbs_agent_order/fbs_agent_order.
        vbs_returns_by_agent, vbs_values_by_agent = [], []
        for i, agent_id in enumerate(vbs_agent_order):
            data = buffers["vbs"][agent_id]
            if len(data["rewards"]) == 0:
                continue
            advs, rets = compute_gae(data["rewards"], data["values"], next_value=bootstrap_vbs[i])
            batch_vbs["obs"].extend(data["obs"])
            batch_vbs["masks"].extend(data["masks"])
            batch_vbs["actions"].extend(data["actions"])
            batch_vbs["logprobs"].extend(data["logprobs"])
            batch_vbs["advantages"].extend(advs)
            vbs_returns_by_agent.append(rets)
            vbs_values_by_agent.append(data["values"])

        fbs_returns_by_agent, fbs_values_by_agent = [], []
        for i, agent_id in enumerate(fbs_agent_order):
            data = buffers["fbs"][agent_id]
            if len(data["rewards"]) == 0:
                continue
            advs, rets = compute_gae(data["rewards"], data["values"], next_value=bootstrap_fbs[i])
            batch_fbs["obs"].extend(data["obs"])
            batch_fbs["masks"].extend(data["masks"])
            batch_fbs["actions"].extend(data["actions"])
            batch_fbs["logprobs"].extend(data["logprobs"])
            batch_fbs["advantages"].extend(advs)
            fbs_returns_by_agent.append(rets)
            fbs_values_by_agent.append(data["values"])

        if len(batch_vbs["obs"]) > 0:
            ppo.update_actor(batch_vbs, "vbs", clip_coef=hp["clip_coef"], ent_coef=hp["ent_coef"],
                             ppo_epochs=hp["ppo_epochs"], batch_size=hp["batch_size"])
        if len(batch_fbs["obs"]) > 0:
            ppo.update_actor(batch_fbs, "fbs", clip_coef=hp["clip_coef"], ent_coef=hp["ent_coef"],
                             ppo_epochs=hp["ppo_epochs"], batch_size=hp["batch_size"])

        # Critic: team_head keeps its own scalar-per-step GAE pass, unchanged.
        # vbs/fbs are transposed from (n_agents, T) to (T, n_agents) so each row
        # lines up with the corresponding vbs_feats/fbs_feats row (one per
        # environment step) that update_critic's forward pass consumes — see
        # CentralizedCritic/update_critic BUG LEDGER for why this is now a
        # per-agent-column target instead of a single type-mean column.
        _, team_returns = compute_gae(joint_buffer["team_rewards"], joint_buffer["team_values"],
                                      next_value=bootstrap_value["team"])
        vbs_returns = [list(row) for row in zip(*vbs_returns_by_agent)] if vbs_returns_by_agent else []
        vbs_values = [list(row) for row in zip(*vbs_values_by_agent)] if vbs_values_by_agent else []
        fbs_returns = [list(row) for row in zip(*fbs_returns_by_agent)] if fbs_returns_by_agent else []
        fbs_values = [list(row) for row in zip(*fbs_values_by_agent)] if fbs_values_by_agent else []

        ppo.update_critic({
            "vbs_feats": joint_buffer["vbs_feats"],
            "fbs_feats": joint_buffer["fbs_feats"],
            "global_extra": joint_buffer["global_extra"],
            "team_values": joint_buffer["team_values"], "team_returns": team_returns,
            "vbs_values": vbs_values, "vbs_returns": vbs_returns,
            "fbs_values": fbs_values, "fbs_returns": fbs_returns,
        }, vf_coef=hp["vf_coef"], ppo_epochs=hp["ppo_epochs"], batch_size=hp["batch_size"])

        # --- LOGGING PHASE ---
        # Rolling windows are now updated EVERY episode (bug ledger #3), not
        # just on the episodes we happen to print. TensorBoard also now
        # receives a scalar dict every episode instead of every 10th — full
        # resolution data for TB's own smoothing slider to work with. Console
        # printing stays at --log-every to avoid flooding stdout.
        reward_window.append(episode_reward)
        coverage_window.append(final_efficiency)
        solve_window.append(1.0 if episode_terminated else 0.0)

        roll_reward_mean = sum(reward_window) / len(reward_window)
        roll_coverage_mean = sum(coverage_window) / len(coverage_window)
        roll_solve_rate = sum(solve_window) / len(solve_window)

        metrics = {
            # Grouped by TensorBoard tag prefix so the UI folds them into sections.
            "Reward/Episode": episode_reward,
            # Raw episode_reward is a sum over a variable-length rollout, so it's
            # not comparable across episodes of different lengths (see bug ledger
            # #6) — Per_Step gives a length-normalized view alongside it.
            "Reward/Per_Step": (episode_reward / env.step_count) if env.step_count else 0.0,
            "Reward/Rolling100": roll_reward_mean,
            "Coverage/Episode": final_efficiency,
            "Coverage/Rolling100": roll_coverage_mean,
            "Coverage/SolveRate_Rolling100": roll_solve_rate,
            "Diagnostics/Episode_Length": env.step_count,
            "Diagnostics/Capacity_Utilization": env_config["agent_manager"].get_capacity_utilization(),
            "Diagnostics/Truncated": 1.0 if episode_truncated else 0.0,
        }
        tracker.log_episode(metrics, step=episode)

        if episode % args.log_every == 0:
            print(
                f"Episode: {episode:4d} | "
                f"Coverage: {final_efficiency:.2%} (avg100: {roll_coverage_mean:.2%}) | "
                f"Reward: {episode_reward:.2f} (avg100: {roll_reward_mean:.2f}) | "
                f"Solve100: {roll_solve_rate:.1%} | "
                f"Length: {env.step_count}"
            )
        if episode % args.save_every == 0:
            _save_models(ppo, save_dir, episode)

    _save_models(ppo, save_dir, args.episodes)  # Final save regardless of cadence
    print("Training Complete. Models saved. Run inference.py to visualize.")
    tracker.close()
    print("Training Complete. Models ready for PyWiSim Evaluation.")


if __name__ == "__main__":
    main()