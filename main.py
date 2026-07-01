import json
import torch
import argparse
from typing import Dict, List, Any
import os
import datetime
from collections import deque

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
    manager.assign_home_branches(num_branches=3) #  hardcoded,  bind to config["graph_settings"] if branch count becomes configurable

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
        # FIXED: Forward graph topology settings from config to the env.
        # Without this, CoverageParallelEnv ignores simulation_config.json entirely
        # for these parameters and falls back to hardcoded defaults on every run.
        "center_node_id": config.get("graph_settings", {}).get("center_node_id", 0),
        "max_slots_per_branch": config.get("graph_settings", {}).get("max_slots_per_branch", 10),
    }

    return env_config, config["hyperparameters"], config


def compute_gae(rewards: List[float], values: List[float], next_value: float, gamma: float = 0.99, lam: float = 0.95):
    """Calculates Generalized Advantage Estimation for stable Critic targets."""
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
    """
    Persists both network state-dicts.

    Two saves per call:
      *_net.pt          — always-current 'latest' snapshot; inference.py default target
      *_net_epN.pt      — tagged backup so we can roll back if reward spikes mid-training

    Using state_dict() (not the full model) keeps files small and avoids
    class-path binding issues when the codebase changes between checkpoint and load.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Latest snapshot (overwritten every checkpoint)
    torch.save(ppo.vbs_net.state_dict(), os.path.join(save_dir, "vbs_net.pt"))
    torch.save(ppo.fbs_net.state_dict(), os.path.join(save_dir, "fbs_net.pt"))

    # Tagged backup for rollback
    torch.save(ppo.vbs_net.state_dict(), os.path.join(save_dir, f"vbs_net_ep{episode}.pt"))
    torch.save(ppo.fbs_net.state_dict(), os.path.join(save_dir, f"fbs_net_ep{episode}.pt"))

    print(f"Checkpoint saved {save_dir}/*net.pt  [ep {episode}]")


def main():
    parser = argparse.ArgumentParser(description="Train VBS/FBS Base Stations")
    parser.add_argument("--config", type=str, default="config/simulation_config.json")
    parser.add_argument("--graph", type=str, default="config/graph_map.json")
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--save-dir",   type=str, default="models")
    parser.add_argument("--save-every", type=int, default=250)
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
    os.mkdir(save_dir)

    reward_window = deque(maxlen=100)  # last 100 episodes
    coverage_window = deque(maxlen=100)

    reward_window.append(0.)
    coverage_window.append(0.)


    # 2. Training Loop
    for episode in range(1, args.episodes + 1):
        distribution_seed = args.seed if args.overfit else episode
        obs_dict, infos_dict = env.reset(seed=distribution_seed)
        
        # Isolated Buffers to prevent weight contamination
        buffers = {
            "vbs": {agent: {"obs": [], "actions": [], "logprobs": [], "rewards": [], "values": [], "masks": []} for
                    agent in env.agents if "vbs" in agent},
            "fbs": {agent: {"obs": [], "actions": [], "logprobs": [], "rewards": [], "values": [], "masks": []} for
                    agent in env.agents if "fbs" in agent}
        }

        episode_reward = 0.0

        # --- ROLLOUT PHASE ---
        while env.agents:
            actions = {}
            # Action Selection Loop
            for agent_id in env.agents:
                agent_type = "vbs" if "vbs" in agent_id else "fbs"

                # Extract strict NumPy arrays and convert for PyTorch inference
                t_obs = torch.tensor(obs_dict[agent_id], dtype=torch.float32)
                t_mask = torch.tensor(infos_dict[agent_id]["action_mask"], dtype=torch.float32)

                # Execute inference
                action, logprob, value = ppo.get_action(t_obs, agent_type, action_mask=t_mask)
                actions[agent_id] = action

                # Store State Transition
                buffers[agent_type][agent_id]["obs"].append(t_obs)
                buffers[agent_type][agent_id]["masks"].append(t_mask)
                buffers[agent_type][agent_id]["actions"].append(action)
                buffers[agent_type][agent_id]["logprobs"].append(logprob)
                buffers[agent_type][agent_id]["values"].append(value)

            vbs_feats, fbs_feats, global_extra = env.get_global_state()
            step_value = ppo.get_value(
                torch.tensor(vbs_feats, dtype=torch.float32).unsqueeze(0),
                torch.tensor(fbs_feats, dtype=torch.float32).unsqueeze(0),
                torch.tensor(global_extra, dtype=torch.float32).unsqueeze(0),
            )
            for agent_id in actions.keys():
                agent_type = "vbs" if "vbs" in agent_id else "fbs"
                buffers[agent_type][agent_id]["values"].append(step_value)

            # Step the parallel environment
            next_obs_dict, rewards_dict, terminations, truncations, next_infos_dict = env.step(actions)

            # FIX: Loop over actions.keys() instead of env.agents to guarantee
            # rewards are captured for agents that terminated/truncated during this step.
            for agent_id in actions.keys():
                agent_type = "vbs" if "vbs" in agent_id else "fbs"
                buffers[agent_type][agent_id]["rewards"].append(rewards_dict[agent_id])
                episode_reward += rewards_dict[agent_id]

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
        final_efficiency = env.last_true_coverage  # was: env_config["agent_manager"].get_total_efficiency()

        batch_vbs = {"obs": [], "actions": [], "logprobs": [], "advantages": [], "returns": [], "values": [],
                     "masks": []}
        batch_fbs = {"obs": [], "actions": [], "logprobs": [], "advantages": [], "returns": [], "values": [],
                     "masks": []}

        # Calculate GAE for every agent independently
        for agent_type in ["vbs", "fbs"]:
            target_batch = batch_vbs if agent_type == "vbs" else batch_fbs
            for agent_id, data in buffers[agent_type].items():
                if len(data["rewards"]) == 0:
                    continue

                # Terminal value is 0.0 because the episode strictly ends on max_cycles or target hit
                advs, rets = compute_gae(data["rewards"], data["values"], next_value=0.0)

                target_batch["obs"].extend(data["obs"])
                target_batch["masks"].extend(data["masks"])
                target_batch["actions"].extend(data["actions"])
                target_batch["logprobs"].extend(data["logprobs"])
                target_batch["values"].extend(data["values"])
                target_batch["advantages"].extend(advs)
                target_batch["returns"].extend(rets)

        # Execute isolated weight updates
        if len(batch_vbs["obs"]) > 0:
            ppo.update_policy(batch_vbs, "vbs", clip_coef=hp["clip_coef"], ent_coef=hp["ent_coef"],
                              vf_coef=hp["vf_coef"], ppo_epochs=hp["ppo_epochs"], batch_size=hp["batch_size"])

        if len(batch_fbs["obs"]) > 0:
            ppo.update_policy(batch_fbs, "fbs", clip_coef=hp["clip_coef"], ent_coef=hp["ent_coef"],
                              vf_coef=hp["vf_coef"], ppo_epochs=hp["ppo_epochs"], batch_size=hp["batch_size"])

        reward_window.append(episode_reward)
        coverage_window.append(final_efficiency)

        # --- LOGGING PHASE ---
        if episode % 10 == 0:
            roll_reward_mean = sum(reward_window) / len(reward_window)
            roll_coverage_mean = sum(coverage_window) / len(coverage_window)

            metrics = {
                "Episode_Reward": episode_reward,
                # True network coverage: unique users covered / total users
                "True_Coverage": final_efficiency,
                # Capacity utilization diagnostic (should be high but is NOT the objective)
                "Capacity_Utilization": env_config["agent_manager"].get_capacity_utilization(),
                "Episode_Length": env.step_count,
                "Rolling100_Reward": roll_reward_mean,
                "Rolling100_Coverage": roll_coverage_mean,
            }
            tracker.log_episode(metrics, step=episode)
            print(
                f"Episode: {episode:4d} | "
                f"Coverage: {final_efficiency:.2%} (avg100: {roll_coverage_mean:.2%}) | "
                f"Reward: {episode_reward:.2f} (avg100: {roll_reward_mean:.2f}) | "
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