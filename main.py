import json
import numpy as np
import torch
import argparse
from typing import Dict, List, Any

# Core & Infrastructure
from core.entities.agents import AgentManager, VehicleBaseStation, FlyingBaseStation
from infrastructure.graph.networkx_engine import NetworkXRoadEngine
from infrastructure.simulation.pywisim_adapter import PyWiSimAdapter
from infrastructure.tracking.wandb_tracker import WandbTracker

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

    for f_cfg in config["fbs_agents"]:
        fbs = FlyingBaseStation(
            id=f_cfg["id"],
            host_vbs_id=f_cfg["host_vbs_id"],
            capacity=f_cfg["capacity"],
            coverage_radius=f_cfg["coverage_radius"]
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
        "max_cycles": config["env_settings"]["max_cycles"]
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


def main():
    parser = argparse.ArgumentParser(description="Train VBS/FBS Base Stations")
    parser.add_argument("--config", type=str, default="config/simulation_config.json")
    parser.add_argument("--graph", type=str, default="config/graph_map.json")
    parser.add_argument("--episodes", type=int, default=5000)
    args = parser.parse_args()

    # 1. Initialize System Architecture
    env_config, hp, raw_config = bootstrap_environment(args.config, args.graph)
    env = CoverageParallelEnv(env_config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Booting Training Loop on: {device.upper()}")

    # Observation space is 3 [norm_x, norm_y, cap_ratio] for both.
    ppo = HeterogeneousPPOManager(vbs_obs_dim=3, fbs_obs_dim=3, lr=hp["learning_rate"], device=device)

    tracker = WandbTracker(project_name="MARL-Network-Sim", config=raw_config, run_name="PPO-Y-Graph-MVP")

    # 2. Training Loop
    for episode in range(1, args.episodes + 1):
        obs_dict, infos_dict = env.reset(seed=episode)

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
        final_efficiency = env_config["agent_manager"].get_total_efficiency()

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

        # --- LOGGING PHASE ---
        if episode % 10 == 0:
            metrics = {
                "Episode_Reward": episode_reward,
                "System_Efficiency": final_efficiency,
                "Episode_Length": env.step_count
            }
            tracker.log_episode(metrics, step=episode)
            print(f"Episode: {episode} | Eff: {final_efficiency:.2%} | Reward: {episode_reward:.2f}")

    tracker.close()
    print("Training Complete. Models ready for PyWiSim Evaluation.")


if __name__ == "__main__":
    main()