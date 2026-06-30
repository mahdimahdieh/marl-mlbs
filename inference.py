import torch
import time
import pygame
import os
from rl.envs.pettingzoo_env import CoverageParallelEnv
from rl.agents.ppo_module import HeterogeneousPPOManager
from visualization.pygame_renderer import PygameRenderer
from main import bootstrap_environment


def run_inference(config_path: str, graph_path: str, model_dir: str = None):
    env_config, hp, _ = bootstrap_environment(config_path, graph_path)
    env = CoverageParallelEnv(env_config)
    env.sim_adapter.set_evaluation_mode(True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Derive dims from env
    vbs_agent_id = next(a for a in env.possible_agents if "vbs" in a)
    fbs_agent_id = next(a for a in env.possible_agents if "fbs" in a)

    vbs_obs_dim = env.observation_space(vbs_agent_id).shape[0]
    fbs_obs_dim = env.observation_space(fbs_agent_id).shape[0]
    vbs_action_dim = env.action_space(vbs_agent_id).n
    fbs_action_dim = env.action_space(fbs_agent_id).n

    ppo = HeterogeneousPPOManager(
        vbs_obs_dim=vbs_obs_dim,
        fbs_obs_dim=fbs_obs_dim,
        vbs_action_dim=vbs_action_dim,
        fbs_action_dim=fbs_action_dim,
        lr=0.0,  # Frozen weights for inference
        device=device
    )

    if model_dir:
        vbs_path = os.path.join(model_dir, "vbs_net.pt")
        fbs_path = os.path.join(model_dir, "fbs_net.pt")

        # Guard clause to ensure the files actually exist there
        if os.path.exists(vbs_path) and os.path.exists(fbs_path):
            ppo.vbs_net.load_state_dict(torch.load(vbs_path, map_location=device))
            ppo.fbs_net.load_state_dict(torch.load(fbs_path, map_location=device))
            print(f"Successfully loaded local W&B weights from: {model_dir}")
        else:
            print(f"Error: Weights not found in {model_dir}")
            print("Make sure 'model_vbs.pt' and 'model_fbs.pt' are inside that folder.")
            print("Running with UNTRAINED policy fallback.")
    else:
        print("⚠️ Running with UNTRAINED policy for visualization testing.")

    # 3. Initialize Pygame Renderer
    renderer = PygameRenderer(map_dim=env.map_dim)

    obs_dict, infos_dict = env.reset(seed=42)
    step = 0
    done = False

    print("Starting Inference Loop. Close the Pygame window to stop.")

    while env.agents and not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        actions = {}
        for agent_id in env.agents:
            agent_type = "vbs" if "vbs" in agent_id else "fbs"
            t_obs = torch.tensor(obs_dict[agent_id], dtype=torch.float32).to(device)
            t_mask = torch.tensor(infos_dict[agent_id]["action_mask"], dtype=torch.float32).to(device)

            action, _, _ = ppo.get_action(t_obs, agent_type, action_mask=t_mask)
            actions[agent_id] = action.cpu().item() if hasattr(action, "item") else action

        obs_dict, rewards_dict, terminations, truncations, infos_dict = env.step(actions)
        step += 1

        episode_done = (bool(terminations) and all(terminations.values())) or (
                    bool(truncations) and all(truncations.values()))

        if episode_done:
            eff = env.last_true_coverage   # unique-user coverage, not capacity saturation
            print(f"Episode completed at step {step}. True Coverage: {eff:.2%}")

        renderer.render(env, step)
        time.sleep(0.5)

    pygame.quit()


if __name__ == "__main__":
    # Pointing directly to your local latest-run directory
    LATEST_MODEL = "./models/20260630-090146"

    run_inference(
        config_path="config/simulation_config.json",
        graph_path="config/graph_map.json",
        model_dir=LATEST_MODEL
    )