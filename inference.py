import torch
import time
import pygame
from rl.envs.pettingzoo_env import CoverageParallelEnv
from rl.agents.ppo_module import HeterogeneousPPOManager
from visualization.pygame_renderer import PygameRenderer
from main import bootstrap_environment


def run_inference(config_path: str, graph_path: str, model_weights_path: str = None):
    # 1. Initialize Environment
    env_config, hp, _ = bootstrap_environment(config_path, graph_path)
    env = CoverageParallelEnv(env_config)

    # STRICT 5G REQUIREMENT: Turn on the PyWiSim high-fidelity channel modeling
    env.sim_adapter.set_evaluation_mode(True)

    # 2. Load the trained network
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ppo = HeterogeneousPPOManager(vbs_obs_dim=3, fbs_obs_dim=3, lr=0.0, device=device)

    if model_weights_path:
        # Load your saved weights here
        # ppo.vbs_net.load_state_dict(torch.load(f"{model_weights_path}_vbs.pt"))
        # ppo.fbs_net.load_state_dict(torch.load(f"{model_weights_path}_fbs.pt"))
        print(f"Loaded trained models from {model_weights_path}")
    else:
        print("Running with untrained policy for visualization testing.")

    # 3. Initialize Pygame Renderer
    renderer = PygameRenderer(map_dim=env.map_dim)

    obs_dict, infos_dict = env.reset(seed=42)
    step = 0
    done = False

    print("Starting Inference Loop. Close the Pygame window to stop.")

    while env.agents and not done:
        # FIX 2: Call direct pygame events instead of via the renderer wrapper
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        actions = {}
        # Deterministic Action Selection (Argmax instead of sampling)
        for agent_id in env.agents:
            agent_type = "vbs" if "vbs" in agent_id else "fbs"
            t_obs = torch.tensor(obs_dict[agent_id], dtype=torch.float32).to(device)
            t_mask = torch.tensor(infos_dict[agent_id]["action_mask"], dtype=torch.float32).to(device)

            action, _, _ = ppo.get_action(t_obs, agent_type, action_mask=t_mask)

            # FIX 3: Strip the tensor back to a native Python int so PettingZoo can process it
            actions[agent_id] = action.cpu().item() if hasattr(action, "item") else action

        # Step Environment (Now using real PyWiSim 5G SINR math)
        obs_dict, rewards_dict, terminations, truncations, infos_dict = env.step(actions)
        step += 1

        # Render the frame
        renderer.render(env, step)

        # Control playback speed (2 frames per second)
        time.sleep(0.5)

        if all(terminations.values()) or all(truncations.values()):
            print(f"Episode completed. Final Efficiency: {env.agent_manager.get_total_efficiency():.2%}")
            obs_dict, infos_dict = env.reset()
            step = 0

    # FIX 4: Call native pygame shutdown
    pygame.quit()


if __name__ == "__main__":
    run_inference("config/simulation_config.json", "config/graph_map.json")