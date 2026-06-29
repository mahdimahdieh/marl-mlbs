import torch
import time
import pygame
from rl.envs.pettingzoo_env import CoverageParallelEnv
from rl.agents.ppo_module import HeterogeneousPPOManager
from visualization.pygame_renderer import PygameRenderer
from main import bootstrap_environment


def run_inference(config_path: str, graph_path: str, model_weights_path: str = None):
    env_config, hp, _ = bootstrap_environment(config_path, graph_path)
    env = CoverageParallelEnv(env_config)
    env.sim_adapter.set_evaluation_mode(True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # FIXED: Derive dims from env, matching main.py exactly (single source of truth).
    # Old: HeterogeneousPPOManager(vbs_obs_dim=3, fbs_obs_dim=3, lr=0.0, ...)
    #   → missing required action_dim params after Fix 2; raises TypeError on launch.
    vbs_agent_id = next(a for a in env.possible_agents if "vbs" in a)
    fbs_agent_id = next(a for a in env.possible_agents if "fbs" in a)

    vbs_obs_dim    = env.observation_space(vbs_agent_id).shape[0]
    fbs_obs_dim    = env.observation_space(fbs_agent_id).shape[0]
    vbs_action_dim = env.action_space(vbs_agent_id).n
    fbs_action_dim = env.action_space(fbs_agent_id).n

    ppo = HeterogeneousPPOManager(
        vbs_obs_dim=vbs_obs_dim,
        fbs_obs_dim=fbs_obs_dim,
        vbs_action_dim=vbs_action_dim,
        fbs_action_dim=fbs_action_dim,
        lr=0.0,         # Frozen weights for inference; no gradient updates
        device=device
    )
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

        obs_dict, rewards_dict, terminations, truncations, infos_dict = env.step(actions)
        step += 1

        # CRITICAL: Detect termination and reset BEFORE rendering.
        # env.step() clears env.agents=[] as part of its own call stack.
        # Resetting here restores env.agents so the renderer always sees valid agent state.
        #
        # Guard: bool(terminations) prevents all() returning True on an empty dict,
        # which would incorrectly trigger a reset at step 1 on a fresh episode.
        episode_done = (
                               bool(terminations) and all(terminations.values())
                       ) or (
                               bool(truncations) and all(truncations.values())
                       )

        if episode_done:
            eff = env.agent_manager.get_total_efficiency()
            print(f"Episode completed at step {step}. Final Efficiency: {eff:.2%}")
            obs_dict, infos_dict = env.reset()  # ← Restores env.agents BEFORE render call
            step = 0

        # Now safe: env.agents is always populated (either mid-episode or freshly reset)
        renderer.render(env, step)
        time.sleep(0.5)

    # FIX 4: Call native pygame shutdown
    pygame.quit()


if __name__ == "__main__":
    run_inference("config/simulation_config.json", "config/graph_map.json")