"""
Automated multi-agent inference / evaluation harness for the VBS/FBS coverage
task. Deterministic evaluation by default, multi-episode batch statistics,
optional rendering / frame dumps, and a JSON summary for CI regression checks.

Usage:
    python inference.py --model-dir models/20260702-220816 --episodes 20 --headless
    python inference.py --episodes 5 --summary-json runs/eval_summary.json
"""

import argparse
import glob
import json
import os
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from infrastructure.training.determinism import lock_determinism
from main import bootstrap_environment
from rl.agents.ppo_module import HeterogeneousPPOManager
from rl.envs.pettingzoo_env import CoverageParallelEnv


# --------------------------------------------------------------------------- #
# Result containers
# --------------------------------------------------------------------------- #

@dataclass
class EpisodeResult:
    episode_index: int
    seed: int
    steps: int
    true_coverage: float
    total_reward: float
    mean_reward_per_step: float
    terminated: bool  # reached the coverage goal
    truncated: bool   # hit max_cycles without reaching it


@dataclass
class EvalSummary:
    episodes: List[EpisodeResult] = field(default_factory=list)

    def add(self, ep: EpisodeResult) -> None:
        self.episodes.append(ep)

    def to_dict(self) -> Dict[str, Any]:
        cov = [e.true_coverage for e in self.episodes]
        rew = [e.total_reward for e in self.episodes]
        length = [e.steps for e in self.episodes]
        solved = [e for e in self.episodes if e.terminated]
        return {
            "num_episodes": len(self.episodes),
            "solve_rate": len(solved) / len(self.episodes) if self.episodes else 0.0,
            "coverage_mean": statistics.mean(cov) if cov else 0.0,
            "coverage_std": statistics.pstdev(cov) if len(cov) > 1 else 0.0,
            "reward_mean": statistics.mean(rew) if rew else 0.0,
            "reward_std": statistics.pstdev(rew) if len(rew) > 1 else 0.0,
            "length_mean": statistics.mean(length) if length else 0.0,
            "episodes": [e.__dict__ for e in self.episodes],
        }


# --------------------------------------------------------------------------- #
# Model / checkpoint handling
# --------------------------------------------------------------------------- #

def discover_latest_model_dir(models_root: str) -> Optional[str]:
    """Returns the most recent (lexicographically-last, given the
    YYYYMMDD-HHMMSS naming) subdirectory under models_root."""
    candidates = sorted(c for c in glob.glob(os.path.join(models_root, "*")) if os.path.isdir(c))
    return candidates[-1] if candidates else None


def load_policy(
    ppo: HeterogeneousPPOManager, model_dir: Optional[str], device: str, allow_untrained: bool
) -> None:
    if model_dir is None:
        if not allow_untrained:
            raise FileNotFoundError(
                "No model directory found or provided. Pass --model-dir explicitly, "
                "populate --models-root with a run, or pass --allow-untrained to "
                "evaluate a fresh (random) policy on purpose."
            )
        print("[warn] No checkpoint supplied -- running an UNTRAINED policy.")
        return

    vbs_path = os.path.join(model_dir, "vbs_actor.pt")
    fbs_path = os.path.join(model_dir, "fbs_actor.pt")

    if not (os.path.exists(vbs_path) and os.path.exists(fbs_path)):
        if allow_untrained:
            print(f"[warn] Checkpoint files missing in {model_dir} -- running UNTRAINED policy.")
            return
        raise FileNotFoundError(
            f"Expected checkpoint files not found:\n  {vbs_path}\n  {fbs_path}\n"
            f"Pass --allow-untrained to evaluate a fresh policy anyway."
        )

    ppo.vbs_actor.load_state_dict(torch.load(vbs_path, map_location=device, weights_only=True))
    ppo.fbs_actor.load_state_dict(torch.load(fbs_path, map_location=device, weights_only=True))
    print(f"[ok] Loaded policy weights from: {model_dir}")


# --------------------------------------------------------------------------- #
# Single-episode rollout
# --------------------------------------------------------------------------- #

def run_episode(
    env: CoverageParallelEnv,
    ppo: HeterogeneousPPOManager,
    device: str,
    seed: int,
    episode_index: int,
    deterministic: bool,
    renderer: Optional[Any],
    step_delay: float,
    frame_dir: Optional[str],
) -> EpisodeResult:
    pygame_mod = None
    if renderer is not None:
        import pygame as pygame_mod

    obs_dict, infos_dict = env.reset(seed=seed)
    step = 0
    terminations: Dict[str, bool] = {}
    truncations: Dict[str, bool] = {}
    total_reward = 0.0

    while env.agents:
        if pygame_mod is not None:
            for event in pygame_mod.event.get():
                if event.type == pygame_mod.QUIT:
                    return _finalize(episode_index, seed, step, env, terminations, truncations, total_reward)

        actions: Dict[str, int] = {}

        # VBS first (same causality fix as main.py): FBS observe their host's
        # committed action via preview_vbs_world_coords before choosing.
        for agent_id in env.agents:
            if "vbs" not in agent_id:
                continue
            t_obs = torch.tensor(obs_dict[agent_id], dtype=torch.float32).to(device)
            t_mask = torch.tensor(infos_dict[agent_id]["action_mask"], dtype=torch.float32).to(device)

            if deterministic:
                action = ppo.get_deterministic_action(t_obs, "vbs", action_mask=t_mask)
            else:
                action, _ = ppo.get_action(t_obs, "vbs", action_mask=t_mask)
            actions[agent_id] = action

        for agent_id in env.agents:
            if "fbs" not in agent_id:
                continue
            host_vbs_id = env.agent_manager.fbs_registry[env._get_raw_id(agent_id)].host_vbs_id
            next_x, next_y = env.preview_vbs_world_coords(f"vbs_{host_vbs_id}", actions[f"vbs_{host_vbs_id}"])

            t_obs = torch.tensor(
                env.augment_fbs_obs(obs_dict[agent_id], next_x, next_y), dtype=torch.float32
            ).to(device)
            t_mask = torch.tensor(infos_dict[agent_id]["action_mask"], dtype=torch.float32).to(device)

            if deterministic:
                action = ppo.get_deterministic_action(t_obs, "fbs", action_mask=t_mask)
            else:
                action, _ = ppo.get_action(t_obs, "fbs", action_mask=t_mask)
            actions[agent_id] = action

        # Snapshot BEFORE stepping: step() empties env.agents on the terminal
        # transition, which would otherwise blank the final rendered frame.
        pre_step_agents = list(env.agents)

        obs_dict, rewards_dict, terminations, truncations, infos_dict = env.step(actions)
        # Team-mean reward, matching main.py's joint_buffer convention.
        total_reward += float(np.mean(list(rewards_dict.values()))) if rewards_dict else 0.0
        step += 1

        episode_done = (bool(terminations) and all(terminations.values())) or (
            bool(truncations) and all(truncations.values())
        )

        if renderer is not None:
            if episode_done:
                env.agents = pre_step_agents  # restore for one valid final frame
            renderer.render(env, step, episode_done=episode_done)
            if episode_done:
                env.agents = []
            if frame_dir:
                pygame_mod.image.save(
                    renderer.screen, os.path.join(frame_dir, f"ep{episode_index:03d}_step{step:04d}.png")
                )
            if step_delay:
                time.sleep(step_delay)

        if episode_done:
            break

    return _finalize(episode_index, seed, step, env, terminations, truncations, total_reward)


def _finalize(episode_index, seed, step, env, terminations, truncations, total_reward) -> EpisodeResult:
    terminated = bool(terminations) and all(terminations.values())
    truncated = bool(truncations) and all(truncations.values()) and not terminated
    return EpisodeResult(
        episode_index=episode_index,
        seed=seed,
        steps=step,
        true_coverage=env.last_true_coverage,
        total_reward=total_reward,
        mean_reward_per_step=(total_reward / step) if step else 0.0,
        terminated=terminated,
        truncated=truncated,
    )


# --------------------------------------------------------------------------- #
# CLI / orchestration
# --------------------------------------------------------------------------- #

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate/visualize a trained VBS/FBS coverage policy.")
    p.add_argument("--config", default="config/simulation_config.json")
    p.add_argument("--graph", default="config/graph_map.json")
    p.add_argument("--model-dir", default=None, help="Explicit checkpoint directory. Auto-discovered if omitted.")
    p.add_argument("--models-root", default="models", help="Searched for the latest run if --model-dir is omitted.")
    p.add_argument("--allow-untrained", action="store_true", help="Proceed with a fresh policy if no checkpoint is found.")
    p.add_argument("--episodes", type=int, default=1, help="Number of evaluation episodes to run.")
    p.add_argument("--seed", type=int, default=42, help="Seed for episode 0; later episodes increment from it unless --fixed-seed.")
    p.add_argument("--fixed-seed", action="store_true", help="Reuse the same seed for every episode instead of incrementing.")
    p.add_argument("--stochastic", action="store_true", help="Sample actions instead of using the deterministic policy.")
    p.add_argument("--headless", action="store_true", help="Run without opening a window. Never imports pygame.")
    p.add_argument("--fps", type=float, default=1.0, help="Rendering frame rate when not headless.")
    p.add_argument("--save-frames-dir", default=None, help="If set (and not --headless), dump each rendered frame as a PNG here.")
    p.add_argument("--summary-json", default=None, help="If set, write aggregate + per-episode metrics to this JSON file.")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    lock_determinism(args.seed)

    if args.save_frames_dir and args.headless:
        raise SystemExit("--save-frames-dir requires rendering; it cannot be combined with --headless.")
    if args.episodes < 1:
        raise SystemExit("--episodes must be >= 1.")

    env_config, _, _ = bootstrap_environment(args.config, args.graph)
    env = CoverageParallelEnv(env_config)
    env.sim_adapter.set_evaluation_mode(True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    vbs_agent_id = next(a for a in env.possible_agents if "vbs" in a)
    fbs_agent_id = next(a for a in env.possible_agents if "fbs" in a)

    ppo = HeterogeneousPPOManager(
        vbs_obs_dim=env.observation_space(vbs_agent_id).shape[0],
        fbs_obs_dim=env.observation_space(fbs_agent_id).shape[0],
        vbs_action_dim=env.action_space(vbs_agent_id).n,
        fbs_action_dim=env.action_space(fbs_agent_id).n,
        global_extra_dim=env.global_extra_dim,
        lr=0.0,  # frozen weights for inference
        device=device,
    )

    model_dir = args.model_dir or discover_latest_model_dir(args.models_root)
    load_policy(ppo, model_dir, device, args.allow_untrained)

    renderer = None
    if not args.headless:
        from visualization.pygame_renderer import PygameRenderer
        renderer = PygameRenderer(map_dim=env.map_dim)

    if args.save_frames_dir:
        os.makedirs(args.save_frames_dir, exist_ok=True)

    summary = EvalSummary()
    step_delay = 1.0 / args.fps if (renderer is not None and args.fps > 0) else 0.0

    print(
        f"Running {args.episodes} episode(s) | device={device} | "
        f"policy={'stochastic' if args.stochastic else 'deterministic'} | headless={args.headless}"
    )

    for i in range(args.episodes):
        seed = args.seed if args.fixed_seed else args.seed + i
        result = run_episode(
            env=env, ppo=ppo, device=device, seed=seed, episode_index=i,
            deterministic=not args.stochastic, renderer=renderer,
            step_delay=step_delay, frame_dir=args.save_frames_dir,
        )
        summary.add(result)
        status = "SOLVED" if result.terminated else "TIME-LIMIT"
        print(
            f"  Episode {i:3d} | seed={seed} | steps={result.steps:3d} | "
            f"coverage={result.true_coverage:.2%} | reward={result.total_reward:8.2f} | {status}"
        )

    if renderer is not None:
        import pygame
        pygame.quit()

    agg = summary.to_dict()
    print("\n-- Summary --------------------------------------------")
    print(f"  Episodes    : {agg['num_episodes']}")
    print(f"  Solve rate  : {agg['solve_rate']:.1%}")
    print(f"  Coverage    : {agg['coverage_mean']:.2%} +/- {agg['coverage_std']:.2%}")
    print(f"  Reward      : {agg['reward_mean']:.2f} +/- {agg['reward_std']:.2f}")
    print(f"  Mean length : {agg['length_mean']:.1f} steps")

    if args.summary_json:
        os.makedirs(os.path.dirname(args.summary_json) or ".", exist_ok=True)
        with open(args.summary_json, "w") as f:
            json.dump(agg, f, indent=2)
        print(f"  Wrote JSON summary to: {args.summary_json}")


if __name__ == "__main__":
    main()