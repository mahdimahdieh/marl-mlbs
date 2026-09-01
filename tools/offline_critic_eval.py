"""Offline critic bake-off: Deep-Sets (v1) vs attention (D-v2) on recorded
rollouts. No training-loop risk — the default config keeps
critic_arch="deepsets" until this tool shows an explained-variance win.

Records rollouts under a uniform random policy (or trained actors from
--models-dir), computes Monte-Carlo discounted returns per agent, trains
BOTH critics on the same train split with the same protocol, and reports
held-out per-head explained variance.

Adoption rule (printed at the end): switch hyperparameters.critic_arch to
"attention" only if it improves EV by >= +0.05 on at least one head and
regresses none by more than 0.01.

Usage:
    python tools/offline_critic_eval.py [--episodes 40] [--epochs 100] [--seed 0]
    python tools/offline_critic_eval.py --models-dir models/<run> --episodes 40
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from main import bootstrap_environment  # noqa: E402
from rl.envs.pettingzoo_env import CoverageParallelEnv  # noqa: E402
from rl.agents.ppo_module import (  # noqa: E402
    AttentionCritic, CentralizedCritic, HeterogeneousPPOManager,
)
from infrastructure.training.determinism import lock_determinism  # noqa: E402


def _rollout_actions(env, ppo, obs_dict, infos_dict, rng):
    """Random actions, or trained-actor samples in main.py's VBS-first /
    FBS-with-host-preview order (keeps the policy on-distribution)."""
    if ppo is None:
        return {a: int(rng.integers(env.action_space(a).n)) for a in env.agents}

    actions = {}
    for agent_id in env.agents:
        if "vbs" not in agent_id:
            continue
        t_obs = torch.tensor(obs_dict[agent_id], dtype=torch.float32)
        t_mask = torch.tensor(infos_dict[agent_id]["action_mask"], dtype=torch.float32)
        actions[agent_id], _ = ppo.get_action(t_obs, "vbs", action_mask=t_mask)

    for agent_id in env.agents:
        if "fbs" not in agent_id:
            continue
        fbs_obj = env.agent_manager.fbs_registry[env._get_raw_id(agent_id)]
        next_x, next_y = env.preview_vbs_world_coords(
            f"vbs_{fbs_obj.host_vbs_id}", actions[f"vbs_{fbs_obj.host_vbs_id}"])
        t_obs = torch.tensor(
            env.augment_fbs_obs(obs_dict[agent_id], next_x, next_y), dtype=torch.float32)
        t_mask = torch.tensor(infos_dict[agent_id]["action_mask"], dtype=torch.float32)
        actions[agent_id], _ = ppo.get_action(t_obs, "fbs", action_mask=t_mask)
    return actions


def record_rollouts(env, ppo, episodes, seed):
    """One entry per episode: per-step critic inputs + per-step per-agent
    rewards (targets are derived later as Monte-Carlo returns)."""
    rng = np.random.default_rng(seed)
    dataset = []
    for ep in range(episodes):
        obs_dict, infos_dict = env.reset(seed=seed + ep)
        steps = {"vbs_feats": [], "fbs_feats": [], "global_extra": [],
                 "rewards": {a: [] for a in env.possible_agents}}
        while env.agents:
            actions = _rollout_actions(env, ppo, obs_dict, infos_dict, rng)
            vbs_feats, fbs_feats, global_extra = env.get_global_state()
            steps["vbs_feats"].append(np.asarray(vbs_feats, dtype=np.float32))
            steps["fbs_feats"].append(np.asarray(fbs_feats, dtype=np.float32))
            steps["global_extra"].append(np.asarray(global_extra, dtype=np.float32))
            obs_dict, rewards, _, _, infos_dict = env.step(actions)
            for a in actions:
                steps["rewards"][a].append(float(rewards[a]))
        steps["host_indices"] = env.get_fbs_host_vbs_indices()
        dataset.append(steps)
    return dataset


def mc_returns(rewards, gamma):
    """G_t = sum_{k>=t} gamma^(k-t) r_k, per reward stream."""
    out = [0.0] * len(rewards)
    acc = 0.0
    for t in reversed(range(len(rewards))):
        acc = rewards[t] + gamma * acc
        out[t] = acc
    return out


def build_rows(dataset, gamma):
    """Flatten episodes into (inputs, targets) rows. Targets are Monte-Carlo
    returns: team = mean-over-agents stream, vbs/fbs = per-agent streams."""
    rows = []
    for ep in dataset:
        T = len(ep["global_extra"])
        agent_ids = list(ep["rewards"].keys())
        vbs_ids = [a for a in agent_ids if "vbs" in a]
        fbs_ids = [a for a in agent_ids if "fbs" in a]
        team_stream = [
            float(np.mean([ep["rewards"][a][t] for a in agent_ids])) for t in range(T)
        ]
        team_g = mc_returns(team_stream, gamma)
        vbs_g = [mc_returns(ep["rewards"][a], gamma) for a in vbs_ids]   # (n_vbs, T)
        fbs_g = [mc_returns(ep["rewards"][a], gamma) for a in fbs_ids]   # (n_fbs, T)

        for t in range(T):
            rows.append({
                "vbs_feats": ep["vbs_feats"][t],
                "fbs_feats": ep["fbs_feats"][t],
                "global_extra": ep["global_extra"][t],
                "host_indices": ep["host_indices"],
                "team": team_g[t],
                "vbs": [g[t] for g in vbs_g],
                "fbs": [g[t] for g in fbs_g],
            })
    return rows


def _to_tensors(rows, device):
    vbs = torch.tensor(np.stack([r["vbs_feats"] for r in rows]), dtype=torch.float32, device=device)
    fbs = torch.tensor(np.stack([r["fbs_feats"] for r in rows]), dtype=torch.float32, device=device)
    extra = torch.tensor(np.stack([r["global_extra"] for r in rows]), dtype=torch.float32, device=device)
    targets = {k: torch.tensor([r[k] for r in rows], dtype=torch.float32, device=device)
               for k in ("team", "vbs", "fbs")}
    host_indices = rows[0]["host_indices"]
    return vbs, fbs, extra, targets, host_indices


def train_critic(critic, rows, device, epochs, batch_size, lr):
    vbs, fbs, extra, targets, host_indices = _to_tensors(rows, device)
    opt = torch.optim.Adam(critic.parameters(), lr=lr)
    T = vbs.shape[0]
    for _ in range(epochs):
        perm = torch.randperm(T, device=device)
        for start in range(0, T, batch_size):
            idx = perm[start:start + batch_size]
            out = critic(vbs[idx], fbs[idx], extra[idx], fbs_host_vbs_indices=host_indices)
            loss = sum(
                torch.mean((out[k].reshape(-1) - targets[k][idx].reshape(out[k].reshape(-1).shape)) ** 2)
                for k in ("team", "vbs", "fbs"))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=0.5)
            opt.step()
    return critic


def explained_variances(critic, rows, device, batch_size=256):
    """Per-head EV = 1 - Var(residual) / Var(target) on held-out rows."""
    vbs, fbs, extra, targets, host_indices = _to_tensors(rows, device)
    critic.eval()
    preds = {k: [] for k in ("team", "vbs", "fbs")}
    with torch.no_grad():
        for start in range(0, vbs.shape[0], batch_size):
            sl = slice(start, start + batch_size)
            out = critic(vbs[sl], fbs[sl], extra[sl], fbs_host_vbs_indices=host_indices)
            for k in preds:
                preds[k].append(out[k].reshape(-1, out[k].shape[-1] if out[k].dim() > 2 else 1))
    evs = {}
    for k in preds:
        pred = torch.cat(preds[k], dim=0).reshape(-1)
        target = targets[k].reshape(-1)
        resid_var = torch.var(target - pred).item()
        target_var = torch.var(target).item()
        evs[k] = 1.0 - resid_var / max(target_var, 1e-12)
    return evs


def main():
    parser = argparse.ArgumentParser(description="Offline critic bake-off")
    parser.add_argument("--config", type=str, default="config/simulation_config.json")
    parser.add_argument("--graph", type=str, default="config/graph_map.json")
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--models-dir", type=str, default=None,
                        help="optional trained actor checkpoint dir (vbs_actor.pt / "
                             "fbs_actor.pt); default records a uniform random policy")
    parser.add_argument("--train-frac", type=float, default=0.7)
    args = parser.parse_args()

    lock_determinism(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    env_config, hp, _ = bootstrap_environment(args.config, args.graph)
    env = CoverageParallelEnv(env_config)

    vbs_agent_id = next(a for a in env.possible_agents if "vbs" in a)
    fbs_agent_id = next(a for a in env.possible_agents if "fbs" in a)

    ppo = None
    if args.models_dir:
        ppo = HeterogeneousPPOManager(
            vbs_obs_dim=env.observation_space(vbs_agent_id).shape[0],
            fbs_obs_dim=env.observation_space(fbs_agent_id).shape[0],
            vbs_action_dim=env.action_space(vbs_agent_id).n,
            fbs_action_dim=env.action_space(fbs_agent_id).n,
            global_extra_dim=env.global_extra_dim,
            lr=hp["learning_rate"], device=device,
        )
        ppo.vbs_actor.load_state_dict(
            torch.load(Path(args.models_dir) / "vbs_actor.pt", map_location=device))
        ppo.fbs_actor.load_state_dict(
            torch.load(Path(args.models_dir) / "fbs_actor.pt", map_location=device))
        print(f"Recording rollouts with trained actors from {args.models_dir}")
    else:
        print("Recording rollouts with a uniform random policy")

    dataset = record_rollouts(env, ppo, episodes=args.episodes, seed=args.seed)
    rows = build_rows(dataset, gamma=args.gamma)
    print(f"Recorded {len(rows)} steps from {len(dataset)} episodes "
          f"(gamma={args.gamma}, device={device})")

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(rows))
    n_train = int(len(rows) * args.train_frac)
    train_rows = [rows[i] for i in order[:n_train]]
    hold_rows = [rows[i] for i in order[n_train:]]
    print(f"Split: {len(train_rows)} train / {len(hold_rows)} held-out rows\n")

    vbs_dim = env.observation_space(vbs_agent_id).shape[0]
    fbs_dim = env.observation_space(fbs_agent_id).shape[0]
    results = {}
    for name, cls in (("deepsets", CentralizedCritic), ("attention", AttentionCritic)):
        torch.manual_seed(args.seed)
        critic = cls(vbs_dim, fbs_dim, env.global_extra_dim).to(device)
        train_critic(critic, train_rows, device,
                     epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
        results[name] = explained_variances(critic, hold_rows, device)

    print(f"{'head':6s} {'EV deepsets':>12s} {'EV attention':>13s} {'delta':>9s}")
    for head in ("team", "vbs", "fbs"):
        ev_a, ev_b = results["deepsets"][head], results["attention"][head]
        print(f"{head:6s} {ev_a:12.4f} {ev_b:13.4f} {ev_b - ev_a:+9.4f}")

    deltas = {h: results["attention"][h] - results["deepsets"][h]
              for h in ("team", "vbs", "fbs")}
    adopt = any(d >= 0.05 for d in deltas.values()) and all(d >= -0.01 for d in deltas.values())
    if adopt:
        print("\nADOPT: attention improves explained variance by >= 0.05 on at "
              "least one head with no regression — set \"critic_arch\": "
              "\"attention\" in config/simulation_config.json")
    else:
        print("\nKEEP deepsets: attention does not meet the adoption rule "
              "(>= +0.05 EV on some head, no head below -0.01)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
