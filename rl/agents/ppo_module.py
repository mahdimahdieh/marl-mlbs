import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import numpy as np
from typing import Dict, Tuple, List


def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
    """Strict orthogonal initialization contract for stable MARL gradient flow."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class DiscreteActor(nn.Module):
    """Decentralized, execution-time. Local-obs-only."""
    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, action_dim), std=0.01)
        )

    def get_action(self, x, action=None, action_mask=None):
        logits = self.actor(x)
        if action_mask is not None:
            action_mask = action_mask.to(logits.device)
            logits = torch.where(action_mask.bool(), logits, torch.tensor(-1e9, device=logits.device))
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy()


class CentralizedCritic(nn.Module):
    """Training-time only. Deep-Sets pooling → permutation- and n_agents-invariant V(s)."""
    def __init__(self, vbs_local_dim: int, fbs_local_dim: int, global_extra_dim: int, hidden: int = 128):
        super().__init__()
        self.vbs_encoder = nn.Sequential(layer_init(nn.Linear(vbs_local_dim, hidden)), nn.Tanh())
        self.fbs_encoder = nn.Sequential(layer_init(nn.Linear(fbs_local_dim, hidden)), nn.Tanh())
        self.head = nn.Sequential(
            layer_init(nn.Linear(hidden * 4 + global_extra_dim, 128)), nn.Tanh(),
            layer_init(nn.Linear(128, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0)
        )

    def forward(self, vbs_feats, fbs_feats, global_extra):
        # vbs_feats: (B, n_vbs, vbs_local_dim)   fbs_feats: (B, n_fbs, fbs_local_dim)
        v = self.vbs_encoder(vbs_feats)
        v_pool = torch.cat([v.mean(dim=1), v.max(dim=1).values], dim=-1)
        f = self.fbs_encoder(fbs_feats)
        f_pool = torch.cat([f.mean(dim=1), f.max(dim=1).values], dim=-1)
        return self.head(torch.cat([v_pool, f_pool, global_extra], dim=-1))

class HeterogeneousPPOManager:
    """
    Manages isolated optimization updates for disparate agent types (VBS vs FBS)
    to completely avoid weight pollution.
    """

    def __init__(self, vbs_obs_dim, fbs_obs_dim, vbs_action_dim, fbs_action_dim,
                 global_extra_dim: int, lr: float = 3e-4, device: str = "cpu"):
        self.device = torch.device(device)
        self.vbs_actor = DiscreteActor(vbs_obs_dim, vbs_action_dim).to(self.device)
        self.fbs_actor = DiscreteActor(fbs_obs_dim, fbs_action_dim).to(self.device)
        self.critic = CentralizedCritic(vbs_obs_dim, fbs_obs_dim, global_extra_dim).to(self.device)
        self.actor_optimizer = optim.Adam(
            list(self.vbs_actor.parameters()) + list(self.fbs_actor.parameters()), lr=lr, eps=1e-5)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr, eps=1e-5)

    def get_action(self, obs, agent_type, action_mask=None):
        net = self.vbs_actor if agent_type == "vbs" else self.fbs_actor
        net.eval()
        with torch.no_grad():
            action, log_prob, _ = net.get_action(obs.to(self.device), action_mask=action_mask)
        return action.cpu().item(), log_prob.cpu().item()

    def get_value(self, vbs_feats, fbs_feats, global_extra):
        self.critic.eval()
        with torch.no_grad():
            return self.critic(vbs_feats.to(self.device), fbs_feats.to(self.device),
                               global_extra.to(self.device)).cpu().item()

    def update_policy(self, batch_data: Dict[str, List], agent_type: str, clip_coef: float = 0.2,
                      ent_coef: float = 0.01, vf_coef: float = 0.5, ppo_epochs: int = 4, batch_size: int = 64):
        """Executes a vectorized mini-batch PPO update cycle over stored trajectories."""
        self.vbs_net.train()
        self.fbs_net.train()

        net = self.vbs_net if agent_type == "vbs" else self.fbs_net
        optimizer = self.vbs_optimizer if agent_type == "vbs" else self.fbs_optimizer

        # Convert historical buffer lists to raw tensor shapes
        b_obs = torch.stack(batch_data["obs"]).to(self.device)
        b_actions = torch.tensor(batch_data["actions"], dtype=torch.long, device=self.device)
        b_logprobs = torch.tensor(batch_data["logprobs"], dtype=torch.float32, device=self.device)
        b_advantages = torch.tensor(batch_data["advantages"], dtype=torch.float32, device=self.device)
        b_returns = torch.tensor(batch_data["returns"], dtype=torch.float32, device=self.device)
        b_values = torch.tensor(batch_data["values"], dtype=torch.float32, device=self.device)

        b_masks = None
        if "masks" in batch_data and batch_data["masks"]:
            b_masks = torch.stack(batch_data["masks"]).to(self.device)

        # --- CRITICAL SANITY CHECK ---
        # Ensure all buffer tensors have the exact same batch dimension size
        dataset_size = b_obs.shape[0]
        for name, tensor in [("actions", b_actions), ("logprobs", b_logprobs),
                             ("advantages", b_advantages), ("returns", b_returns), ("values", b_values)]:
            if tensor.shape[0] != dataset_size:
                raise ValueError(
                    f"Size mismatch: 'obs' has size {dataset_size}, but '{name}' has size {tensor.shape[0]}. "
                    f"Check your rollout accumulation in main.py!")

        # Normalize advantages inside the target sub-batch to zero-out rollout structural variance
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        # FIX: Generate indices directly as a PyTorch LongTensor on the GPU device
        indices = torch.arange(dataset_size, device=self.device)

        for epoch in range(ppo_epochs):
            # Generate the permutation directly on self.device: inherits the globally
            # pinned generator deterministically, and avoids a CPU/CUDA index-tensor
            # mismatch RuntimeError the moment device="cuda".
            perm = torch.randperm(dataset_size, device=self.device)
            indices = indices[perm]

            for start in range(0, dataset_size, batch_size):
                end = start + batch_size
                mb_idx = indices[start:end]

                # Extract slices (now safe and entirely on GPU device memory)
                mb_obs = b_obs[mb_idx]
                mb_actions = b_actions[mb_idx]
                mb_logprobs = b_logprobs[mb_idx]
                mb_advantages = b_advantages[mb_idx]
                mb_returns = b_returns[mb_idx]
                mb_values = b_values[mb_idx]
                mb_masks = b_masks[mb_idx] if b_masks is not None else None

                _, new_logprob, entropy, new_value = net.get_action_and_value(
                    mb_obs, mb_actions, action_mask=mb_masks
                )

                # Policy Gradient Clipping (L^CLIP)
                logratio = new_logprob - mb_logprobs
                ratio = torch.exp(logratio)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Critic Loss Clipping (L^VF) to prevent value variance explosions
                new_value = new_value.flatten()
                v_loss_unclipped = (new_value - mb_returns) ** 2
                v_clipped = mb_values + torch.clamp(new_value - mb_values, -clip_coef, clip_coef)
                v_loss_clipped = (v_clipped - mb_returns) ** 2
                v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                # Entropy Exploration Bonus
                entropy_loss = entropy.mean()

                # Total Objective
                loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef

                # Clean optimization step
                optimizer.zero_grad()
                loss.backward()
                # Strict gradient norm clipping to safeguard against exploding graph updates
                nn.utils.clip_grad_norm_(net.parameters(), max_norm=0.5)
                optimizer.step()

    def get_deterministic_action(
            self,
            obs: torch.Tensor,
            agent_type: str,
            action_mask: torch.Tensor = None,
    ) -> int:
        """
        Greedy argmax policy for evaluation / inference.

        CRITICAL DISTINCTION from get_action():
          - get_action()              → probs.sample()  ← stochastic; ONLY for training rollouts
          - get_deterministic_action()→ logits.argmax() ← greedy;    ONLY for inference

        Calling sample() during inference produces the random-looking behavior the
        user observed. Argmax collapses the categorical distribution to its mode,
        producing consistent, reproducible agent decisions.

        Args:
            obs:         (obs_dim,) float32 tensor — single agent observation
            agent_type:  "vbs" or "fbs"
            action_mask: (action_dim,) float32 tensor — 1=legal, 0=masked

        Returns:
            int: the greedy legal action index
        """
        net = self.vbs_net if agent_type == "vbs" else self.fbs_net
        net.eval()  # Disable dropout/batchnorm (safe; update_policy() re-enables .train())

        with torch.no_grad():
            # Only need the actor head — critic is irrelevant for action selection
            logits = net.actor(obs.to(self.device))

            if action_mask is not None:
                action_mask = action_mask.to(self.device)
                # Replace illegal action logits with -inf BEFORE argmax
                # so masked actions can never win the greedy selection
                logits = torch.where(
                    action_mask.bool(),
                    logits,
                    torch.full_like(logits, -1e9),  # -inf proxy
                )

            # Greedy selection: no sampling, fully deterministic
            action = logits.argmax(dim=-1)

        return action.cpu().item()

