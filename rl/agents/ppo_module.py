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

    def update_actor(self, batch_data: Dict[str, List], agent_type: str,
                     clip_coef: float = 0.2, ent_coef: float = 0.01,
                     ppo_epochs: int = 4, batch_size: int = 64):
        actor = self.vbs_actor if agent_type == "vbs" else self.fbs_actor
        actor.train()

        b_obs = torch.stack(batch_data["obs"]).to(self.device)
        b_actions = torch.tensor(batch_data["actions"], dtype=torch.long, device=self.device)
        b_logprobs = torch.tensor(batch_data["logprobs"], dtype=torch.float32, device=self.device)
        b_advantages = torch.tensor(batch_data["advantages"], dtype=torch.float32, device=self.device)
        b_masks = torch.stack(batch_data["masks"]).to(self.device) if batch_data.get("masks") else None

        dataset_size = b_obs.shape[0]
        for name, tensor in [("actions", b_actions), ("logprobs", b_logprobs), ("advantages", b_advantages)]:
            if tensor.shape[0] != dataset_size:
                raise ValueError(
                    f"Size mismatch: 'obs' has size {dataset_size}, but '{name}' has size {tensor.shape[0]}.")

        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)
        indices = torch.arange(dataset_size, device=self.device)

        for epoch in range(ppo_epochs):
            perm = torch.randperm(dataset_size, device=self.device)
            indices = indices[perm]
            for start in range(0, dataset_size, batch_size):
                mb_idx = indices[start:start + batch_size]
                mb_masks = b_masks[mb_idx] if b_masks is not None else None

                _, new_logprob, entropy = actor.get_action(
                    b_obs[mb_idx], action=b_actions[mb_idx], action_mask=mb_masks
                )
                logratio = new_logprob - b_logprobs[mb_idx]
                ratio = torch.exp(logratio)

                pg_loss1 = -b_advantages[mb_idx] * ratio
                pg_loss2 = -b_advantages[mb_idx] * torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                loss = pg_loss - ent_coef * entropy.mean()

                self.actor_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.vbs_actor.parameters()) + list(self.fbs_actor.parameters()), max_norm=0.5
                )
                self.actor_optimizer.step()

    def update_critic(self, joint_batch: Dict[str, List], vf_coef: float = 0.5,
                      ppo_epochs: int = 4, batch_size: int = 64, clip_coef: float = 0.2):
        """One training sample per ENVIRONMENT STEP, not per agent — the critic estimates
        a single V(joint_state); regressing it against several different per-agent
        returns for the same pooled input would give it contradictory gradients."""
        self.critic.train()

        b_vbs = torch.tensor(np.array(joint_batch["vbs_feats"]), dtype=torch.float32, device=self.device)
        b_fbs = torch.tensor(np.array(joint_batch["fbs_feats"]), dtype=torch.float32, device=self.device)
        b_extra = torch.tensor(np.array(joint_batch["global_extra"]), dtype=torch.float32, device=self.device)
        b_returns = torch.tensor(joint_batch["returns"], dtype=torch.float32, device=self.device)
        b_values = torch.tensor(joint_batch["values"], dtype=torch.float32, device=self.device)

        dataset_size = b_returns.shape[0]
        indices = torch.arange(dataset_size, device=self.device)

        for epoch in range(ppo_epochs):
            perm = torch.randperm(dataset_size, device=self.device)
            indices = indices[perm]
            for start in range(0, dataset_size, batch_size):
                mb_idx = indices[start:start + batch_size]
                new_value = self.critic(b_vbs[mb_idx], b_fbs[mb_idx], b_extra[mb_idx]).flatten()

                v_loss_unclipped = (new_value - b_returns[mb_idx]) ** 2
                v_clipped = b_values[mb_idx] + torch.clamp(new_value - b_values[mb_idx], -clip_coef, clip_coef)
                v_loss_clipped = (v_clipped - b_returns[mb_idx]) ** 2
                v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean() * vf_coef

                self.critic_optimizer.zero_grad()
                v_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
                self.critic_optimizer.step()

    def get_deterministic_action(self, obs: torch.Tensor, agent_type: str, action_mask: torch.Tensor = None) -> int:
        actor = self.vbs_actor if agent_type == "vbs" else self.fbs_actor
        actor.eval()
        with torch.no_grad():
            logits = actor.actor(obs.to(self.device))
            if action_mask is not None:
                action_mask = action_mask.to(self.device)
                logits = torch.where(action_mask.bool(), logits, torch.full_like(logits, -1e9))
            action = logits.argmax(dim=-1)
        return action.cpu().item()

