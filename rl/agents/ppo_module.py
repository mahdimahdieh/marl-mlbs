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


class DiscreteActorCritic(nn.Module):
    """
    Decoupled Actor-Critic network with strict action masking support.
    """

    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()
        # Share standard trunk representation to reduce parameter footprint
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0)
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, action_dim), std=0.01)  # Low std ensures initial uniform exploration
        )

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        return self.critic(x)

    def get_action_and_value(self, x: torch.Tensor, action: torch.Tensor = None, action_mask: torch.Tensor = None) -> \
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.actor(x)

        # STRICT FIX: Move action_mask to the same device as logits
        if action_mask is not None:
            # Ensure the mask is on the same device as the logits
            action_mask = action_mask.to(logits.device)

            # Now both tensors are guaranteed to be on the same device (e.g., cuda:0)
            logits = torch.where(action_mask.bool(), logits, torch.tensor(-1e9, device=logits.device))

        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()

        return action, probs.log_prob(action), probs.entropy(), self.critic(x)

class HeterogeneousPPOManager:
    """
    Manages isolated optimization updates for disparate agent types (VBS vs FBS)
    to completely avoid weight pollution.
    """

    def __init__(self, vbs_obs_dim: int, fbs_obs_dim: int, lr: float = 3e-4, device: str = "cpu"):
        self.device = torch.device(device)

        # VBS Action Space = 3 (Graph Edge Navigation)
        self.vbs_net = DiscreteActorCritic(vbs_obs_dim, 3).to(self.device)
        # FBS Action Space = 17 (0: Hover, 1-8: Half-Dist, 9-16: Full-Dist)
        self.fbs_net = DiscreteActorCritic(fbs_obs_dim, 17).to(self.device)

        self.vbs_optimizer = optim.Adam(self.vbs_net.parameters(), lr=lr, eps=1e-5)
        self.fbs_optimizer = optim.Adam(self.fbs_net.parameters(), lr=lr, eps=1e-5)

    def get_action(self, obs: torch.Tensor, agent_type: str, action_mask: torch.Tensor = None) -> Tuple[
        int, float, float]:
        """Inference execution. Invoked during the environment rollout loop."""
        self.vbs_net.eval()
        self.fbs_net.eval()

        net = self.vbs_net if agent_type == "vbs" else self.fbs_net

        with torch.no_grad():
            action, log_prob, _, value = net.get_action_and_value(obs.to(self.device), action_mask=action_mask)

        return action.cpu().item(), log_prob.cpu().item(), value.cpu().item()

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

        # Normalize advantages inside the target sub-batch to zero-out rollout structural variance
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        dataset_size = b_obs.shape[0]
        indices = np.arange(dataset_size)

        for epoch in range(ppo_epochs):
            np.random.shuffle(indices)
            for start in range(0, dataset_size, batch_size):
                end = start + batch_size
                mb_idx = indices[start:end]

                # Extract slices
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