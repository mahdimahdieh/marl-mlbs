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

    # BUG LEDGER — FIXED (baseline/reward-stream granularity mismatch, Option A):
    # vbs_head/fbs_head used to consume ONLY the pooled, type-level trunk output `h`
    # — i.e. exactly one value per type per step, trained against the MEAN reward
    # across all agents of that type. That single type-mean value was then reused,
    # unmodified, as the GAE baseline for EACH individual agent's own distinct
    # marginal-contribution reward trajectory — a systematic bias whenever agents
    # of the same type diverge in behavior, which the reward design explicitly
    # encourages. Fixed by making vbs_head/fbs_head agent-conditioned: each head
    # now consumes the concatenation of (a) that agent's own per-agent encoded
    # features `v_i`/`f_i` (computed by vbs_encoder/fbs_encoder BEFORE pooling —
    # this already contains the agent's identity_hot feature carried over from its
    # local observation, satisfying the "per-agent identity embedding" requirement
    # without a separate embedding table) and (b) the shared team-level context `h`
    # (so the value estimate remains centralized/joint-state-conditioned, not a
    # decentralized per-agent critic). team_head is UNCHANGED — still a single
    # team-level value computed from the pooled `h` only.
    def __init__(self, vbs_local_dim, fbs_local_dim, global_extra_dim, hidden=128):
        super().__init__()
        self.vbs_encoder = nn.Sequential(layer_init(nn.Linear(vbs_local_dim, hidden)), nn.Tanh())
        self.fbs_encoder = nn.Sequential(layer_init(nn.Linear(fbs_local_dim, hidden)), nn.Tanh())
        self.trunk = nn.Sequential(
            layer_init(nn.Linear(hidden * 4 + global_extra_dim, 128)), nn.Tanh(),
            layer_init(nn.Linear(128, 64)), nn.Tanh(),
        )
        # One head per reward-generating distribution, so each baseline
        # matches the stream it's subtracted from.
        self.team_head = layer_init(nn.Linear(64, 1), std=1.0)
        # Agent-conditioned heads: input is [per-agent encoded features (hidden) ;
        # pooled team context (64)] — see class docstring BUG LEDGER above.
        self.vbs_head = layer_init(nn.Linear(hidden + 64, 1), std=1.0)
        self.fbs_head = layer_init(nn.Linear(hidden + 64, 1), std=1.0)

    def forward(self, vbs_feats, fbs_feats, global_extra):
        v = self.vbs_encoder(vbs_feats)  # (B, n_vbs, hidden) — per-agent, pre-pooling
        v_pool = torch.cat([v.mean(dim=1), v.max(dim=1).values], dim=-1)
        f = self.fbs_encoder(fbs_feats)  # (B, n_fbs, hidden) — per-agent, pre-pooling
        f_pool = torch.cat([f.mean(dim=1), f.max(dim=1).values], dim=-1)
        h = self.trunk(torch.cat([v_pool, f_pool, global_extra], dim=-1))  # (B, 64), team-level context

        n_vbs, n_fbs = v.shape[1], f.shape[1]
        h_for_vbs = h.unsqueeze(1).expand(-1, n_vbs, -1)  # (B, n_vbs, 64)
        h_for_fbs = h.unsqueeze(1).expand(-1, n_fbs, -1)  # (B, n_fbs, 64)

        # (B, n_vbs, 1) -> (B, n_vbs); one value per VBS agent, conditioned on its
        # own encoded features AND the shared team context.
        vbs_values = self.vbs_head(torch.cat([v, h_for_vbs], dim=-1)).squeeze(-1)
        fbs_values = self.fbs_head(torch.cat([f, h_for_fbs], dim=-1)).squeeze(-1)

        return {"team": self.team_head(h), "vbs": vbs_values, "fbs": fbs_values}

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
        """Returns {"team": float, "vbs": List[float], "fbs": List[float]}.

        team is still a single scalar (team_head is unchanged). vbs/fbs are now
        one value PER AGENT (see CentralizedCritic BUG LEDGER) in the same agent
        order as the vbs_feats/fbs_feats rows passed in (i.e. the order produced
        by env.get_global_state()) — callers must preserve that ordering when
        matching these values back to agent_ids.
        """
        self.critic.eval()
        with torch.no_grad():
            out = self.critic(vbs_feats.to(self.device), fbs_feats.to(self.device), global_extra.to(self.device))
            return {
                "team": out["team"].cpu().item(),
                "vbs": out["vbs"].squeeze(0).cpu().tolist(),
                "fbs": out["fbs"].squeeze(0).cpu().tolist(),
            }

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

    def update_critic(self, joint_batch, vf_coef=0.5, ppo_epochs=4, batch_size=64, clip_coef=0.2):
        """One training SAMPLE (row) per ENVIRONMENT STEP, matching the pooled
        vbs_feats/fbs_feats/global_extra inputs. "team" targets are a scalar per
        step (team_head, unchanged). "vbs"/"fbs" targets are now a vector of one
        value PER AGENT per step (see CentralizedCritic BUG LEDGER) — each column
        trains the corresponding per-agent output slot of vbs_head/fbs_head against
        that specific agent's own return, rather than a single type-mean target
        applied identically across agents."""
        self.critic.train()
        b_vbs = torch.tensor(np.array(joint_batch["vbs_feats"]), dtype=torch.float32, device=self.device)
        b_fbs = torch.tensor(np.array(joint_batch["fbs_feats"]), dtype=torch.float32, device=self.device)
        b_extra = torch.tensor(np.array(joint_batch["global_extra"]), dtype=torch.float32, device=self.device)

        heads = {
            "team": (joint_batch["team_returns"], joint_batch["team_values"]),
            "vbs": (joint_batch["vbs_returns"], joint_batch["vbs_values"]),
            "fbs": (joint_batch["fbs_returns"], joint_batch["fbs_values"]),
        }
        b_returns = {k: torch.tensor(v[0], dtype=torch.float32, device=self.device) for k, v in heads.items()}
        b_values = {k: torch.tensor(v[1], dtype=torch.float32, device=self.device) for k, v in heads.items()}

        dataset_size = b_returns["team"].shape[0]
        indices = torch.arange(dataset_size, device=self.device)

        for epoch in range(ppo_epochs):
            perm = torch.randperm(dataset_size, device=self.device)
            indices = indices[perm]
            for start in range(0, dataset_size, batch_size):
                mb_idx = indices[start:start + batch_size]
                new_out = self.critic(b_vbs[mb_idx], b_fbs[mb_idx], b_extra[mb_idx])

                loss = 0.0
                for k in ("team", "vbs", "fbs"):
                    # Flatten uniformly: "team" is (batch, 1) -> (batch,); "vbs"/"fbs"
                    # are (batch, n_agents) -> (batch * n_agents,). b_returns[k]/
                    # b_values[k] are reshaped to match before flattening so each
                    # element still pairs a per-agent (or per-step, for team)
                    # prediction with its own matching target — never a shared
                    # type-mean target broadcast across agents.
                    nv = new_out[k].reshape(-1)
                    target_returns = b_returns[k][mb_idx].reshape(nv.shape)
                    target_values = b_values[k][mb_idx].reshape(nv.shape)
                    v_unclipped = (nv - target_returns) ** 2
                    v_clipped = target_values + torch.clamp(nv - target_values, -clip_coef, clip_coef)
                    v_clipped_loss = (v_clipped - target_returns) ** 2
                    loss = loss + 0.5 * torch.max(v_unclipped, v_clipped_loss).mean() * vf_coef

                self.critic_optimizer.zero_grad()
                loss.backward()
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

