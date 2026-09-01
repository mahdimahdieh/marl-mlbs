"""D-v2 attention critic (rl/agents/ppo_module.py): same output contract as
CentralizedCritic, tether-pairing sensitivity, permutation equivariance, and
HeterogeneousPPOManager integration behind the critic_arch flag."""

import os
import sys

import numpy as np
import torch
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rl.agents.ppo_module import AttentionCritic, HeterogeneousPPOManager


def _make_critic(seed=0):
    torch.manual_seed(seed)
    return AttentionCritic(vbs_local_dim=13, fbs_local_dim=16, global_extra_dim=5)


def test_attention_critic_output_contract_matches_deepsets():
    n_vbs, n_fbs, vbs_dim, fbs_dim, extra_dim = 3, 2, 13, 16, 5
    critic = _make_critic()

    batch = 4
    vbs_feats = torch.randn(batch, n_vbs, vbs_dim)
    fbs_feats = torch.randn(batch, n_fbs, fbs_dim)
    global_extra = torch.randn(batch, extra_dim)

    out = critic(vbs_feats, fbs_feats, global_extra, fbs_host_vbs_indices=[2, 1])

    assert out["team"].shape == (batch, 1)
    assert out["vbs"].shape == (batch, n_vbs)
    assert out["fbs"].shape == (batch, n_fbs)


def test_attention_critic_agent_conditioned_values():
    """Two agents with different local features must get different values
    (the per-token heads are conditioned on each agent's own encoding)."""
    critic = _make_critic()
    critic.eval()

    vbs_feats = torch.zeros(1, 2, 13)
    vbs_feats[0, 0] = torch.full((13,), 1.0)
    vbs_feats[0, 1] = torch.full((13,), -1.0)
    fbs_feats = torch.zeros(1, 1, 16)
    global_extra = torch.zeros(1, 5)

    with torch.no_grad():
        out = critic(vbs_feats, fbs_feats, global_extra, fbs_host_vbs_indices=[0])

    assert out["vbs"][0, 0].item() != out["vbs"][0, 1].item()


def test_attention_critic_tether_pairing_changes_outputs():
    """Identical features but a different FBS->host mapping must change the
    outputs — the tether is an attention bias, not pooled away."""
    critic = _make_critic()
    critic.eval()

    vbs_feats = torch.randn(1, 2, 13)
    fbs_feats = torch.randn(1, 2, 16)
    global_extra = torch.randn(1, 5)

    with torch.no_grad():
        both_on_host_0 = critic(vbs_feats, fbs_feats, global_extra,
                                fbs_host_vbs_indices=[0, 0])
        split_hosts = critic(vbs_feats, fbs_feats, global_extra,
                             fbs_host_vbs_indices=[0, 1])

    assert not torch.allclose(both_on_host_0["fbs"], split_hosts["fbs"])
    assert not torch.allclose(both_on_host_0["team"], split_hosts["team"])


def test_attention_critic_permutation_equivariant_across_disjoint_pairs():
    """Permuting whole (VBS, FBS) pairs leaves the team value unchanged and
    permutes per-agent values with the rows."""
    critic = _make_critic(seed=1)
    critic.eval()

    n_vbs, n_fbs = 3, 3
    vbs_feats = torch.randn(1, n_vbs, 13)
    fbs_feats = torch.randn(1, n_fbs, 16)
    global_extra = torch.randn(1, 5)
    hosts = [0, 1, 2]  # disjoint pairs: (vbs_i, fbs_i)

    with torch.no_grad():
        out = critic(vbs_feats, fbs_feats, global_extra, fbs_host_vbs_indices=hosts)

    pi = [2, 0, 1]
    vbs_feats_p = vbs_feats[:, pi]
    fbs_feats_p = fbs_feats[:, pi]
    hosts_p = [pi.index(hosts[pi[j]]) for j in range(n_fbs)]

    with torch.no_grad():
        out_p = critic(vbs_feats_p, fbs_feats_p, global_extra, fbs_host_vbs_indices=hosts_p)

    assert torch.allclose(out["team"], out_p["team"], atol=1e-5), (
        "team value must be invariant to permuting whole (VBS, FBS) pairs"
    )
    assert torch.allclose(out["vbs"][:, pi], out_p["vbs"], atol=1e-5)
    assert torch.allclose(out["fbs"][:, pi], out_p["fbs"], atol=1e-5)


def test_attention_critic_invariant_within_shared_host_group():
    """Two FBS tethered to the SAME host form an unordered group: swapping
    them leaves every output (appropriately permuted) unchanged."""
    critic = _make_critic(seed=2)
    critic.eval()

    vbs_feats = torch.randn(1, 2, 13)
    fbs_feats = torch.randn(1, 2, 16)
    global_extra = torch.randn(1, 5)

    with torch.no_grad():
        out = critic(vbs_feats, fbs_feats, global_extra, fbs_host_vbs_indices=[0, 0])
        out_swapped = critic(vbs_feats, fbs_feats[:, [1, 0]], global_extra,
                             fbs_host_vbs_indices=[0, 0])

    assert torch.allclose(out["team"], out_swapped["team"], atol=1e-5)
    assert torch.allclose(out["vbs"], out_swapped["vbs"], atol=1e-5)
    assert torch.allclose(out["fbs"][:, [1, 0]], out_swapped["fbs"], atol=1e-5)


def test_manager_attention_arch_get_value_and_update():
    """The manager runs end-to-end with critic_arch='attention': rollout-time
    get_value and a real update_critic batch must both work unchanged."""
    n_vbs, n_fbs, T = 3, 2, 10
    ppo = HeterogeneousPPOManager(
        vbs_obs_dim=13, fbs_obs_dim=16, vbs_action_dim=33, fbs_action_dim=17,
        global_extra_dim=5, device="cpu", critic_arch="attention",
    )
    assert isinstance(ppo.critic, AttentionCritic)

    vbs_feats = torch.randn(1, n_vbs, 13)
    fbs_feats = torch.randn(1, n_fbs, 16)
    global_extra = torch.randn(1, 5)
    out = ppo.get_value(vbs_feats, fbs_feats, global_extra,
                        fbs_host_vbs_indices=[2, 1])
    assert isinstance(out["team"], float)
    assert len(out["vbs"]) == n_vbs and len(out["fbs"]) == n_fbs

    rng = np.random.default_rng(0)
    joint_batch = {
        "vbs_feats": rng.standard_normal((T, n_vbs, 13)).astype(np.float32),
        "fbs_feats": rng.standard_normal((T, n_fbs, 16)).astype(np.float32),
        "global_extra": rng.standard_normal((T, 5)).astype(np.float32),
        "team_values": rng.standard_normal(T).astype(np.float32).tolist(),
        "team_returns": rng.standard_normal(T).astype(np.float32).tolist(),
        "vbs_values": rng.standard_normal((T, n_vbs)).astype(np.float32).tolist(),
        "vbs_returns": rng.standard_normal((T, n_vbs)).astype(np.float32).tolist(),
        "fbs_values": rng.standard_normal((T, n_fbs)).astype(np.float32).tolist(),
        "fbs_returns": rng.standard_normal((T, n_fbs)).astype(np.float32).tolist(),
        "fbs_host_vbs_indices": [2, 1],
    }
    ppo.update_critic(joint_batch, ppo_epochs=1, batch_size=4)  # must not raise


def test_manager_unknown_critic_arch_raises():
    with pytest.raises(ValueError, match="critic_arch"):
        HeterogeneousPPOManager(
            vbs_obs_dim=13, fbs_obs_dim=16, vbs_action_dim=33, fbs_action_dim=17,
            global_extra_dim=5, device="cpu", critic_arch="bogus",
        )
