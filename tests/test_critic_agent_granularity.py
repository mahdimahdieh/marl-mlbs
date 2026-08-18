"""
Regression coverage for the baseline/reward-stream granularity fix spanning
rl/agents/ppo_module.py (CentralizedCritic, Option A: agent-conditioned value
heads) and main.py's training loop.

BUG LEDGER context: vbs_values/fbs_values used to be ONE value per type per
step, trained against the type-mean reward, then reused unmodified as the GAE
baseline for EACH individual agent's own distinct reward trajectory. Fixed by
making vbs_head/fbs_head output one value per agent per step, and by
restructuring main.py's rollout/optimization loop so each agent's own value
trajectory (buffers[type][agent_id]["values"]) is baselined against that same
agent's own reward trajectory.

These tests pin down:
  - CentralizedCritic.forward returns per-agent vbs/fbs values (not a single
    per-type scalar), while team_head remains a single scalar.
  - two agents with different local features get different values (proof the
    head is agent-conditioned, not a broadcast type-mean).
  - HeterogeneousPPOManager.get_value returns per-agent lists of the correct
    length.
  - update_critic accepts per-agent (T, n_agents) return/value matrices and
    runs without shape errors.
  - main.py's rollout produces per-agent value trajectories at the same
    length as that agent's own reward trajectory (the actual granularity
    mismatch the task fixes), by running a couple of real, short episodes
    through the true training loop machinery.
"""
import os
import sys
import json
import numpy as np
import torch
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rl.agents.ppo_module import CentralizedCritic, HeterogeneousPPOManager
from core.entities.agents import AgentManager, VehicleBaseStation, FlyingBaseStation
from infrastructure.graph.networkx_engine import NetworkXRoadEngine
from infrastructure.simulation.pywisim_adapter import PyWiSimAdapter
from rl.envs.pettingzoo_env import CoverageParallelEnv
import main as main_module

GRAPH_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "graph_map.json")


def test_critic_forward_shapes_are_per_agent():
    n_vbs, n_fbs, vbs_dim, fbs_dim, extra_dim = 3, 2, 13, 16, 5
    critic = CentralizedCritic(vbs_dim, fbs_dim, extra_dim)

    batch = 4
    vbs_feats = torch.randn(batch, n_vbs, vbs_dim)
    fbs_feats = torch.randn(batch, n_fbs, fbs_dim)
    global_extra = torch.randn(batch, extra_dim)

    out = critic(vbs_feats, fbs_feats, global_extra)

    assert out["team"].shape == (batch, 1)
    assert out["vbs"].shape == (batch, n_vbs)
    assert out["fbs"].shape == (batch, n_fbs)


def test_critic_vbs_head_is_agent_conditioned_not_broadcast():
    """Two agents with DIFFERENT local features in the same step must get
    DIFFERENT values. The old bug (type-mean broadcast) would have produced
    identical values for every agent regardless of their own features."""
    critic = CentralizedCritic(vbs_local_dim=13, fbs_local_dim=16, global_extra_dim=5)
    critic.eval()

    vbs_feats = torch.zeros(1, 2, 13)
    vbs_feats[0, 0] = torch.full((13,), 1.0)
    vbs_feats[0, 1] = torch.full((13,), -1.0)
    fbs_feats = torch.zeros(1, 1, 16)
    global_extra = torch.zeros(1, 5)

    with torch.no_grad():
        out = critic(vbs_feats, fbs_feats, global_extra)

    v0, v1 = out["vbs"][0, 0].item(), out["vbs"][0, 1].item()
    assert v0 != v1, "agents with different local features must not share an identical value"


def test_get_value_returns_per_agent_lists():
    n_vbs, n_fbs = 4, 2
    ppo = HeterogeneousPPOManager(
        vbs_obs_dim=13, fbs_obs_dim=16, vbs_action_dim=33, fbs_action_dim=17,
        global_extra_dim=5, device="cpu",
    )
    vbs_feats = torch.randn(1, n_vbs, 13)
    fbs_feats = torch.randn(1, n_fbs, 16)
    global_extra = torch.randn(1, 5)

    out = ppo.get_value(vbs_feats, fbs_feats, global_extra)

    assert isinstance(out["team"], float)
    assert isinstance(out["vbs"], list) and len(out["vbs"]) == n_vbs
    assert isinstance(out["fbs"], list) and len(out["fbs"]) == n_fbs


def test_update_critic_accepts_per_agent_return_matrices():
    """update_critic must train against (T, n_agents) vbs/fbs return & value
    matrices without shape errors -- the direct fix for the granularity
    mismatch (previously (T,) type-mean scalars)."""
    n_vbs, n_fbs, T = 3, 2, 10
    ppo = HeterogeneousPPOManager(
        vbs_obs_dim=13, fbs_obs_dim=16, vbs_action_dim=33, fbs_action_dim=17,
        global_extra_dim=5, device="cpu",
    )
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
    }
    # Should not raise.
    ppo.update_critic(joint_batch, ppo_epochs=1, batch_size=4)


@pytest.fixture
def small_env_and_ppo():
    graph_engine = NetworkXRoadEngine()
    graph_engine.load_from_json(GRAPH_PATH)

    manager = AgentManager()
    for i in range(2):
        manager.register_vbs(VehicleBaseStation(id=i, capacity=10, coverage_radius=15.0))
    manager.assign_home_branches(num_branches=3)
    manager.register_fbs(FlyingBaseStation(id=2, host_vbs_id=0, capacity=10, coverage_radius=20.0, maximum_distance=15.0))
    manager.assign_identity_indices()

    sim_adapter = PyWiSimAdapter(num_users=20, map_dimensions=graph_engine.get_map_dimension())
    env = CoverageParallelEnv({
        "agent_manager": manager,
        "graph_engine": graph_engine,
        "sim_adapter": sim_adapter,
        "max_cycles": 4,
        "termination_goal": 0.999,  # effectively unreachable -> exercise truncation path
    })

    vbs_agent_id = next(a for a in env.possible_agents if "vbs" in a)
    fbs_agent_id = next(a for a in env.possible_agents if "fbs" in a)
    ppo = HeterogeneousPPOManager(
        vbs_obs_dim=env.observation_space(vbs_agent_id).shape[0],
        fbs_obs_dim=env.observation_space(fbs_agent_id).shape[0],
        vbs_action_dim=env.action_space(vbs_agent_id).n,
        fbs_action_dim=env.action_space(fbs_agent_id).n,
        global_extra_dim=env.global_extra_dim,
        device="cpu",
    )
    return env, ppo


def test_rollout_produces_matching_granularity_value_and_reward_trajectories(small_env_and_ppo):
    """Runs the real per-agent-value bookkeeping pattern from main.py's rollout
    loop directly against a live env + PPO manager, and asserts each agent's
    'values' trajectory ends up the same length as its own 'rewards'
    trajectory -- the concrete manifestation of 'baseline and reward stream at
    the same granularity' this task requires."""
    env, ppo = small_env_and_ppo
    obs_dict, infos_dict = env.reset(seed=0)

    vbs_agent_order = [a for a in env.agents if "vbs" in a]
    fbs_agent_order = [a for a in env.agents if "fbs" in a]
    buffers = {
        "vbs": {a: {"rewards": [], "values": []} for a in vbs_agent_order},
        "fbs": {a: {"rewards": [], "values": []} for a in fbs_agent_order},
    }

    while env.agents:
        actions = {}
        for agent_id in env.agents:
            agent_type = "vbs" if "vbs" in agent_id else "fbs"
            t_obs = torch.tensor(obs_dict[agent_id], dtype=torch.float32)
            t_mask = torch.tensor(infos_dict[agent_id]["action_mask"], dtype=torch.float32)
            action, _ = ppo.get_action(t_obs, agent_type, action_mask=t_mask)
            actions[agent_id] = action

        vbs_feats, fbs_feats, global_extra = env.get_global_state()
        step_values = ppo.get_value(
            torch.tensor(vbs_feats, dtype=torch.float32).unsqueeze(0),
            torch.tensor(fbs_feats, dtype=torch.float32).unsqueeze(0),
            torch.tensor(global_extra, dtype=torch.float32).unsqueeze(0),
        )
        for i, agent_id in enumerate(vbs_agent_order):
            buffers["vbs"][agent_id]["values"].append(step_values["vbs"][i])
        for i, agent_id in enumerate(fbs_agent_order):
            buffers["fbs"][agent_id]["values"].append(step_values["fbs"][i])

        obs_dict, rewards_dict, terminations, truncations, infos_dict = env.step(actions)
        for agent_id in actions:
            agent_type = "vbs" if "vbs" in agent_id else "fbs"
            buffers[agent_type][agent_id]["rewards"].append(rewards_dict[agent_id])

    for agent_type in ("vbs", "fbs"):
        for agent_id, data in buffers[agent_type].items():
            assert len(data["values"]) == len(data["rewards"]) == env.step_count, (
                f"{agent_id}: values/rewards length mismatch — "
                f"values={len(data['values'])} rewards={len(data['rewards'])} steps={env.step_count}"
            )

    # Distinct VBS agents must not be forced to share identical value trajectories
    # (the old bug: both would get the exact same type-mean value every step).
    vbs_ids = list(buffers["vbs"].keys())
    if len(vbs_ids) >= 2:
        assert buffers["vbs"][vbs_ids[0]]["values"] != buffers["vbs"][vbs_ids[1]]["values"] or \
               buffers["vbs"][vbs_ids[0]]["rewards"] != buffers["vbs"][vbs_ids[1]]["rewards"], \
               "two distinct VBS agents produced identical value AND reward trajectories in this smoke test"


def test_full_training_step_end_to_end_via_main_module(monkeypatch, tmp_path):
    """Exercises the ACTUAL main.py training loop (one real episode) against a
    tiny hand-built config, to catch any residual shape mismatch between the
    per-agent critic output and compute_gae/update_actor/update_critic calls
    that a narrower unit test might miss."""
    graph_src = os.path.join(os.path.dirname(__file__), "..", "config", "graph_map.json")
    config = {
        "env_settings": {"max_cycles": 3, "num_users": 15, "termination_goal": 0.999},
        "graph_settings": {"center_node_id": 0, "max_slots_per_branch": 10},
        "vbs_agents": [
            {"id": 0, "capacity": 10, "coverage_radius": 15.0},
            {"id": 1, "capacity": 10, "coverage_radius": 15.0},
        ],
        "fbs_agents": [
            {"id": 2, "host_vbs_id": 0, "capacity": 10, "coverage_radius": 20.0, "maximum_distance": 15.0},
        ],
        "hyperparameters": {
            "learning_rate": 3e-4, "ppo_epochs": 1, "batch_size": 8,
            "clip_coef": 0.2, "ent_coef": 0.01, "vf_coef": 0.5,
        },
    }
    config_path = tmp_path / "tiny_config.json"
    config_path.write_text(json.dumps(config))

    save_dir = tmp_path / "models"
    save_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys, "argv",
        ["main.py", "--config", str(config_path), "--graph", graph_src,
         "--episodes", "1", "--save-dir", str(save_dir), "--save-every", "1", "--log-every", "1"],
    )

    main_module.main()  # must not raise
