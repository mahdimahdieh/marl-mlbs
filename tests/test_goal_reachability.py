"""termination_goal must sit at or below the greedy placement ceiling for
the shipped config + graph — a guard against silently shipping an
unreachable goal (see tools/ceiling_probe.py)."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from main import bootstrap_environment
from rl.envs.pettingzoo_env import CoverageParallelEnv
from tools.ceiling_probe import estimate_ceiling

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "simulation_config.json")
GRAPH_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "graph_map.json")


def test_termination_goal_is_within_greedy_reach():
    env_config, _, _ = bootstrap_environment(CONFIG_PATH, GRAPH_PATH)
    env = CoverageParallelEnv(env_config)
    env.reset(seed=0)

    ceiling = estimate_ceiling(env, restarts=1, seed=0)

    assert env.termination_goal <= ceiling + 1e-9, (
        f"termination_goal={env.termination_goal} exceeds the greedy placement "
        f"ceiling {ceiling:.4f} — the goal is unreachable and every episode can "
        f"only end in truncation"
    )
