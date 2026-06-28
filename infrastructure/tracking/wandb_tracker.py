import wandb
from typing import Dict, Any
import numpy as np

try:
    import torch
except ImportError:
    torch = None

import os

# Force offline mode so the environment never attempts to hit the W&B servers
os.environ["WANDB_MODE"] = "offline"

class WandbTracker:
    def __init__(self, project_name: str, config: Dict[str, Any], run_name: str = None):
        # We explicitly set the mode here to ensure it overrides any environment settings
        self.run = wandb.init(
            project=project_name,
            name=run_name,
            config=config,
            settings=wandb.Settings(
                start_method="thread",
                mode="offline"  # Ensures the client never tries to reach the server
            )
        )
        print("W&B Initialized in OFFLINE mode. Data will be saved locally in the 'wandb' folder.")

    def _sanitize_metrics(self, metrics: Dict[str, Any]) -> Dict[str, float]:
        """
        Sanitizes tensors. Now explicitly assumes CUDA tensors will be moved
        to CPU before being converted to standard floats.
        """
        clean_metrics = {}
        for key, value in metrics.items():
            # Check for PyTorch tensors (handling both CPU and CUDA)
            if hasattr(value, "device"):
                clean_metrics[key] = value.detach().cpu().item()
            elif isinstance(value, (np.float32, np.float64, np.integer)):
                clean_metrics[key] = value.item()
            elif isinstance(value, (float, int)):
                clean_metrics[key] = value
            else:
                raise TypeError(f"Metric '{key}' has unsupported type {type(value)}.")
        return clean_metrics

    def log_episode(self, metrics: Dict[str, float], step: int) -> None:
        """
        Pushes aggregate episodic metrics to W&B.
        Should include: Episode Reward, System Efficiency, Actor Loss, Critic Loss, Entropy.
        """
        sanitized_metrics = self._sanitize_metrics(metrics)

        # We explicitly pass the global step to ensure W&B aligns MARL rollout steps
        # accurately, rather than relying on its internal counter.
        wandb.log(sanitized_metrics, step=step)

    def render_frame(self, state_data: Dict[str, Any]) -> None:
        """
        Logs visual arrays as W&B images.
        DO NOT call this every episode. Gate it in your training loop (e.g., if episode % 100 == 0).
        """
        if "image_array" in state_data:
            # Assuming state_data["image_array"] is an RGB numpy array from pygame
            img = wandb.Image(state_data["image_array"], caption="Environment State")
            wandb.log({"Evaluation/Render": img})

    def close(self):
        """Safely shuts down the W&B thread."""
        wandb.finish()