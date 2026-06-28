import wandb
from typing import Dict, Any
import numpy as np
import os

try:
    import torch
except ImportError:
    torch = None

# SECURITY FIX: Remove hardcoded API key from source code.
# Set it in your shell or CI environment instead:
#   export WANDB_API_KEY="your_key_here"
# The os.environ['WANDB_API_KEY'] = '...' line has been intentionally removed.
os.environ["WANDB_MODE"] = "offline"   # Safe to keep at module level


class WandbTracker:
    def __init__(
        self,
        project_name: str,          # FIXED: now actually used (was silently ignored)
        config: Dict[str, Any],
        run_name: str = None,
        wandb_entity: str = None    # FIXED: was hardcoded username; now an optional parameter
    ):
        # FIXED: project=project_name  (was hardcoded "marl-mlbs", ignoring the argument)
        # FIXED: entity=wandb_entity   (was hardcoded "mmm4561232229")
        # wandb_entity=None is valid for offline mode; W&B resolves entity on sync.
        self.run = wandb.init(
            entity=wandb_entity,
            project=project_name,   # FIXED: now correctly consumes the injected parameter
            mode='offline',
            sync_tensorboard=True,
            name=run_name,
            config=config,
            settings=wandb.Settings(
                start_method="thread",
                mode="offline"
            )
        )
        print(
            f"W&B Initialized in OFFLINE mode. "
            f"Project: '{project_name}' | Run: '{run_name}'"
        )

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