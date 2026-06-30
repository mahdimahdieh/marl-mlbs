import os
from typing import Dict, Any

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from core.interfaces.tracking_abc import TrackingABC

try:
    import torch
except ImportError:
    torch = None  # Graceful fallback: metric sanitization still works via NumPy


class TensorBoardTracker(TrackingABC):
    """
    Lightweight TensorBoard wrapper matching WandbTracker's public interface.

    Produces three types of TensorBoard artefacts:
      SCALARS  — one chart per metric key, timestep-aligned   (log_episode)
      TEXT     — full config dump in a markdown table          (__init__)
      IMAGES   — optional Pygame frame captures                (render_frame)

    All writes are flushed immediately so the live TensorBoard UI stays
    current without waiting for process exit.
    """

    def __init__(
            self,
            project_name: str,
            config: Dict[str, Any],
            run_name: str = None,
            log_dir: str = "runs",
    ) -> None:
        """
        Args:
            project_name: Creates a subdirectory under log_dir/ (mirrors WandB project)
            config:       Full simulation config dict — logged to the TEXT tab
            run_name:     Experiment label appended to the log path
            log_dir:      Root log directory; default "runs/" in the project root
        """
        tag = run_name or "default"
        # Mirrors the WandB project/run nesting:  runs/<project>/<run>/
        self.log_path = os.path.join(log_dir, project_name, tag)
        self.writer = SummaryWriter(log_dir=self.log_path)

        # ── Log full config as markdown table → visible in TB TEXT tab ──────
        # Allows exact experiment reproduction without hunting through terminal logs
        flat = self._flatten_config(config)
        header = "| Parameter | Value |\n|---|---|\n"
        rows = "\n".join(f"| `{k}` | `{v}` |" for k, v in sorted(flat.items()))
        config_md = header + rows
        self.writer.add_text("Config/Hyperparameters", config_md, global_step=0)
        self.writer.flush()

        print(
            f"\n{'─' * 60}\n"
            f"TensorBoard Tracker initialized.\n"
            f"  Log path : {self.log_path}\n"
            f"  View with: tensorboard --logdir={log_dir}\n"
            f"  Then open: http://localhost:6006\n"
            f"{'─' * 60}\n"
        )

    # ── TrackingABC interface ────────────────────────────────────────────────

    def log_episode(self, metrics: Dict[str, float], step: int) -> None:
        """
        Logs all scalar metrics. Each dict key becomes a separate chart in TB SCALARS.

        Called ONLY at terminal states (as per TrackingABC contract) to prevent
        I/O blocking during the step loop.

        Args:
            metrics: e.g. {"Episode_Reward": -12.4, "True_Coverage": 0.63, ...}
            step:    global episode counter — TB uses this for the x-axis
        """
        for key, value in metrics.items():
            self.writer.add_scalar(
                tag=key,
                scalar_value=self._sanitize(value),
                global_step=step,
            )
        # Flush immediately so the TensorBoard UI updates in real-time during training
        self.writer.flush()

    def render_frame(self, state_data: Dict[str, Any]) -> None:
        """
        Optional: logs a Pygame RGB screenshot to TensorBoard's IMAGES tab.

        Gate this in your training loop to avoid I/O overhead:
            if episode % 100 == 0:
                frame = pygame.surfarray.array3d(renderer.screen)
                tracker.render_frame({"image_array": frame.transpose(1,0,2), "step": episode})

        Expected keys in state_data:
            "image_array"  (np.ndarray, HxWx3, uint8) — RGB from pygame.surfarray
            "step"         (int) — episode / global step for TB timeline alignment
        """
        if "image_array" not in state_data or "step" not in state_data:
            return

        img = np.asarray(state_data["image_array"], dtype=np.uint8)
        if img.ndim != 3 or img.shape[2] != 3:
            # Silently skip malformed frames rather than crashing the training loop
            return

        # TensorBoard expects (C, H, W) uint8; Pygame produces (H, W, C)
        img_chw = np.transpose(img, (2, 0, 1))
        self.writer.add_image(
            tag="Eval/EnvironmentFrame",
            img_tensor=img_chw,
            global_step=int(state_data["step"]),
        )
        self.writer.flush()

    def close(self) -> None:
        """Flushes all pending writes and closes the SummaryWriter file handle."""
        self.writer.close()
        print(
            f"\nTensorBoard run closed.\n"
            f"  Replay anytime: tensorboard --logdir={os.path.dirname(self.log_path)}"
        )

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _sanitize(self, value: Any) -> float:
        """
        Converts PyTorch tensors (CPU or CUDA), NumPy scalars, and Python
        numerics to a plain Python float safe for TensorBoard's C++ backend.
        """
        if torch is not None and hasattr(value, "device"):
            # Handles both cuda:0 and cpu tensors without an extra .is_cuda check
            return float(value.detach().cpu().item())
        elif isinstance(value, (np.floating, np.integer)):
            return float(value.item())
        elif isinstance(value, (float, int)):
            return float(value)
        else:
            raise TypeError(
                f"Unsupported metric type {type(value)} for key. "
                f"Pass float, int, numpy scalar, or torch.Tensor."
            )

    def _flatten_config(self, d: Dict, prefix: str = "") -> Dict[str, str]:
        """
        Recursively flattens nested config dicts with '/' separator.
        e.g. {"hyperparameters": {"lr": 3e-4}} → {"hyperparameters/lr": "0.0003"}
        """
        result = {}
        for key, val in d.items():
            full_key = f"{prefix}/{key}" if prefix else key
            if isinstance(val, dict):
                result.update(self._flatten_config(val, prefix=full_key))
            else:
                result[full_key] = str(val)
        return result
