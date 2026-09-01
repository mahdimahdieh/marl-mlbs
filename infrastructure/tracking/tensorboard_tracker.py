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
    """Lightweight TensorBoard wrapper: SCALARS via log_episode, a full config
    dump as TEXT at init, optional IMAGES via render_frame. All writes flush
    immediately so the live UI stays current."""

    def __init__(
            self,
            project_name: str,
            config: Dict[str, Any],
            run_name: str = None,
            log_dir: str = "runs",
    ) -> None:
        """
        Args:
            project_name: Creates a subdirectory under log_dir/
            config:       Full simulation config dict — logged to the TEXT tab
            run_name:     Experiment label appended to the log path
            log_dir:      Root log directory; default "runs/" in the project root
        """
        tag = run_name or "default"
        self.log_path = os.path.join(log_dir, project_name, tag)
        self.writer = SummaryWriter(log_dir=self.log_path)

        # Full config as a markdown table → TB TEXT tab, for reproduction.
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
        """Logs one scalar per dict key (each becomes its own TB chart);
        step is the episode counter used for the x-axis."""
        for key, value in metrics.items():
            self.writer.add_scalar(
                tag=key,
                scalar_value=self._sanitize(value),
                global_step=step,
            )
        self.writer.flush()

    def render_frame(self, state_data: Dict[str, Any]) -> None:
        """Logs a Pygame RGB screenshot ("image_array", HxWx3 uint8) to TB's
        IMAGES tab, aligned via "step". Gate this in the training loop to
        avoid I/O overhead."""
        if "image_array" not in state_data or "step" not in state_data:
            return

        img = np.asarray(state_data["image_array"], dtype=np.uint8)
        if img.ndim != 3 or img.shape[2] != 3:
            # Silently skip malformed frames rather than crashing training
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
        """Flushes all pending writes and closes the SummaryWriter."""
        self.writer.close()
        print(
            f"\nTensorBoard run closed.\n"
            f"  Replay anytime: tensorboard --logdir={os.path.dirname(self.log_path)}"
        )

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _sanitize(self, value: Any) -> float:
        """Converts torch tensors, NumPy scalars and Python numerics to float."""
        if torch is not None and isinstance(value, torch.Tensor):
            # Explicit isinstance: NumPy >=2.0 scalars also expose .device,
            # so duck-typing would misclassify them as tensors.
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
        """Recursively flattens nested config dicts with '/' separator."""
        result = {}
        for key, val in d.items():
            full_key = f"{prefix}/{key}" if prefix else key
            if isinstance(val, dict):
                result.update(self._flatten_config(val, prefix=full_key))
            else:
                result[full_key] = str(val)
        return result
