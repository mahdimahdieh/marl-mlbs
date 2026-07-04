import numpy as np

class RunningNorm:
    """Welford's online algorithm — single-pass mean/std, no stored history."""
    def __init__(self, eps: float = 1e-4, freeze_after: int = None):
        self.mean, self.var, self.count = 0.0, 1.0, eps
        self.freeze_after = freeze_after
        self.frozen = False

    def update(self, x: float) -> None:
        if self.frozen:
            return
        self.count += 1
        if self.freeze_after is not None and self.count >= self.freeze_after:
            self.frozen = True
        delta = x - self.mean
        self.mean += delta / self.count
        self.var += delta * (x - self.mean)

    def normalize(self, x: float) -> float:
        std = np.sqrt(self.var / max(self.count, 1.0)) + 1e-6
        return float((x - self.mean) / std)