import numpy as np


class RunningNorm:
    """Welford-style normalizer with a decay window + std floor.

    FIXED: lifetime variance collapsed toward ~0 after convergence, so
    normalize() divided by a near-zero std and produced reward spikes.
    decay bounds the effective sample size (recent-window tracking);
    min_std/clip floor the output as a second safeguard.
    """

    def __init__(self, eps: float = 1e-4, decay: float = 0.999,
                 min_std: float = 0.05, clip: float = 5.0, freeze_after: int = None):
        self.mean, self.var = 0.0, 0.0
        self.count = eps
        self._eff_n = eps
        self.decay = decay
        self.min_std = min_std
        self.clip = clip
        self.freeze_after = freeze_after
        self.frozen = False

    def update(self, x: float) -> None:
        if self.frozen:
            return
        self.count += 1
        if self.freeze_after is not None and self.count >= self.freeze_after:
            self.frozen = True

        self._eff_n = min(self._eff_n + 1.0, 1.0 / (1.0 - self.decay))  # bounded window
        delta = x - self.mean
        self.mean += delta / self._eff_n
        self.var += (delta * (x - self.mean) - self.var) / self._eff_n

    def normalize(self, x: float) -> float:
        std = max(np.sqrt(max(self.var, 0.0)), self.min_std)
        return float(np.clip((x - self.mean) / std, -self.clip, self.clip))