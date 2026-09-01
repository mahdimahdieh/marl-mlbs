import numpy as np


class StationaryScaler:
    """Fixed, time-invariant scaling for bounded metrics in [0.0, 1.0].

    Keeps no running statistics (unlike RunningNorm, whose mean tracked the
    policy and zeroed out converged coverage rewards); maintaining high
    coverage therefore yields a monotonic, positive signal in every episode.
    """

    def __init__(self, scale: float = 1.0, offset: float = 0.0):
        self.scale = float(scale)
        self.offset = float(offset)

    def __call__(self, x: float) -> float:
        return self.offset + self.scale * float(np.clip(x, 0.0, 1.0))


class RunningNorm:
    """Welford-style normalizer with a decay window, std floor and output clip.

    Only for genuinely unbounded signals; bounded [0, 1] metrics must use
    StationaryScaler (see class docstring there).
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
