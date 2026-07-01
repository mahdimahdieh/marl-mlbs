class RunningNorm:
    """Welford's online algorithm — single-pass mean/std, no stored history."""
    def __init__(self, eps: float = 1e-4):
        self.mean, self.var, self.count = 0.0, 1.0, eps

    def update(self, x: float) -> None:
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        self.var += delta * (x - self.mean)

    def normalize(self, x: float) -> float:
        std = np.sqrt(self.var / max(self.count, 1.0)) + 1e-6
        return (x - self.mean) / std