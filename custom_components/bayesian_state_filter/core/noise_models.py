import math


class GaussianNoise:
    def __init__(self, sigma: float = 0.1, sigma_min: float = 1e-6, sigma_max: float = 1e6):
        self.sigma = float(sigma)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)

    def name(self) -> str:
        return "gaussian"

    def set_sigma(self, sigma: float) -> None:
        if not math.isfinite(float(sigma)):
            return
        self.sigma = max(self.sigma_min, min(self.sigma_max, float(sigma)))

    def variance(self, y_mean, dt, obs=None) -> float:
        if obs is not None and obs.variance is not None:
            v = float(obs.variance)
            if math.isfinite(v) and v > 0:
                return max(v, 1e-12)
        return max(self.sigma * self.sigma, 1e-12)

    def p_value(self, innovation: float, innovation_var: float) -> float:
        if innovation_var <= 0:
            return 1.0
        z = abs(float(innovation)) / math.sqrt(float(innovation_var))
        return math.erfc(z / math.sqrt(2.0))

    def info(self) -> dict:
        return {"type": "gaussian", "sigma": round(self.sigma, 8)}


class PoissonLikeNoise:
    """Variance proportional to level, for rate/count-like sensors.

    ``k`` is inferred/configured in value units.  If an Observation supplies an
    explicit variance it takes precedence, which lets source calibration and
    true exposure-aware front ends coexist with this generic model.
    """

    def __init__(self, k: float = 0.1, eps: float = 1e-9):
        self.k = max(float(k), 1e-12)
        self.eps = float(eps)

    def name(self) -> str:
        return "poisson"

    def set_scale(self, k: float) -> None:
        if math.isfinite(float(k)):
            self.k = max(float(k), 1e-12)

    def variance(self, y_mean, dt, obs=None) -> float:
        if obs is not None and obs.variance is not None:
            v = float(obs.variance)
            if math.isfinite(v) and v > 0:
                return max(v, 1e-12)
        return max(self.k * max(abs(float(y_mean)), self.eps), 1e-12)

    def p_value(self, innovation: float, innovation_var: float) -> float:
        if innovation_var <= 0:
            return 1.0
        z = abs(float(innovation)) / math.sqrt(float(innovation_var))
        return math.erfc(z / math.sqrt(2.0))

    def info(self) -> dict:
        return {"type": "poisson", "k": round(self.k, 8), "eps": self.eps}
