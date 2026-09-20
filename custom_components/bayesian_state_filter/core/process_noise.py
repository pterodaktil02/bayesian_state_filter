from __future__ import annotations

import math
import numpy as np


class IntegratedWienerProcessNoise:
    """White noise on the derivative immediately above the state order.

    State order 0: x, white velocity noise.
    State order 1: x-v, white acceleration noise.
    State order 2: x-v-a, white jerk noise.
    State order 3: x-v-a-j, white snap noise.

    q is intentionally not physically clamped. q=0 is a valid deterministic
    polynomial hypothesis. Numerical regularization belongs in covariance
    algebra, not in the physical parameter.
    """

    def __init__(self, order: int, q: float = 0.0):
        self.order = int(order)
        if self.order < 0 or self.order > 3:
            raise ValueError("order must be between 0 and 3")
        self.q = self._clean_q(q)

    @staticmethod
    def _clean_q(value: float) -> float:
        value = float(value)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("process noise q must be finite and >= 0")
        return value

    def Q(self, dt: float, q: float | None = None, **_kwargs):
        qv = self.q if q is None else self._clean_q(q)
        n = self.order
        size = n + 1
        dt = max(float(dt), np.finfo(float).tiny)
        Q = np.zeros((size, size), dtype=float)
        if qv == 0.0:
            return Q
        for i in range(size):
            for j in range(size):
                power = 2 * n + 1 - i - j
                denom = power * math.factorial(n - i) * math.factorial(n - j)
                Q[i, j] = qv * (dt ** power) / denom
        return Q


class WhiteJerkProcessNoise(IntegratedWienerProcessNoise):
    """Backward-compatible x-v-a white-jerk process noise alias."""

    def __init__(self, q_jerk: float = 0.0):
        super().__init__(2, q_jerk)

    @property
    def q_jerk(self) -> float:
        return self.q

    @q_jerk.setter
    def q_jerk(self, value: float) -> None:
        self.q = self._clean_q(value)

    def Q(self, dt: float, q_jerk: float | None = None, **_kwargs):
        return super().Q(dt, q=self.q if q_jerk is None else q_jerk)
