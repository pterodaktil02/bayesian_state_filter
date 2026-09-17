import math
import numpy as np


class LevelVelocityModel:
    """Level + exponentially decaying velocity.

    x[0] = level
    x[1] = local velocity

    ``tau`` is the characteristic time over which the current velocity loses
    predictive value.  For tau -> infinity this tends to the usual constant
    velocity model.
    """

    def __init__(self, tau: float = 3600.0):
        self.tau = max(float(tau), 1e-6)

    def dim_x(self) -> int:
        return 2

    def transition(self, dt: float, tau: float | None = None):
        dt = max(float(dt), 0.0)
        tau = max(float(self.tau if tau is None else tau), 1e-6)
        a = math.exp(-dt / tau)
        b = tau * (1.0 - a)
        return np.array([[1.0, b], [0.0, a]], dtype=float)

    def predict(self, x, P, dt, Q, tau: float | None = None):
        F = self.transition(dt, tau=tau)
        x_pred = F @ x
        P_pred = F @ P @ F.T + Q
        return x_pred, P_pred

    def measurement(self, x):
        return float(x[0])

    def jacobian(self, x):
        return np.array([[1.0, 0.0]], dtype=float)
