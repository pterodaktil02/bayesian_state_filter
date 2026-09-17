import math
import numpy as np


class DampedAccelerationProcessNoise:
    """Exact discrete Q for

        dx/dt = v
        dv/dt = -v/tau + sqrt(q_acc) * white_noise

    The small-dt limit is the familiar white-acceleration covariance.
    """

    def __init__(self, q_acc: float = 1e-9, tau: float = 3600.0):
        self.q_acc = max(float(q_acc), 1e-18)
        self.tau = max(float(tau), 1e-6)

    def Q(self, dt, x=None, *, tau: float | None = None, q_acc: float | None = None):
        dt = max(float(dt), 1e-9)
        tau = max(float(self.tau if tau is None else tau), 1e-6)
        q = max(float(self.q_acc if q_acc is None else q_acc), 1e-18)
        u = dt / tau

        # Series form avoids catastrophic cancellation for dt << tau.
        if u < 1e-4:
            return q * np.array([
                [dt**3 / 3.0, dt**2 / 2.0],
                [dt**2 / 2.0, dt],
            ], dtype=float)

        a = math.exp(-u)
        qvv = q * tau * 0.5 * (1.0 - a * a)
        qxv = q * tau * tau * 0.5 * (1.0 - a) ** 2
        qxx = q * tau * tau * (
            dt - 2.0 * tau * (1.0 - a) + 0.5 * tau * (1.0 - a * a)
        )
        Q = np.array([[qxx, qxv], [qxv, qvv]], dtype=float)
        return 0.5 * (Q + Q.T)


# Compatibility alias for old imports.
WhiteAccelerationProcessNoise = DampedAccelerationProcessNoise
