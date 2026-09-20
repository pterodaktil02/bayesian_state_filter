from __future__ import annotations

import math
import numpy as np


class PolynomialStateModel:
    """Local polynomial state [x, dx/dt, ..., d^order x/dt^order]."""

    def __init__(self, order: int):
        order = int(order)
        if order < 0 or order > 3:
            raise ValueError("order must be between 0 and 3")
        self.order = order

    def dim_x(self) -> int:
        return self.order + 1

    def transition(self, dt: float):
        dt = max(float(dt), 0.0)
        n = self.dim_x()
        F = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(i, n):
                k = j - i
                F[i, j] = (dt ** k) / math.factorial(k)
        return F

    def predict(self, x, P, dt, Q):
        F = self.transition(dt)
        x_pred = F @ x
        P_pred = F @ P @ F.T + Q
        return x_pred, P_pred

    def measurement(self, x):
        return float(x[0])

    def jacobian(self, x):
        H = np.zeros((1, self.dim_x()), dtype=float)
        H[0, 0] = 1.0
        return H


class AdaptivePolynomialStateModel(PolynomialStateModel):
    """Full [x, v, a, j] state with posterior-confidence derivative gating.

    The model order never changes.  For each derivative we derive confidence
    directly from its Gaussian posterior significance ``z=|d|/sigma_d``::

        c(z) = 2*Phi(z) - 1 = erf(z / sqrt(2))

    No gate threshold or exponent is trained.  Confidence is therefore a
    probabilistic statement about the derivative estimate rather than a knob
    tuned to prediction RMSE.

    Coupling is hierarchical along the kinematic chain::

        w_v = c_v
        w_a = c_v * c_a
        w_j = c_v * c_a * c_j

    Hence 1 >= w_v >= w_a >= w_j >= 0 while all four state components remain
    estimated at all times.
    """

    def __init__(self, order: int):
        super().__init__(order)

    def gate_parameters(self):
        return {"law": "erf_abs_z_over_sqrt2"}

    @staticmethod
    def significance(value: float, variance: float) -> float:
        sigma = math.sqrt(max(float(variance), np.finfo(float).tiny))
        return abs(float(value)) / sigma

    def confidence_weight(self, value: float, variance: float, derivative_order: int = 1) -> float:
        z = self.significance(value, variance)
        if not math.isfinite(z):
            return 1.0 if z > 0 else 0.0
        if z <= 0.0:
            return 0.0
        return float(min(1.0, max(0.0, math.erf(z / math.sqrt(2.0)))))

    def derivative_weight(self, x, P, derivative_order: int, dt: float | None = None) -> float:
        j = int(derivative_order)
        if j <= 0 or j >= self.dim_x():
            return 1.0
        return self.confidence_weight(x[j], P[j, j], j)

    def confidence_weights(self, x, P):
        """Independent posterior confidence for each derivative order."""
        c = np.ones(self.dim_x(), dtype=float)
        for order in range(1, self.dim_x()):
            c[order] = self.derivative_weight(x, P, order)
        return c

    def effective_weights(self, x, P, dt: float):
        """Hierarchical coupling weights for the derivative chain."""
        c = self.confidence_weights(x, P)
        w = np.ones(self.dim_x(), dtype=float)
        cumulative = 1.0
        for order in range(1, self.dim_x()):
            cumulative *= float(c[order])
            w[order] = cumulative
        return w

    def weighted_transition(self, x, P, dt: float):
        F = self.transition(dt)
        Fw = F.copy()
        weights = self.effective_weights(x, P, dt)
        # Keep diagonal persistence untouched; gate only the way a higher
        # derivative bends lower-order states over this local step.
        for j in range(1, self.dim_x()):
            for i in range(j):
                Fw[i, j] *= weights[j]
        return Fw, weights

    def predict(self, x, P, dt, Q):
        Fw, _weights = self.weighted_transition(x, P, dt)
        x_pred = Fw @ x
        # Mean and covariance use the same local transition to avoid large
        # hidden-state kicks through ungated cross-covariances.
        P_pred = Fw @ P @ Fw.T + Q
        return x_pred, P_pred


class LevelRateCurvatureModel(PolynomialStateModel):
    """Backward-compatible x-v-a model alias."""

    def __init__(self):
        super().__init__(2)
