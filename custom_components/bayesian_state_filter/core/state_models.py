from __future__ import annotations

import math
import numpy as np

from ..const import NUMERIC_VARIANCE_FLOOR


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
    """Full [x, v, a, j] state with history-trained plausibility gating.

    Derivative *confidence* and derivative *plausibility* are intentionally
    different quantities:

    - ``confidence_weights`` reports posterior significance
      ``erf(|d| / sigma_posterior / sqrt(2))`` for diagnostics only;
    - ``effective_weights`` controls kinematic transport/prediction. Local
      derivative plausibility falls as magnitude becomes unusual relative to
      the robust history scale, while effective higher-order weights inherit
      every lower-order penalty.

    For derivative order k the production gate is

        w_k = exp(-0.5 * (|d_k| / sigma_hist,k)^2),  |d_k| < 5 sigma_hist,k
        w_k = 0,                                     otherwise

    ``sigma_hist`` is a Gaussian-consistent robust scale, 1.4826*MAD, learned
    from local-polynomial derivatives of the training history.  The gate is
    centred at zero on purpose: a small derivative is useful evidence that the
    process is quiet and therefore receives maximum weight.
    """

    PLAUSIBILITY_CUTOFF_SIGMA = 5.0

    def __init__(self, order: int):
        super().__init__(order)
        self._derivative_centers = np.zeros(self.dim_x(), dtype=float)
        self._derivative_scales = np.full(self.dim_x(), np.nan, dtype=float)
        self._derivative_samples = np.zeros(self.dim_x(), dtype=int)

    def set_derivative_plausibility(self, scales=None, centers=None, samples=None):
        """Install history-trained robust derivative scales.

        ``centers`` are retained for diagnostics only.  The production gate is
        deliberately centred at zero, not at the historical median derivative.
        """
        n = self.dim_x()
        out_scales = np.full(n, np.nan, dtype=float)
        out_centers = np.zeros(n, dtype=float)
        out_samples = np.zeros(n, dtype=int)
        if scales is not None:
            for i, value in enumerate(list(scales)[:n]):
                try:
                    value = float(value)
                    if math.isfinite(value) and value >= 0.0:
                        out_scales[i] = value
                except (TypeError, ValueError):
                    pass
        if centers is not None:
            for i, value in enumerate(list(centers)[:n]):
                try:
                    value = float(value)
                    if math.isfinite(value):
                        out_centers[i] = value
                except (TypeError, ValueError):
                    pass
        if samples is not None:
            for i, value in enumerate(list(samples)[:n]):
                try:
                    out_samples[i] = max(int(value), 0)
                except (TypeError, ValueError):
                    pass
        self._derivative_scales = out_scales
        self._derivative_centers = out_centers
        self._derivative_samples = out_samples

    def derivative_plausibility_parameters(self):
        return {
            "law": "gaussian_abs_derivative_over_robust_sigma",
            "cutoff_sigma": float(self.PLAUSIBILITY_CUTOFF_SIGMA),
            "centers": self._derivative_centers.tolist(),
            "scales": self._derivative_scales.tolist(),
            "samples": self._derivative_samples.tolist(),
        }

    def gate_parameters(self):
        return self.derivative_plausibility_parameters()

    @staticmethod
    def significance(value: float, variance: float) -> float:
        sigma = math.sqrt(max(float(variance), NUMERIC_VARIANCE_FLOOR))
        return abs(float(value)) / sigma

    def confidence_weight(self, value: float, variance: float, derivative_order: int = 1) -> float:
        """Posterior non-zero significance retained for diagnostics only."""
        z = self.significance(value, variance)
        if not math.isfinite(z):
            return 1.0 if z > 0 else 0.0
        if z <= 0.0:
            return 0.0
        return float(min(1.0, max(0.0, math.erf(z / math.sqrt(2.0)))))

    def plausibility_weight(self, value: float, derivative_order: int) -> float:
        order = int(derivative_order)
        if order <= 0 or order >= self.dim_x():
            return 1.0
        scale = float(self._derivative_scales[order])
        value = abs(float(value))

        # No trained scale yet: do not recreate the old positive-feedback gate.
        # A missing model means neutral coupling until history training supplies
        # a process-specific scale.
        if not math.isfinite(scale):
            return 1.0
        if scale <= 0.0:
            return 1.0 if value <= 1e-15 else 0.0

        z = value / scale
        if not math.isfinite(z) or z >= self.PLAUSIBILITY_CUTOFF_SIGMA:
            return 0.0
        return float(math.exp(-0.5 * z * z))

    def derivative_weight(self, x, P, derivative_order: int, dt: float | None = None) -> float:
        order = int(derivative_order)
        if order <= 0 or order >= self.dim_x():
            return 1.0
        return self.plausibility_weight(x[order], order)

    def confidence_weights(self, x, P):
        """Independent posterior significance for each derivative order."""
        c = np.ones(self.dim_x(), dtype=float)
        for order in range(1, self.dim_x()):
            c[order] = self.confidence_weight(x[order], P[order, order], order)
        return c

    def effective_weights(self, x, P, dt: float):
        """Hierarchical history-plausibility weights for v/a/j coupling.

        Each derivative has its own Gaussian plausibility gate, but higher
        orders inherit every lower-order penalty:

            W_v = w_v
            W_a = w_v * w_a
            W_j = w_v * w_a * w_j

        This makes the state model degrade monotonically toward a lower-order
        model as its dynamics become implausible: x-v-a-j -> x-v-a -> x-v -> x.
        Small derivatives are not suppressed: d=0 has local weight 1.  Any
        local five-sigma rejection also disables all higher derivatives.
        """
        w = np.ones(self.dim_x(), dtype=float)
        cumulative = 1.0
        for order in range(1, self.dim_x()):
            local = self.plausibility_weight(x[order], order)
            cumulative *= local
            w[order] = cumulative
        return w

    def weighted_transition(self, x, P, dt: float):
        F = self.transition(dt)
        Fw = F.copy()
        weights = self.effective_weights(x, P, dt)
        for j in range(1, self.dim_x()):
            for i in range(j):
                Fw[i, j] *= weights[j]
        return Fw, weights

    def predict(self, x, P, dt, Q):
        # Plausibility gating controls only how strongly derivatives bend the
        # mean trajectory. Covariance continues to use the full kinematic
        # transition so hidden derivatives remain observable from level data.
        F = self.transition(dt)
        Fw = F.copy()
        weights = self.effective_weights(x, P, dt)
        for j in range(1, self.dim_x()):
            for i in range(j):
                Fw[i, j] *= weights[j]

        x_pred = Fw @ x
        P_pred = F @ P @ F.T + Q
        return x_pred, P_pred


class LevelRateCurvatureModel(PolynomialStateModel):
    """Backward-compatible x-v-a model alias."""

    def __init__(self):
        super().__init__(2)
