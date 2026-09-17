"""Predictive damped-velocity hyperparameter inference.

The tau estimated here is the memory of the local slope (``velocity_tau``),
not the characteristic time of the level process.  The latter is estimated
independently in :mod:`variogram`.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np


@dataclass
class DynamicsEstimate:
    tau: float
    q_acc: float
    p10: float
    p90: float
    confidence: float
    entropy_confidence: float
    samples: int
    identifiable: bool
    edge_mass: float = 0.0
    boundary_limited: bool = False


class DynamicsBank:
    """Vectorized velocity-tau x process-noise model bank.

    Every hypothesis is a 2-state damped-velocity Kalman model.  Model evidence
    is Student-t predictive likelihood, so isolated bad samples do not get
    interpreted as a radically shorter slope-memory time.
    """

    def __init__(self, tau_grid, q_factors=(1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0), nu: float = 4.0):
        self.tau_grid = np.asarray(list(tau_grid), dtype=float)
        self.q_factors = np.asarray(list(q_factors), dtype=float)
        self.nu = max(float(nu), 1.01)
        self.samples = 0
        self.forget_time_s = None
        self._last_event_t = None
        self._ready = False

        # Constant part of Student-t log density.
        self._t_const = (
            math.lgamma((self.nu + 1.0) / 2.0)
            - math.lgamma(self.nu / 2.0)
            - 0.5 * math.log(self.nu * math.pi)
        )

    def initialize(self, t0: float, x0: float, var0: float, slope_scale: float):
        taus = []
        qs = []
        slope2 = max(float(slope_scale) ** 2, 1e-20)
        for tau in self.tau_grid:
            q_ref = max(2.0 * slope2 / tau, 1e-18)
            for fac in self.q_factors:
                taus.append(float(tau))
                qs.append(float(q_ref * fac))

        self._tau = np.asarray(taus, dtype=float)
        self._q = np.asarray(qs, dtype=float)
        n = len(self._tau)
        self._x0 = np.full(n, float(x0), dtype=float)
        self._x1 = np.zeros(n, dtype=float)
        v0 = max(float(var0), 1e-12)
        self._p00 = np.full(n, v0, dtype=float)
        self._p01 = np.zeros(n, dtype=float)
        self._p11 = np.maximum(v0 / (self._tau * self._tau), 1e-18)
        self._t = np.full(n, float(t0), dtype=float)
        self._logw = np.zeros(n, dtype=float)
        self.samples = 0
        self._last_event_t = float(t0)
        self._ready = True

    def _q_terms(self, dt):
        """Exact OU-velocity process covariance, vectorized by hypothesis."""
        dt = np.maximum(dt, 1e-9)
        tau = self._tau
        q = self._q
        u = dt / tau
        a = np.exp(-u)

        qvv = q * tau * 0.5 * (1.0 - a * a)
        qxv = q * tau * tau * 0.5 * (1.0 - a) ** 2
        qxx = q * tau * tau * (
            dt - 2.0 * tau * (1.0 - a) + 0.5 * tau * (1.0 - a * a)
        )

        small = u < 1e-4
        if np.any(small):
            d = dt[small]
            qq = q[small]
            qxx[small] = qq * d**3 / 3.0
            qxv[small] = qq * d**2 / 2.0
            qvv[small] = qq * d
        return a, qxx, qxv, qvv

    def update(self, t: float, z: float, variance: float):
        if not self._ready:
            return
        t = float(t)
        z = float(z)
        variance = max(float(variance), 1e-12)

        if self.forget_time_s and self._last_event_t is not None:
            elapsed = max(t - self._last_event_t, 0.0)
            decay = math.exp(-elapsed / max(float(self.forget_time_s), 1.0))
            if decay < 0.999999:
                self._logw = (self._logw - np.max(self._logw)) * decay
        self._last_event_t = max(t, self._last_event_t or t)

        dt = np.maximum(t - self._t, 1e-3)
        self._t = np.maximum(self._t, t)
        a, qxx, qxv, qvv = self._q_terms(dt)
        b = self._tau * (1.0 - a)

        # Predict state.
        x0p = self._x0 + b * self._x1
        x1p = a * self._x1

        # Predict symmetric covariance FPF' + Q.
        p00p = self._p00 + 2.0 * b * self._p01 + b * b * self._p11 + qxx
        p01p = a * (self._p01 + b * self._p11) + qxv
        p11p = a * a * self._p11 + qvv
        p00p = np.maximum(p00p, 1e-15)
        p11p = np.maximum(p11p, 1e-18)

        innovation = z - x0p
        S = np.maximum(p00p + variance, 1e-15)
        z2 = innovation * innovation / S

        self._logw += (
            self._t_const
            - 0.5 * np.log(S)
            - 0.5 * (self.nu + 1.0) * np.log1p(z2 / self.nu)
        )

        # Always-on robust update.  Cap at one: inliers never get more gain
        # than their Gaussian counterpart.
        rw = np.minimum(1.0, np.maximum(1e-4, (self.nu + 1.0) / (self.nu + z2)))
        S_eff = S / rw
        k0 = p00p / S_eff
        k1 = p01p / S_eff
        self._x0 = x0p + k0 * innovation
        self._x1 = x1p + k1 * innovation

        # Equivalent to the Joseph update for scalar H=[1,0] and the effective
        # Student-t measurement variance, but cheaper and symmetric by form.
        self._p00 = np.maximum(p00p - k0 * p00p, 1e-15)
        self._p01 = p01p - k0 * p01p
        self._p11 = np.maximum(p11p - k1 * p01p, 1e-18)

        self.samples += 1
        if self.samples % 256 == 0:
            self._logw -= np.max(self._logw)

    def _weights(self):
        if not self._ready:
            return np.array([])
        lw = self._logw - np.max(self._logw)
        w = np.exp(np.clip(lw, -745.0, 0.0))
        s = float(w.sum())
        return w / s if s > 0 else np.ones_like(w) / len(w)

    @staticmethod
    def _weighted_quantile(values, weights, q):
        order = np.argsort(values)
        v = np.asarray(values, dtype=float)[order]
        w = np.asarray(weights, dtype=float)[order]
        c = np.cumsum(w)
        if len(c) == 0 or c[-1] <= 0:
            return float(np.median(v))
        c /= c[-1]
        return float(np.interp(float(q), c, v))

    def estimate(self) -> DynamicsEstimate | None:
        if not self._ready:
            return None
        w = self._weights()
        tw = np.zeros(len(self.tau_grid), dtype=float)
        # Hypotheses are laid out tau-major, then q-factor.
        nq = len(self.q_factors)
        for i in range(len(self.tau_grid)):
            tw[i] = float(w[i * nq:(i + 1) * nq].sum())
        tw /= max(float(tw.sum()), 1e-30)

        p10 = self._weighted_quantile(self.tau_grid, tw, 0.10)
        p50 = self._weighted_quantile(self.tau_grid, tw, 0.50)
        p90 = self._weighted_quantile(self.tau_grid, tw, 0.90)
        q50 = self._weighted_quantile(self._q, w, 0.50)

        full = max(math.log(self.tau_grid[-1] / self.tau_grid[0]), 1e-9)
        width = max(math.log(max(p90, p10 * 1.000001) / p10), 0.0)
        width_conf = max(0.0, min(1.0, 1.0 - width / full))
        entropy = -float(np.sum(tw * np.log(np.maximum(tw, 1e-300))))
        max_entropy = math.log(max(len(tw), 2))
        entropy_conf = max(0.0, min(1.0, 1.0 - entropy / max_entropy))
        raw_confidence = 0.6 * width_conf + 0.4 * entropy_conf
        edge_mass = float(tw[0] + tw[-1])
        # A narrow posterior pinned to a search boundary is not a high-confidence
        # identification; it is a censored lower/upper bound.  Penalize the
        # reported confidence accordingly instead of returning a misleading 0.9+.
        confidence = raw_confidence * max(0.0, 1.0 - edge_mass)
        boundary_limited = edge_mass >= 0.25
        identifiable = (
            self.samples >= 20
            and confidence >= 0.35
            and edge_mass < 0.50
        )

        return DynamicsEstimate(
            tau=float(p50), q_acc=float(q50), p10=float(p10), p90=float(p90),
            confidence=float(confidence), entropy_confidence=float(entropy_conf),
            samples=int(self.samples), identifiable=bool(identifiable),
            edge_mass=float(edge_mass), boundary_limited=bool(boundary_limited),
        )
