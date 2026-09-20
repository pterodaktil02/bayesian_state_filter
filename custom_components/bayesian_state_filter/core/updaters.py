from __future__ import annotations

import math
import numpy as np

from ..const import NUMERIC_VARIANCE_FLOOR


class GaussianUpdater:
    """Plain scalar Gaussian update using observation variance directly."""

    def update(self, x_pred, P_pred, obs, dt, state_model):
        H = state_model.jacobian(x_pred)
        z_pred = float(state_model.measurement(x_pred))
        R = max(float(obs.variance), NUMERIC_VARIANCE_FLOOR)
        S = float((H @ P_pred @ H.T)[0, 0] + R)
        S = max(S, NUMERIC_VARIANCE_FLOOR)
        innovation = float(obs.z - z_pred)
        K = (P_pred @ H.T) / S
        x_post = x_pred + K[:, 0] * innovation
        I = np.eye(P_pred.shape[0], dtype=float)
        KH = K @ H
        P_post = (I - KH) @ P_pred @ (I - KH).T + K * R @ K.T
        P_post = 0.5 * (P_post + P_post.T)
        loglik = -0.5 * (math.log(2.0 * math.pi * S) + innovation * innovation / S)
        return x_post, P_post, {
            "innovation": innovation,
            "innovation_var": S,
            "measurement_var": R,
            "effective_innovation_var": S,
            "weight": 1.0,
            "loglik": loglik,
        }


class StudentTUpdater:
    """Robust one-dimensional observation update.

    This implementation is intentionally identical to the proven
    bayesian_trend_filter v0.6.3 updater.  In particular, Student-t inliers may
    have weight > 1 and therefore a slightly smaller effective R; do not clamp
    the upper side to 1.0.
    """

    def __init__(self, nu: float = 4.0, min_weight: float = 0.05):
        self.nu = max(float(nu), 1.01)
        self.min_weight = max(min(float(min_weight), 1.0), 1e-6)

    def update(self, x_pred, P_pred, obs, dt, state_model):
        H = state_model.jacobian(x_pred)
        z_pred = float(state_model.measurement(x_pred))
        R = max(float(obs.variance), NUMERIC_VARIANCE_FLOOR)
        S = float((H @ P_pred @ H.T)[0, 0] + R)
        S = max(S, NUMERIC_VARIANCE_FLOOR)
        innovation = float(obs.z - z_pred)
        z2 = (innovation * innovation) / S
        weight = (self.nu + 1.0) / (self.nu + z2)
        weight = max(float(weight), self.min_weight)
        R_eff = R / weight
        S_eff = float((H @ P_pred @ H.T)[0, 0] + R_eff)
        S_eff = max(S_eff, NUMERIC_VARIANCE_FLOOR)
        K = (P_pred @ H.T) / S_eff
        x_post = x_pred + (K[:, 0] * innovation)
        I = np.eye(P_pred.shape[0], dtype=float)
        KH = K @ H
        P_post = (I - KH) @ P_pred @ (I - KH).T + K * R_eff @ K.T
        P_post = 0.5 * (P_post + P_post.T)
        sign, logdet = np.linalg.slogdet(np.array([[S]], dtype=float))
        if sign <= 0:
            loglik = -0.5 * z2
        else:
            loglik = -0.5 * (math.log(2.0 * math.pi) + logdet + z2)
        return x_post, P_post, {
            "innovation": innovation,
            "innovation_var": S,
            "measurement_var": R,
            "effective_innovation_var": S_eff,
            "weight": weight,
            "loglik": loglik,
        }
