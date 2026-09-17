import math
import numpy as np

from ..const import NUMERIC_VARIANCE_FLOOR, STUDENT_T_MIN_WEIGHT


def _measurement_variance(noise_model, y, dt, obs):
    try:
        return float(noise_model.variance(y, dt, obs))
    except TypeError:
        return float(noise_model.variance(y, dt))


def _joseph_update(P_pred, K, H, R_eff):
    I = np.eye(P_pred.shape[0])
    IKH = I - K @ H
    P = IKH @ P_pred @ IKH.T + (K @ K.T) * float(R_eff)
    P = 0.5 * (P + P.T)
    # Tiny floor against roundoff-driven loss of PSD.
    vals, vecs = np.linalg.eigh(P)
    vals = np.maximum(vals, NUMERIC_VARIANCE_FLOOR)
    return (vecs * vals) @ vecs.T


class GaussianUpdater:
    def update(self, x_pred, P_pred, obs, dt, state_model, noise_model):
        H = state_model.jacobian(x_pred)
        y = state_model.measurement(x_pred)
        R = _measurement_variance(noise_model, y, dt, obs)
        v = float(obs.z - y)
        A = float((H @ P_pred @ H.T).item())
        S = max(A + R, NUMERIC_VARIANCE_FLOOR)
        K = (P_pred @ H.T) / S
        x_post = x_pred + K.flatten() * v
        P_post = _joseph_update(P_pred, K, H, R)
        loglik = -0.5 * (math.log(2.0 * math.pi * S) + v * v / S)
        return x_post, P_post, {
            "innovation": v,
            "innovation_var": S,
            "weight": 1.0,
            "loglik": loglik,
            "measurement_var": R,
        }


class StudentTUpdater:
    """Always-on Student-t robust update.

    The robust weight scales the *Gaussian gain itself*, rather than merely
    inflating R.  This remains robust even when predicted state uncertainty is
    much larger than measurement noise.
    """

    def __init__(self, nu: float = 4.0, min_weight: float = STUDENT_T_MIN_WEIGHT):
        self.nu = max(float(nu), 1.01)
        self.min_weight = max(float(min_weight), 1e-9)

    def update(self, x_pred, P_pred, obs, dt, state_model, noise_model):
        H = state_model.jacobian(x_pred)
        y = state_model.measurement(x_pred)
        R = _measurement_variance(noise_model, y, dt, obs)
        v = float(obs.z - y)
        A = float((H @ P_pred @ H.T).item())
        S = max(A + R, NUMERIC_VARIANCE_FLOOR)
        z2 = v * v / S

        # For inliers Student-t may produce w > 1; never give more gain than
        # the Gaussian filter, only reduce outlier influence.
        w = min(1.0, max(self.min_weight, (self.nu + 1.0) / (self.nu + z2)))
        S_eff = S / w
        R_eff = max(R, S_eff - A)
        K = (P_pred @ H.T) / S_eff

        x_post = x_pred + K.flatten() * v
        P_post = _joseph_update(P_pred, K, H, R_eff)

        loglik = (
            math.lgamma((self.nu + 1.0) / 2.0)
            - math.lgamma(self.nu / 2.0)
            - 0.5 * math.log(self.nu * math.pi * S)
            - 0.5 * (self.nu + 1.0) * math.log1p(z2 / self.nu)
        )
        return x_post, P_post, {
            "innovation": v,
            "innovation_var": S,
            "effective_innovation_var": S_eff,
            "weight": w,
            "loglik": loglik,
            "measurement_var": R,
        }
