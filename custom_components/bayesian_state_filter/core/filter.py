import math
import numpy as np

from ..const import NUMERIC_VARIANCE_FLOOR
from .types import Observation, PosteriorSummary


class CoreFilter:
    """Robust Bayesian state filter for a generic polynomial state.

    Production v0.3 uses a fixed full [x, v, a, j] state.  ``tau`` and
    ``q_acc`` remain compatibility aliases for old persistence/diagnostics;
    internally they map to the prior time scale and generic process-noise q.
    """

    def __init__(self, state_model, noise_model, updater, process_noise, prior_timescale_s: float = 60.0):
        self.state_model = state_model
        self.noise_model = noise_model
        self.updater = updater
        self.process_noise = process_noise
        self.prior_timescale_s = max(float(prior_timescale_s), np.finfo(float).eps)

        self.x = np.zeros(state_model.dim_x(), dtype=float)
        self.P = np.eye(state_model.dim_x(), dtype=float)
        self.t_last = None
        self.mode = "prior"
        self._last_pred_var = None

    @property
    def q_process(self) -> float:
        if hasattr(self.process_noise, "q"):
            return float(self.process_noise.q)
        if hasattr(self.process_noise, "q_jerk"):
            return float(self.process_noise.q_jerk)
        return float(getattr(self.process_noise, "q_acc", 0.0))

    @q_process.setter
    def q_process(self, value: float) -> None:
        value = float(value)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("q_process must be finite and >= 0")
        if hasattr(self.process_noise, "q"):
            self.process_noise.q = value
        elif hasattr(self.process_noise, "q_jerk"):
            self.process_noise.q_jerk = value
        else:
            self.process_noise.q_acc = value


    @property
    def level_q_process(self) -> float:
        return float(getattr(self.process_noise, "level_q", 0.0))

    @level_q_process.setter
    def level_q_process(self, value: float) -> None:
        value = float(value)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("level_q_process must be finite and >= 0")
        if hasattr(self.process_noise, "level_q"):
            self.process_noise.level_q = value

    @property
    def tau(self) -> float:
        return float(self.prior_timescale_s)

    @tau.setter
    def tau(self, value: float) -> None:
        self.prior_timescale_s = max(float(value), np.finfo(float).eps)

    @property
    def q_acc(self) -> float:
        return self.q_process

    @q_acc.setter
    def q_acc(self, value: float) -> None:
        self.q_process = value

    def reset(self, value: float, *, t: float | None = None, variance: float | None = None,
              timescale_s: float | None = None) -> None:
        var = max(float(variance) if variance is not None else 1.0, NUMERIC_VARIANCE_FLOOR)
        T = max(float(self.prior_timescale_s if timescale_s is None else timescale_s), np.finfo(float).eps)
        self.prior_timescale_s = T
        self.x[:] = 0.0
        self.x[0] = float(value)
        diag = [var]
        tiny = np.finfo(float).tiny
        for derivative_order in range(1, self.state_model.dim_x()):
            diag.append(max(var / (T ** (2 * derivative_order)), tiny))
        self.P = np.diag(diag).astype(float)
        self.t_last = None if t is None else float(t)
        self._last_pred_var = None
        self.mode = "prior"


    def reset_untrusted_derivative_tail(self, *, weight_threshold: float = 0.01):
        """Reset a low-plausibility derivative tail to a neutral learned prior.

        This is the runtime recovery path for a derivative that has effectively
        dropped out of the state model.  No secondary polynomial estimator is
        involved: the first derivative whose historical plausibility weight is
        at or below ``weight_threshold`` invalidates itself and all higher
        derivatives.  Their means are set to zero, every covariance touching
        the invalid tail is cleared, and each diagonal variance is restored from
        the history-trained derivative sigma (falling back to the existing
        posterior variance only if no trained scale is available).

        The level and every lower-order derivative remain untouched.  A zero
        mean here is a neutral reset prior, not a claim that the physical
        derivative is exactly zero; the learned variance gives the Bayesian
        filter room to re-estimate the dynamics from fresh level observations.
        """
        threshold = float(weight_threshold)
        if not math.isfinite(threshold) or threshold < 0.0 or threshold > 1.0:
            raise ValueError("weight_threshold must be finite and in [0, 1]")

        model = self.state_model
        if not hasattr(model, "derivative_plausibility_parameters"):
            return None
        params = model.derivative_plausibility_parameters()
        scales = list(params.get("scales", []) or [])

        start_order = None
        trigger_weight = None
        trigger_z = None
        trigger_scale = None
        for order in range(1, self.state_model.dim_x()):
            try:
                weight = float(model.derivative_weight(self.x, self.P, order))
            except Exception:
                weight = 1.0
            if math.isfinite(weight) and weight <= threshold:
                start_order = order
                trigger_weight = weight
                scale = None
                if order < len(scales):
                    try:
                        candidate = float(scales[order])
                        if math.isfinite(candidate) and candidate > 0.0:
                            scale = candidate
                    except (TypeError, ValueError):
                        pass
                trigger_scale = scale
                if scale is not None:
                    trigger_z = abs(float(self.x[order])) / scale
                break

        if start_order is None:
            return None

        old_x = np.array(self.x, dtype=float, copy=True)
        old_P = np.array(self.P, dtype=float, copy=True)
        P_new = np.array(old_P, dtype=float, copy=True)
        restored_sigmas = {}

        for order in range(start_order, self.state_model.dim_x()):
            self.x[order] = 0.0
            P_new[order, :] = 0.0
            P_new[:, order] = 0.0

            scale = None
            if order < len(scales):
                try:
                    candidate = float(scales[order])
                    if math.isfinite(candidate) and candidate > 0.0:
                        scale = candidate
                except (TypeError, ValueError):
                    pass
            if scale is None:
                scale = math.sqrt(max(float(old_P[order, order]), NUMERIC_VARIANCE_FLOOR))
            P_new[order, order] = max(scale * scale, NUMERIC_VARIANCE_FLOOR)
            restored_sigmas[order] = float(scale)

        self.P = 0.5 * (P_new + P_new.T)
        self._last_pred_var = float(self.P[0, 0])
        return {
            "from_order": int(start_order),
            "trigger_z": float(trigger_z) if trigger_z is not None else float("nan"),
            "trigger_weight": float(trigger_weight),
            "trigger_scale": float(trigger_scale) if trigger_scale is not None else float("nan"),
            "old_derivatives": old_x[start_order:].tolist(),
            "restored_sigmas": restored_sigmas,
            "applied_hints": {},
            "reason": "low_derivative_weight_reset",
        }

    def recondition_implausible_derivatives(self, derivative_hints=None):
        """Replace an impossible derivative tail with its learned prior.

        The level is left untouched. If the first historically implausible
        derivative is order k, means k..N are reset to a zero-centred prior,
        every covariance touching that tail is cleared, and each tail variance
        is restored to at least its history-trained robust derivative variance.

        Optional ``derivative_hints`` are independent pseudo-measurements for
        derivative components, keyed by derivative order. Each value is a
        ``(mean, sigma)`` pair. They are applied *after* the tail has been
        reconditioned, so a falsely tiny stale posterior variance cannot make
        the filter ignore the external hint. This is intentionally only a
        recovery path; normal tracking still estimates derivatives from the
        state-space model and level observations.

        The operation is hierarchical: an impossible velocity invalidates
        velocity/acceleration/jerk together; an impossible acceleration
        invalidates acceleration/jerk; jerk alone only resets jerk.
        """
        model = self.state_model
        if not hasattr(model, "first_rejected_derivative"):
            return None
        rejected = model.first_rejected_derivative(self.x)
        if rejected is None:
            return None

        start_order, trigger_z, trigger_scale = rejected
        params = (
            model.derivative_plausibility_parameters()
            if hasattr(model, "derivative_plausibility_parameters") else {}
        )
        scales = list(params.get("scales", []) or [])

        old_x = np.array(self.x, dtype=float, copy=True)
        old_P = np.array(self.P, dtype=float, copy=True)
        P_new = np.array(old_P, dtype=float, copy=True)

        # Quarantine the whole invalid derivative tail from the otherwise
        # healthy level/lower-order posterior. A block-diagonal replacement is
        # positive semidefinite when the retained principal block is PSD.
        for order in range(int(start_order), self.state_model.dim_x()):
            self.x[order] = 0.0
            P_new[order, :] = 0.0
            P_new[:, order] = 0.0

        restored_sigmas = {}
        for order in range(int(start_order), self.state_model.dim_x()):
            scale = None
            if order < len(scales):
                try:
                    candidate = float(scales[order])
                    if math.isfinite(candidate) and candidate > 0.0:
                        scale = candidate
                except (TypeError, ValueError):
                    pass
            if scale is None:
                old_var = max(float(old_P[order, order]), NUMERIC_VARIANCE_FLOOR)
                scale = math.sqrt(old_var)
            prior_var = max(scale * scale, NUMERIC_VARIANCE_FLOOR)
            P_new[order, order] = prior_var
            restored_sigmas[order] = scale

        self.P = 0.5 * (P_new + P_new.T)

        # Apply independent derivative witnesses as ordinary scalar Kalman
        # pseudo-measurements. Because the invalid tail has just been reset to
        # a historically calibrated prior, these updates are not defeated by
        # the stale overconfident posterior that triggered recovery.
        applied_hints = {}
        hints = derivative_hints or {}
        for order in range(int(start_order), self.state_model.dim_x()):
            hint = hints.get(order)
            if hint is None:
                continue
            try:
                z_hint, sigma_hint = float(hint[0]), float(hint[1])
            except (TypeError, ValueError, IndexError):
                continue
            if not (math.isfinite(z_hint) and math.isfinite(sigma_hint) and sigma_hint > 0.0):
                continue

            R = max(sigma_hint * sigma_hint, NUMERIC_VARIANCE_FLOOR)
            S = max(float(self.P[order, order]) + R, NUMERIC_VARIANCE_FLOOR)
            K = np.array(self.P[:, order], dtype=float, copy=True) / S
            innovation = z_hint - float(self.x[order])
            self.x = self.x + K * innovation

            # Joseph-form scalar covariance update for H=e_order.
            dim = self.state_model.dim_x()
            H = np.zeros((1, dim), dtype=float)
            H[0, order] = 1.0
            I = np.eye(dim, dtype=float)
            KH = np.outer(K, H[0])
            A = I - KH
            self.P = A @ self.P @ A.T + np.outer(K, K) * R
            self.P = 0.5 * (self.P + self.P.T)

            applied_hints[order] = {
                "value": z_hint,
                "sigma": sigma_hint,
                "posterior_value": float(self.x[order]),
                "posterior_sigma": math.sqrt(max(float(self.P[order, order]), 0.0)),
            }

        self._last_pred_var = float(self.P[0, 0])
        return {
            "from_order": int(start_order),
            "trigger_z": float(trigger_z),
            "trigger_scale": float(trigger_scale),
            "old_derivatives": old_x[int(start_order):].tolist(),
            "restored_sigmas": restored_sigmas,
            "applied_hints": applied_hints,
        }

    def recondition_derivatives_from_hints(self, *, start_order: int = 1, derivative_hints=None, reason: str = "external_witness"):
        """Hard re-anchor a derivative tail from an independent witness.

        The caller has already validated the witness (N+span, multiscale
        consistency, history plausibility, and for live recovery persistent
        divergence).  Therefore this is deliberately *not* another soft
        Kalman pseudo-update.  A soft update can leave enough stale covariance
        in the derivative tail for the next level measurement to immediately
        drive curvature/jerk back into the pathological state.

        Hinted orders are set directly to the witness value with witness
        variance.  Cross-covariances for the repaired tail are cleared.
        Unhinted higher orders (normally jerk) are reset to their trained prior
        center, while their variance is capped by what can be generated from
        the nearest lower repaired derivative over one characteristic time.
        """
        dim = self.state_model.dim_x()
        start_order = int(start_order)
        if start_order <= 0 or start_order >= dim:
            return None

        model = self.state_model
        params = (
            model.derivative_plausibility_parameters()
            if hasattr(model, "derivative_plausibility_parameters") else {}
        )
        scales = list(params.get("scales", []) or [])
        centers = list(params.get("centers", []) or [])
        hints = derivative_hints or {}

        old_x = np.array(self.x, dtype=float, copy=True)
        old_P = np.array(self.P, dtype=float, copy=True)
        P_new = np.array(old_P, dtype=float, copy=True)

        # Decouple the whole repaired derivative tail from level and from each
        # other.  The normal prediction/update cycle can build physically
        # justified correlations again from this clean local start.
        for order in range(start_order, dim):
            P_new[order, :] = 0.0
            P_new[:, order] = 0.0

        restored_sigmas = {}
        applied_hints = {}
        previous_sigma = None
        tau = max(float(self.prior_timescale_s), np.finfo(float).eps)

        for order in range(start_order, dim):
            trained_sigma = None
            if order < len(scales):
                try:
                    candidate = float(scales[order])
                    if math.isfinite(candidate) and candidate > 0.0:
                        trained_sigma = candidate
                except (TypeError, ValueError):
                    pass
            if trained_sigma is None:
                trained_sigma = math.sqrt(max(float(old_P[order, order]), NUMERIC_VARIANCE_FLOOR))

            prior_center = 0.0
            if order < len(centers):
                try:
                    candidate = float(centers[order])
                    if math.isfinite(candidate):
                        prior_center = candidate
                except (TypeError, ValueError):
                    pass

            hint = hints.get(order)
            if hint is not None:
                try:
                    z_hint, sigma_hint = float(hint[0]), float(hint[1])
                except (TypeError, ValueError, IndexError):
                    hint = None
                else:
                    if not (math.isfinite(z_hint) and math.isfinite(sigma_hint) and sigma_hint > 0.0):
                        hint = None

            if hint is not None:
                # The witness has already earned trust through the external
                # gates.  Anchor exactly rather than blending with stale state.
                sigma_use = max(float(sigma_hint), math.sqrt(NUMERIC_VARIANCE_FLOOR))
                self.x[order] = float(z_hint)
                P_new[order, order] = max(sigma_use * sigma_use, NUMERIC_VARIANCE_FLOOR)
                previous_sigma = sigma_use
                applied_hints[order] = {
                    "value": float(z_hint),
                    "sigma": sigma_use,
                    "posterior_value": float(self.x[order]),
                    "posterior_sigma": sigma_use,
                }
            else:
                # Normally this is jerk.  Reset its mean to the trained prior
                # and prevent a huge historical variance from instantly
                # contaminating the repaired lower derivative through F(dt).
                sigma_use = trained_sigma
                if previous_sigma is not None:
                    sigma_use = min(sigma_use, previous_sigma / tau)
                sigma_use = max(float(sigma_use), math.sqrt(NUMERIC_VARIANCE_FLOOR))
                self.x[order] = prior_center
                P_new[order, order] = max(sigma_use * sigma_use, NUMERIC_VARIANCE_FLOOR)
                previous_sigma = sigma_use

            restored_sigmas[order] = sigma_use

        self.P = 0.5 * (P_new + P_new.T)
        self._last_pred_var = float(self.P[0, 0])
        return {
            "from_order": start_order,
            "trigger_z": float("nan"),
            "trigger_scale": float("nan"),
            "old_derivatives": old_x[start_order:].tolist(),
            "restored_sigmas": restored_sigmas,
            "applied_hints": applied_hints,
            "reason": str(reason),
        }

    def recover_level_jump(self, value: float, variance: float, *, t: float | None = None) -> None:
        """Recover from a confirmed discrete level change without relearning metrology.

        A regime change is not ordinary polynomial dynamics.  Snap the level mean to
        the confirmed observation, clear derivative means inherited from the previous
        regime, and preserve at least the existing derivative uncertainty so the new
        regime can immediately re-estimate rate/curvature/jerk from subsequent level
        observations.  Source calibration and learned process-noise parameters are
        intentionally untouched.
        """
        value = float(value)
        var = max(float(variance), NUMERIC_VARIANCE_FLOOR)
        old_P = np.array(self.P, dtype=float, copy=True)

        self.x[:] = 0.0
        self.x[0] = value

        P_new = np.zeros_like(old_P, dtype=float)
        P_new[0, 0] = max(var, float(old_P[0, 0]), NUMERIC_VARIANCE_FLOOR)
        T = max(float(self.prior_timescale_s), np.finfo(float).eps)
        for order in range(1, self.state_model.dim_x()):
            prior_var = var / (T ** (2 * order))
            P_new[order, order] = max(
                float(old_P[order, order]),
                prior_var,
                NUMERIC_VARIANCE_FLOOR,
            )

        self.P = P_new
        if t is not None:
            self.t_last = float(t)
        self._last_pred_var = float(P_new[0, 0])
        self.mode = "tracking"

    def dump_state(self) -> dict:
        return {
            "x": self.x.tolist(),
            "P": self.P.tolist(),
            "t_last": self.t_last,
            "_last_pred_var": self._last_pred_var,
            "q_process": self.q_process,
            "level_q_process": self.level_q_process,
            "prior_timescale_s": self.prior_timescale_s,
            "order": self.state_model.dim_x() - 1,
        }

    def load_state(self, data: dict) -> bool:
        try:
            x = np.array(data["x"], dtype=float)
            P = np.array(data["P"], dtype=float)
            if x.shape != self.x.shape or P.shape != self.P.shape:
                return False
            if not np.all(np.isfinite(x)) or not np.all(np.isfinite(P)):
                return False
            self.x = x
            self.P = 0.5 * (P + P.T)
            self.t_last = data.get("t_last")
            self._last_pred_var = data.get("_last_pred_var")
            q = data.get("q_process", data.get("q_jerk", data.get("q_acc")))
            if q is not None:
                self.q_process = q
            level_q = data.get("level_q_process")
            if level_q is not None:
                self.level_q_process = level_q
            T = data.get("prior_timescale_s", data.get("tau"))
            if T is not None:
                self.tau = T
            self.mode = "tracking"
            return True
        except Exception:
            return False

    def _prior_summary(self, obs: Observation) -> PosteriorSummary:
        var = max(float(obs.variance or 1.0), NUMERIC_VARIANCE_FLOOR)
        self.reset(obs.z, t=obs.t, variance=var)
        std = math.sqrt(var)
        return PosteriorSummary(
            x_mean=self.x.copy(), x_cov=self.P.copy(),
            y_mean=float(obs.z), y_var=var,
            innovation=0.0, innovation_var=var, loglik=0.0,
            ci68=(obs.z - std, obs.z + std),
            ci95=(obs.z - 1.96 * std, obs.z + 1.96 * std),
            probability_of_event=1.0, noise_velocity=0.0,
            dt=0.0, mode="prior",
            diag={"init": True, "weight": 1.0, "measurement_var": var,
                  "effective_innovation_var": var},
        )

    def step(self, obs: Observation) -> PosteriorSummary:
        if self.t_last is None:
            return self._prior_summary(obs)

        dt_raw = float(obs.t) - float(self.t_last)
        if dt_raw < -1e-6:
            raise ValueError("observations must be time ordered")
        dt = max(dt_raw, 1e-6)
        self.t_last = max(float(obs.t), float(self.t_last))

        Q = self.process_noise.Q(dt)
        x_pred, P_pred = self.state_model.predict(self.x, self.P, dt, Q)
        x_post, P_post, aux = self.updater.update(
            x_pred, P_pred, obs, dt, self.state_model
        )
        self.x, self.P = x_post, P_post
        self.mode = "tracking"

        y = float(self.state_model.measurement(self.x))
        pred_var = max(float(self.P[0, 0]), NUMERIC_VARIANCE_FLOOR)
        if self._last_pred_var is None:
            noise_velocity = 0.0
        else:
            noise_velocity = (pred_var - self._last_pred_var) / dt
        self._last_pred_var = pred_var

        innovation = float(aux.get("innovation", 0.0))
        innovation_var = max(float(aux.get("innovation_var", 0.0)), NUMERIC_VARIANCE_FLOOR)
        if hasattr(self.noise_model, "p_value"):
            p_event = self.noise_model.p_value(innovation, innovation_var)
        else:
            p_event = 1.0
        std = math.sqrt(pred_var)

        return PosteriorSummary(
            x_mean=self.x.copy(), x_cov=self.P.copy(),
            y_mean=y, y_var=pred_var,
            innovation=innovation, innovation_var=innovation_var,
            loglik=float(aux.get("loglik", 0.0)),
            ci68=(y - std, y + std),
            ci95=(y - 1.96 * std, y + 1.96 * std),
            probability_of_event=float(p_event),
            noise_velocity=float(noise_velocity), dt=dt,
            mode="tracking",
            diag={
                "weight": float(aux.get("weight", 1.0)),
                "measurement_var": float(aux.get("measurement_var", 0.0)),
                "effective_innovation_var": float(aux.get("effective_innovation_var", innovation_var)),
            },
        )
