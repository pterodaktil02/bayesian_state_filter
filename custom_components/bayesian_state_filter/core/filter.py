import math
import numpy as np

from ..const import NUMERIC_VARIANCE_FLOOR
from .types import Observation, PosteriorSummary


class CoreFilter:
    """Robust 2-state Bayesian filter with externally learned tau and q."""

    def __init__(self, state_model, noise_model, updater, process_noise):
        self.state_model = state_model
        self.noise_model = noise_model
        self.updater = updater
        self.process_noise = process_noise

        self.x = np.zeros(state_model.dim_x(), dtype=float)
        self.P = np.eye(state_model.dim_x(), dtype=float)
        self.t_last = None
        self.mode = "prior"
        self._last_pred_var = None

    @property
    def tau(self) -> float:
        return float(self.state_model.tau)

    @tau.setter
    def tau(self, value: float) -> None:
        v = max(float(value), 1e-6)
        self.state_model.tau = v
        self.process_noise.tau = v

    @property
    def q_acc(self) -> float:
        return float(self.process_noise.q_acc)

    @q_acc.setter
    def q_acc(self, value: float) -> None:
        self.process_noise.q_acc = max(float(value), 1e-18)

    def reset(self, value: float, *, t: float | None = None, variance: float | None = None) -> None:
        self.x[:] = [float(value), 0.0]
        v = max(float(variance) if variance is not None else 1.0, 1e-12)
        self.P = np.array([[v, 0.0], [0.0, v]], dtype=float)
        self.t_last = None if t is None else float(t)
        self._last_pred_var = None
        self.mode = "prior"

    def dump_state(self) -> dict:
        return {
            "x": self.x.tolist(),
            "P": self.P.tolist(),
            "t_last": self.t_last,
            "_last_pred_var": self._last_pred_var,
            "tau": self.tau,
            "q_acc": self.q_acc,
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
            if data.get("tau") is not None:
                self.tau = data["tau"]
            if data.get("q_acc") is not None:
                self.q_acc = data["q_acc"]
            self.mode = "tracking"
            return True
        except Exception:
            return False

    def _prior_summary(self, obs: Observation) -> PosteriorSummary:
        var = max(float(obs.variance or 1.0), 1e-12)
        self.reset(obs.z, t=obs.t, variance=var)
        # Initial slope uncertainty: one measurement unit per tau.
        self.P[1, 1] = max(var / (self.tau * self.tau), 1e-18)
        std = math.sqrt(var)
        return PosteriorSummary(
            x_mean=self.x.copy(), x_cov=self.P.copy(),
            y_mean=float(obs.z), y_var=var,
            innovation=0.0, innovation_var=var, loglik=0.0,
            ci68=(obs.z - std, obs.z + std),
            ci95=(obs.z - 1.96 * std, obs.z + 1.96 * std),
            probability_of_event=1.0, noise_velocity=0.0,
            dt=0.0, mode="prior", diag={"init": True, "weight": 1.0},
        )

    def step(self, obs: Observation) -> PosteriorSummary:
        if self.t_last is None:
            return self._prior_summary(obs)

        dt_raw = float(obs.t) - float(self.t_last)
        if dt_raw < -1e-6:
            raise ValueError("observations must be time ordered")
        dt = max(dt_raw, 1e-3)
        self.t_last = max(float(obs.t), float(self.t_last))

        Q = self.process_noise.Q(dt, self.x, tau=self.tau, q_acc=self.q_acc)
        x_pred, P_pred = self.state_model.predict(self.x, self.P, dt, Q, tau=self.tau)
        x_post, P_post, aux = self.updater.update(
            x_pred, P_pred, obs, dt, self.state_model, self.noise_model
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
