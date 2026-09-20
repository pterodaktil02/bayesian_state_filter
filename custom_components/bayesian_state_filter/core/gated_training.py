from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

from .filter import CoreFilter
from .process_noise import IntegratedWienerProcessNoise
from .state_models import AdaptivePolynomialStateModel
from .types import Observation
from .updaters import StudentTUpdater

_FULL_ORDER = 3


@dataclass
class GatedDynamicsEstimate:
    q_process: float
    timescale_s: float
    train_rmse: float
    validation_rmse: float
    validation_rmse_step1: float
    validation_rmse_step2: float
    validation_loglik_mean: float
    validation_loglik_se: float
    history_span_s: float
    grid_dt_s: float
    train_points: int
    validation_points: int
    search_boundary_limited: bool
    search_rounds: int

    def dump(self) -> dict:
        return self.__dict__.copy()

    @classmethod
    def load(cls, data: dict | None):
        if not data:
            return None
        return cls(**{k: data[k] for k in cls.__dataclass_fields__ if k in data})


def _make_filter(q: float, timescale_s: float, nu: float, sigma: float) -> CoreFilter:
    return CoreFilter(
        state_model=AdaptivePolynomialStateModel(_FULL_ORDER),
        noise_model=None,
        updater=StudentTUpdater(nu=nu, min_weight=0.05),
        process_noise=IntegratedWienerProcessNoise(_FULL_ORDER, q),
        prior_timescale_s=max(float(timescale_s), np.finfo(float).eps),
    )


def q_from_timescale(order: int, measurement_variance: float, timescale_s: float) -> float:
    T = float(timescale_s)
    if not math.isfinite(T) or T <= 0:
        return 0.0
    n = int(order)
    coeff_denom = (2 * n + 1) * (math.factorial(n) ** 2)
    logq = (
        math.log(max(float(measurement_variance), np.finfo(float).tiny))
        + math.log(coeff_denom)
        - (2 * n + 1) * math.log(T)
    )
    if logq < math.log(np.finfo(float).tiny):
        return 0.0
    if logq > math.log(np.finfo(float).max):
        return np.finfo(float).max
    return math.exp(logq)


def _rmse(values) -> float:
    if not values:
        return math.inf
    a = np.asarray(values, dtype=float)
    return float(math.sqrt(np.mean(a * a)))


def _run_local(points, q: float, T: float, nu: float, *, warmup_fraction: float = 0.05):
    sigmas = [max(float(p[2]), 1e-12) for p in points]
    sigma0 = float(np.median(sigmas)) if sigmas else 1.0
    f = _make_filter(q, T, nu, sigma0)
    errors = {1: [], 2: []}
    lls = []
    warmup = min(max(10, int(len(points) * warmup_fraction)), max(len(points) - 3, 0))

    for i, (t, value, sigma) in enumerate(points):
        out = f.step(Observation(t=float(t), z=float(value), variance=float(sigma) ** 2, source="gated_training"))
        if i >= warmup and out.dt > 0:
            lls.append(float(out.loglik))
        if i < warmup:
            continue
        for horizon in (1, 2):
            j = i + horizon
            if j >= len(points):
                continue
            future_t, future_value, _future_sigma = points[j]
            dt = float(future_t) - float(t)
            if dt <= 0:
                continue
            Q = f.process_noise.Q(dt)
            x_pred, _P_pred = f.state_model.predict(f.x, f.P, dt, Q)
            errors[horizon].append(float(future_value) - float(x_pred[0]))

    rmse1 = _rmse(errors[1])
    rmse2 = _rmse(errors[2])
    finite = [x for x in (rmse1, rmse2) if math.isfinite(x)]
    aggregate = float(math.sqrt(np.mean(np.square(finite)))) if finite else math.inf
    if lls:
        arr = np.asarray(lls, dtype=float)
        ll_mean = float(np.mean(arr))
        ll_se = float(np.std(arr, ddof=1) / math.sqrt(len(arr))) if len(arr) > 1 else math.inf
    else:
        ll_mean, ll_se = -math.inf, math.inf
    return f, aggregate, rmse1, rmse2, ll_mean, ll_se


def fit_q(train_points, nu: float):
    if len(train_points) < 8:
        sigma = float(train_points[0][2]) if train_points else 1.0
        T = 60.0
        q = q_from_timescale(_FULL_ORDER, sigma * sigma, T)
        return q, T, math.inf, True, 0

    times = np.asarray([p[0] for p in train_points], dtype=float)
    sigmas = np.asarray([p[2] for p in train_points], dtype=float)
    dts = np.diff(times)
    dts = dts[dts > 0]
    dt = float(np.median(dts)) if dts.size else 60.0
    span = max(float(times[-1] - times[0]), dt)
    R = max(float(np.median(sigmas * sigmas)), np.finfo(float).tiny)

    lo = max(dt * 0.5, np.finfo(float).eps)
    hi = max(span, dt * 32.0)
    best = None
    boundary_limited = False
    rounds = 0

    for rounds in range(1, 9):
        grid = np.geomspace(lo, hi, 9)
        round_best = None
        round_idx = None
        for idx, T in enumerate(grid):
            q = q_from_timescale(_FULL_ORDER, R, float(T))
            _f, score, _r1, _r2, _ll, _se = _run_local(train_points, q, float(T), nu)
            cand = (score, float(T), float(q))
            if round_best is None or cand[0] < round_best[0]:
                round_best = cand
                round_idx = idx
            if best is None or cand[0] < best[0]:
                best = cand

        _f, score, _r1, _r2, _ll, _se = _run_local(train_points, 0.0, float(hi), nu)
        cand = (score, math.inf, 0.0)
        if best is None or cand[0] < best[0]:
            best = cand

        if best is not None and best[2] == 0.0:
            boundary_limited = False
            break
        idx = int(round_idx if round_idx is not None else len(grid) // 2)
        at_low = idx <= 1
        at_high = idx >= len(grid) - 2
        if not at_low and not at_high:
            boundary_limited = False
            break
        boundary_limited = True
        if at_low:
            new_lo = lo / 100.0
            if new_lo <= np.finfo(float).eps:
                break
            lo = new_lo
        if at_high:
            new_hi = hi * 100.0
            if not math.isfinite(new_hi):
                break
            hi = new_hi

    if best is None:
        T = span
        q = q_from_timescale(_FULL_ORDER, R, T)
        return q, T, math.inf, True, rounds
    score, T, q = best
    return float(q), float(T), float(score), bool(boundary_limited), rounds


def train_gated_dynamics(fused_points, nu: float = 4.0, train_fraction: float = 0.80) -> GatedDynamicsEstimate:
    """Train q/timescale for the permanent confidence-gated [x,v,a,j] model.

    ``fused_points`` are (t, level, variance).  Prediction quality is scored only
    one and two natural grid steps ahead.  Derivative confidence itself is not
    trained: c(z)=erf(|z|/sqrt(2)).
    """
    points = [(float(t), float(z), math.sqrt(max(float(var), 1e-18))) for t, z, var in fused_points]
    if len(points) < 40:
        raise ValueError("not enough history for gated dynamics training")
    split = max(20, min(len(points) - 10, int(len(points) * train_fraction)))
    train = points[:split]
    validation = points[split:]

    q, T, train_rmse, boundary, rounds = fit_q(train, nu)
    effective_T = (train[-1][0] - train[0][0]) if math.isinf(T) else T
    sigmas = [p[2] for p in train]
    f = _make_filter(q, max(effective_T, 1e-6), nu, float(np.median(sigmas)))
    for t, value, sigma in train:
        f.step(Observation(t=t, z=value, variance=sigma * sigma, source="gated_training"))

    errors = {1: [], 2: []}
    lls = []
    for i, (t, value, sigma) in enumerate(validation):
        out = f.step(Observation(t=t, z=value, variance=sigma * sigma, source="gated_training"))
        if out.dt > 0:
            lls.append(float(out.loglik))
        for horizon in (1, 2):
            j = i + horizon
            if j >= len(validation):
                continue
            future_t, future_value, _ = validation[j]
            dt = future_t - t
            if dt <= 0:
                continue
            Q = f.process_noise.Q(dt)
            x_pred, _ = f.state_model.predict(f.x, f.P, dt, Q)
            errors[horizon].append(future_value - float(x_pred[0]))

    rmse1 = _rmse(errors[1])
    rmse2 = _rmse(errors[2])
    finite = [x for x in (rmse1, rmse2) if math.isfinite(x)]
    val_rmse = float(math.sqrt(np.mean(np.square(finite)))) if finite else math.inf
    if lls:
        arr = np.asarray(lls, dtype=float)
        ll_mean = float(np.mean(arr))
        ll_se = float(np.std(arr, ddof=1) / math.sqrt(len(arr))) if len(arr) > 1 else math.inf
    else:
        ll_mean, ll_se = -math.inf, math.inf

    span = float(points[-1][0] - points[0][0])
    dts = np.diff(np.asarray([p[0] for p in points], dtype=float))
    dts = dts[dts > 0]
    grid_dt = float(np.median(dts)) if dts.size else 0.0
    return GatedDynamicsEstimate(
        q_process=q,
        timescale_s=float(T),
        train_rmse=train_rmse,
        validation_rmse=val_rmse,
        validation_rmse_step1=rmse1,
        validation_rmse_step2=rmse2,
        validation_loglik_mean=ll_mean,
        validation_loglik_se=ll_se,
        history_span_s=span,
        grid_dt_s=grid_dt,
        train_points=len(train),
        validation_points=len(validation),
        search_boundary_limited=bool(boundary),
        search_rounds=int(rounds),
    )
