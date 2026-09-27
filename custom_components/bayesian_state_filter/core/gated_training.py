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
_Q00_DENOM = (2 * _FULL_ORDER + 1) * (math.factorial(_FULL_ORDER) ** 2)  # 252
_HORIZON_STEPS = (1, 2, 4, 8, 16, 32)


@dataclass
class GatedDynamicsEstimate:
    q_process: float
    level_q_process: float
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


def level_q_from_timescale(measurement_variance: float, timescale_s: float) -> float:
    """Legacy helper retained for checkpoint/backward compatibility only.

    Production training no longer derives level diffusion from one fitted time
    constant.  ``level_q_process`` is identified directly from multi-horizon
    predictive residuals.  For dimensional correctness a random-walk diffusion
    that accumulates one measurement variance over T seconds is R/T.
    """
    T = float(timescale_s)
    if not math.isfinite(T) or T <= 0:
        return 0.0
    R = max(float(measurement_variance), np.finfo(float).tiny)
    return R / T


def q_from_timescale(order: int, measurement_variance: float, timescale_s: float) -> float:
    """Dimensional helper used only to construct a broad search grid."""
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


def _make_filter(q: float, level_q: float, prior_timescale_s: float, nu: float, sigma: float) -> CoreFilter:
    return CoreFilter(
        state_model=AdaptivePolynomialStateModel(_FULL_ORDER),
        noise_model=None,
        updater=StudentTUpdater(nu=nu, min_weight=0.05),
        process_noise=IntegratedWienerProcessNoise(_FULL_ORDER, q, level_q=level_q),
        prior_timescale_s=max(float(prior_timescale_s), 1e-6),
    )


def _rmse(values) -> float:
    if not values:
        return math.inf
    a = np.asarray(values, dtype=float)
    return float(math.sqrt(np.mean(a * a)))


def _median_dt(points) -> float:
    if len(points) < 2:
        return 60.0
    times = np.asarray([p[0] for p in points], dtype=float)
    dts = np.diff(times)
    dts = dts[(dts > 0) & np.isfinite(dts)]
    return float(np.median(dts)) if dts.size else 60.0


def _representative_points(points, max_points: int = 6000, blocks: int = 6):
    """Keep native cadence in several windows spread across the history.

    Uniform decimation of a week-long 1 Hz series destroys the very fast
    dynamics we want to identify.  Instead retain contiguous native-cadence
    blocks distributed over the horizon.  Large gaps are treated as reset
    boundaries by the scorer below.
    """
    n = len(points)
    if n <= max_points:
        return list(points)
    blocks = max(1, min(int(blocks), max_points // 100))
    block_len = max(100, max_points // blocks)
    block_len = min(block_len, n)
    starts = np.linspace(0, max(n - block_len, 0), blocks, dtype=int)
    out = []
    last_end = -1
    for st in starts:
        st = int(st)
        en = min(st + block_len, n)
        if st < last_end:
            st = last_end
        if st >= en:
            continue
        out.extend(points[st:en])
        last_end = en
    return out


def _student_score(err: float, variance: float, nu: float) -> float:
    """Robust proper-ish predictive score; lower is better."""
    v = max(float(variance), 1e-18)
    e2 = float(err) * float(err)
    nu = max(float(nu), 1.0)
    return math.log(v) + (nu + 1.0) * math.log1p(e2 / (nu * v))


def _evaluate(points, q: float, level_q: float, prior_T: float, nu: float,
              *, max_forecasts: int = 1000):
    if len(points) < 8:
        return math.inf, {}, -math.inf, math.inf
    sigmas = [max(float(p[2]), 1e-12) for p in points]
    sigma0 = float(np.median(sigmas)) if sigmas else 1.0
    f = _make_filter(q, level_q, prior_T, nu, sigma0)
    dt0 = max(_median_dt(points), 1e-6)
    gap_reset = 8.0 * dt0
    warmup = min(max(16, int(len(points) * 0.03)), max(len(points) - 3, 0))
    stride = max(1, int(math.ceil(max(len(points) - warmup, 1) / max_forecasts)))
    errors = {h: [] for h in _HORIZON_STEPS}
    score_sum = {h: 0.0 for h in _HORIZON_STEPS}
    score_n = {h: 0 for h in _HORIZON_STEPS}
    lls = []
    prev_t = None

    for i, (t, value, sigma) in enumerate(points):
        t = float(t)
        value = float(value)
        sigma = max(float(sigma), 1e-12)
        if prev_t is not None and t - prev_t > gap_reset:
            f.reset(value, t=t, variance=sigma * sigma, timescale_s=prior_T)
            prev_t = t
            continue
        out = f.step(Observation(t=t, z=value, variance=sigma * sigma, source="gated_training"))
        prev_t = t
        if i >= warmup and out.dt > 0:
            lls.append(float(out.loglik))
        if i < warmup or ((i - warmup) % stride):
            continue

        for h in _HORIZON_STEPS:
            j = i + h
            if j >= len(points):
                continue
            future_t, future_value, future_sigma = points[j]
            dt = float(future_t) - t
            if dt <= 0 or dt > gap_reset:
                continue
            Q = f.process_noise.Q(dt)
            x_pred, P_pred = f.state_model.predict(f.x, f.P, dt, Q)
            err = float(future_value) - float(x_pred[0])
            pred_var = max(float(P_pred[0, 0]) + float(future_sigma) ** 2, 1e-18)
            errors[h].append(err)
            score_sum[h] += _student_score(err, pred_var, nu)
            score_n[h] += 1

    horizon_scores = [score_sum[h] / score_n[h] for h in _HORIZON_STEPS if score_n[h] > 0]
    aggregate = float(np.mean(horizon_scores)) if horizon_scores else math.inf
    if lls:
        arr = np.asarray(lls, dtype=float)
        ll_mean = float(np.mean(arr))
        ll_se = float(np.std(arr, ddof=1) / math.sqrt(len(arr))) if len(arr) > 1 else math.inf
    else:
        ll_mean, ll_se = -math.inf, math.inf
    return aggregate, errors, ll_mean, ll_se


def _candidate_scales(R: float, dt: float, span: float):
    # Search by physically meaningful *horizons*, but fit q and level_q as two
    # independent parameters.  No single fitted T is allowed to dictate Q.
    max_h = max(dt * 256.0, 60.0)
    if span > 0:
        max_h = min(max_h, max(span / 8.0, dt * 8.0))
    horizons = np.geomspace(max(dt, 1e-3), max(max_h, dt * 8.0), 5)
    level = [0.0] + sorted({float(R / h) for h in horizons})
    snap = [0.0] + sorted({float(q_from_timescale(_FULL_ORDER, R, h)) for h in horizons})
    return level, snap, float(max_h)


def _crossover_timescale(q: float, level_q: float, R: float, dt: float, max_h: float) -> tuple[float, bool]:
    """Horizon where accumulated process variance reaches measurement R.

    This is a derived diagnostic only; it does not determine Q.
    """
    R = max(float(R), 1e-18)
    lo = max(float(dt), 1e-6)
    hi = max(float(max_h), lo)

    def proc_var(h):
        return max(float(level_q), 0.0) * h + max(float(q), 0.0) * (h ** 7) / _Q00_DENOM

    if proc_var(lo) >= R:
        return lo, True
    if proc_var(hi) < R:
        return hi, True
    for _ in range(64):
        mid = math.sqrt(lo * hi)
        if proc_var(mid) >= R:
            hi = mid
        else:
            lo = mid
    return hi, False


def fit_process_noise(train_points, nu: float):
    if len(train_points) < 8:
        sigma = float(train_points[0][2]) if train_points else 1.0
        R = sigma * sigma
        dt = 60.0
        return 0.0, R / 600.0, 600.0, math.inf, True, 1

    points = _representative_points(train_points)
    dt = max(_median_dt(points), 1e-6)
    span = max(float(train_points[-1][0] - train_points[0][0]), dt)
    sigmas = np.asarray([p[2] for p in points], dtype=float)
    R = max(float(np.median(sigmas * sigmas)), np.finfo(float).tiny)
    prior_T = max(16.0 * dt, 60.0)
    level_candidates, snap_candidates, max_h = _candidate_scales(R, dt, span)

    best = None
    best_idx = None
    for li, level_q in enumerate(level_candidates):
        for qi, q in enumerate(snap_candidates):
            score, _errors, _ll, _se = _evaluate(points, q, level_q, prior_T, nu)
            cand = (score, float(q), float(level_q))
            if best is None or cand[0] < best[0]:
                best = cand
                best_idx = (li, qi)

    if best is None:
        return 0.0, 0.0, prior_T, math.inf, True, 1
    score, q, level_q = best
    li, qi = best_idx
    boundary = li in (0, len(level_candidates) - 1) or qi in (0, len(snap_candidates) - 1)
    Tdiag, t_boundary = _crossover_timescale(q, level_q, R, dt, max_h)
    # Keep the historical diagnostic name ``train_rmse`` meaningful even
    # though candidate selection itself uses the multi-horizon predictive
    # score above.
    _s, errors, _ll, _se = _evaluate(points, q, level_q, prior_T, nu, max_forecasts=1200)
    rmses = [_rmse(errors.get(h, [])) for h in _HORIZON_STEPS]
    finite = [x for x in rmses if math.isfinite(x)]
    train_rmse = float(math.sqrt(np.mean(np.square(finite)))) if finite else math.inf
    return q, level_q, Tdiag, train_rmse, bool(boundary or t_boundary), 1


def train_gated_dynamics(fused_points, nu: float = 4.0, train_fraction: float = 0.80) -> GatedDynamicsEstimate:
    """Fit [x,v,a,j] process noise directly from multi-horizon predictions.

    ``q_process`` (snap-driven covariance) and ``level_q_process`` (orthogonal
    level random walk) are independent fitted parameters.  Prediction quality
    is scored at 1,2,4,...64 natural grid steps with a robust predictive score.
    The reported ``timescale_s`` is derived *after* fitting as the horizon where
    accumulated process variance reaches the typical measurement variance; it
    never controls Q.
    """
    points = [(float(t), float(z), math.sqrt(max(float(var), 1e-18))) for t, z, var in fused_points]
    if len(points) < 40:
        raise ValueError("not enough history for gated dynamics training")
    split = max(20, min(len(points) - 10, int(len(points) * train_fraction)))
    train = points[:split]
    validation = points[split:]

    q, level_q, Tdiag, train_score, boundary, rounds = fit_process_noise(train, nu)
    dt = max(_median_dt(train), 1e-6)
    prior_T = max(16.0 * dt, 60.0)
    val_points = _representative_points(validation, max_points=6000, blocks=6)
    _score, errors, ll_mean, ll_se = _evaluate(val_points, q, level_q, prior_T, nu, max_forecasts=1200)
    rmse1 = _rmse(errors.get(1, []))
    rmse2 = _rmse(errors.get(2, []))
    per_h_rmse = [_rmse(errors.get(h, [])) for h in _HORIZON_STEPS]
    finite = [x for x in per_h_rmse if math.isfinite(x)]
    val_rmse = float(math.sqrt(np.mean(np.square(finite)))) if finite else math.inf

    span = float(points[-1][0] - points[0][0])
    grid_dt = max(_median_dt(points), 0.0)
    return GatedDynamicsEstimate(
        q_process=float(q),
        level_q_process=float(level_q),
        timescale_s=float(Tdiag),
        train_rmse=float(train_score),
        validation_rmse=val_rmse,
        validation_rmse_step1=rmse1,
        validation_rmse_step2=rmse2,
        validation_loglik_mean=float(ll_mean),
        validation_loglik_se=float(ll_se),
        history_span_s=span,
        grid_dt_s=grid_dt,
        train_points=len(train),
        validation_points=len(validation),
        search_boundary_limited=bool(boundary),
        search_rounds=int(rounds),
    )
