from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

from .filter import CoreFilter
from .process_noise import IntegratedWienerProcessNoise
from .state_models import AdaptivePolynomialStateModel
from .types import Observation
from .updaters import StudentTUpdater
from ..const import (
    REGIME_CHANGE_Z_THRESHOLD, REGIME_CHANGE_CONFIRMATIONS,
    REGIME_CHANGE_COMPACT_SIGMA,
)

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
    # Robust local-polynomial derivative distribution learned from the same
    # history. Index 0 is level (unused by gating); 1/2/3 are v/a/j.
    derivative_centers: list[float] | None = None
    derivative_scales: list[float] | None = None
    derivative_samples: list[int] | None = None
    derivative_means: list[float] | None = None
    derivative_training_diagnostics: dict | None = None

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


def _make_filter(q: float, level_q: float, prior_timescale_s: float, nu: float, sigma: float,
                 derivative_profile=None) -> CoreFilter:
    model = AdaptivePolynomialStateModel(_FULL_ORDER)
    if derivative_profile is not None:
        centers, scales, samples = derivative_profile
        model.set_derivative_plausibility(scales=scales, centers=centers, samples=samples)
    return CoreFilter(
        state_model=model,
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


def _historical_regime_boundaries(points, *, z_threshold: float = REGIME_CHANGE_Z_THRESHOLD,
                                  confirmations: int = REGIME_CHANGE_CONFIRMATIONS,
                                  compact_sigma: float = REGIME_CHANGE_COMPACT_SIGMA):
    """Locate discrete level-regime boundaries in historical observations.

    This mirrors the live ``regime_change`` semantics as closely as possible
    without using the posterior filter state: a candidate must be a large
    (>= z_threshold) same-direction departure from the immediately preceding
    plateau, survive ``confirmations`` samples, and settle compactly around a
    new level.  The returned indices point at the *first* sample of the new
    plateau.  Local derivative windows crossing such an index must not be used
    to teach v/a/j plausibility: a level step is not a huge derivative.
    """
    n = len(points)
    if n < confirmations + 4:
        return set(), {"detected_level_jumps": 0}

    dt0 = max(_median_dt(points), 1e-9)
    # A short robust preceding plateau.  Long enough to suppress sample noise,
    # short enough not to smear normal slow dynamics into a false jump.
    pre_n = max(5, int(confirmations) + 2)
    boundaries = set()
    cooldown_until = -math.inf

    i = pre_n
    while i + confirmations <= n:
        t0 = float(points[i][0])
        if t0 < cooldown_until:
            i += 1
            continue

        pre = points[i - pre_n:i]
        new = points[i:i + confirmations]
        times = np.asarray([float(p[0]) for p in pre + new], dtype=float)
        if not np.all(np.isfinite(times)):
            i += 1
            continue
        gaps = np.diff(times)
        if gaps.size and (np.any(gaps <= 0.0) or np.max(gaps) > 4.0 * dt0):
            i += 1
            continue

        pre_vals = np.asarray([float(p[1]) for p in pre], dtype=float)
        new_vals = np.asarray([float(p[1]) for p in new], dtype=float)
        pre_sig = np.asarray([max(float(p[2]), 1e-12) for p in pre], dtype=float)
        new_sig = np.asarray([max(float(p[2]), 1e-12) for p in new], dtype=float)
        if not (np.all(np.isfinite(pre_vals)) and np.all(np.isfinite(new_vals))):
            i += 1
            continue

        old_level = float(np.median(pre_vals))
        target = float(np.median(new_vals))
        innovation = target - old_level
        if innovation == 0.0:
            i += 1
            continue
        sign = 1.0 if innovation > 0.0 else -1.0

        # Include both stated measurement sigma and the robust width of the old
        # plateau.  This avoids calling quantisation/noise a phase transition.
        pre_mad_sigma = 1.4826 * float(np.median(np.abs(pre_vals - old_level)))
        sigma_old = max(float(np.median(pre_sig)), pre_mad_sigma, 1e-12)
        sigma_new = max(float(np.median(new_sig)), 1e-12)
        innovation_sigma = math.sqrt(sigma_old * sigma_old + sigma_new * sigma_new)
        z = abs(innovation) / innovation_sigma
        if (not math.isfinite(z)) or z < float(z_threshold):
            i += 1
            continue

        # Same-direction confirmation: every point in the confirmation window
        # must remain clearly on the new side of the old plateau.
        deviations = new_vals - old_level
        same_direction = bool(np.all(sign * deviations > 0.0))
        if not same_direction:
            i += 1
            continue

        target_sigma = max(float(np.median(new_sig)), 1e-12)
        compact_limit = float(compact_sigma) * target_sigma
        compact = float(np.max(np.abs(new_vals - target))) <= compact_limit
        if not compact:
            i += 1
            continue

        boundaries.add(i)
        cooldown_until = float(new[-1][0]) + max(30.0, 4.0 * dt0)
        i += confirmations

    return boundaries, {"detected_level_jumps": int(len(boundaries))}


def _estimate_derivative_profile(points, *, timescale_s: float, max_samples: int = 8000,
                                 regime_z_threshold: float = REGIME_CHANGE_Z_THRESHOLD,
                                 regime_confirmations: int = REGIME_CHANGE_CONFIRMATIONS,
                                 regime_compact_sigma: float = REGIME_CHANGE_COMPACT_SIGMA):
    """Estimate characteristic v/a/j scales at the Bayesian dynamics timescale.

    The old implementation fitted a cubic through nine neighbouring samples.
    For slowly sampled or noisy sources that mostly measured amplified
    high-frequency measurement structure, not the dynamics that the Bayesian
    model actually transports.

    Here derivatives are finite differences on one common physical horizon
    ``T = timescale_s``:

      v = (x0 - x1) / T
      a = (x0 - 2*x1 + x2) / T**2
      j = (x0 - 3*x1 + 3*x2 - x3) / T**3

    where xk is the linearly interpolated level at t-k*T.  Samples never cross
    confirmed regime changes or large recorder gaps.  The production
    plausibility scale is zero-centred robust sigma
    ``1.4826 * median(abs(d))``.  Signed median and arithmetic mean are kept
    only as diagnostics.
    """
    n = len(points)
    centers = [0.0] * (_FULL_ORDER + 1)
    means = [0.0] * (_FULL_ORDER + 1)
    scales = [float("nan")] * (_FULL_ORDER + 1)
    samples = [0] * (_FULL_ORDER + 1)
    T = float(timescale_s)
    diagnostics = {
        "method": "finite_difference_at_gated_timescale",
        "timescale_s": T,
        "candidate_samples": 0,
        "accepted_samples": 0,
        "rejected_short_segment_samples": 0,
        "rejected_regime_crossing_samples": 0,
        "rejected_gap_samples": 0,
        "detected_level_jumps": 0,
        "median_abs": [None] * (_FULL_ORDER + 1),
    }
    if n < 4 or not math.isfinite(T) or T <= 0.0:
        return centers, scales, samples, means, diagnostics

    dt0 = max(_median_dt(points), 1e-9)
    boundaries, regime_diag = _historical_regime_boundaries(
        points,
        z_threshold=regime_z_threshold,
        confirmations=regime_confirmations,
        compact_sigma=regime_compact_sigma,
    )
    diagnostics.update(regime_diag)

    # Build independent continuous segments.  A confirmed regime boundary or a
    # large Recorder gap starts a new segment, so interpolation can never bridge
    # a step or missing-data interval.
    split_before = set(int(b) for b in boundaries if 0 < int(b) < n)
    for i in range(1, n):
        dt = float(points[i][0]) - float(points[i - 1][0])
        if not math.isfinite(dt) or dt <= 0.0 or dt > 4.0 * dt0:
            split_before.add(i)
            diagnostics["rejected_gap_samples"] += 1

    cuts = [0] + sorted(split_before) + [n]
    collected = [[] for _ in range(_FULL_ORDER + 1)]

    for a, b in zip(cuts[:-1], cuts[1:]):
        segment = points[a:b]
        if len(segment) < 2:
            continue
        tt = np.asarray([float(p[0]) for p in segment], dtype=float)
        zz = np.asarray([float(p[1]) for p in segment], dtype=float)
        good = np.isfinite(tt) & np.isfinite(zz)
        tt = tt[good]
        zz = zz[good]
        if tt.size < 2:
            continue

        # Need 3*T of clean history to estimate jerk.  Rate and acceleration
        # begin contributing as soon as their own horizons are available.
        idx = np.arange(tt.size, dtype=int)
        if idx.size > max_samples:
            pick = np.linspace(0, idx.size - 1, max_samples, dtype=int)
            idx = np.unique(pick)

        t0_seg = float(tt[0])
        for i in idx:
            t = float(tt[i])
            diagnostics["candidate_samples"] += 1
            x0 = float(zz[i])
            accepted_here = False

            # Linear interpolation is intentional: unlike a local polynomial it
            # does not manufacture higher-order structure between samples.
            if t - T >= t0_seg:
                x1 = float(np.interp(t - T, tt, zz))
                collected[1].append((x0 - x1) / T)
                accepted_here = True
            else:
                diagnostics["rejected_short_segment_samples"] += 1

            if t - 2.0 * T >= t0_seg:
                x1 = float(np.interp(t - T, tt, zz))
                x2 = float(np.interp(t - 2.0 * T, tt, zz))
                collected[2].append((x0 - 2.0 * x1 + x2) / (T * T))
                accepted_here = True

            if t - 3.0 * T >= t0_seg:
                x1 = float(np.interp(t - T, tt, zz))
                x2 = float(np.interp(t - 2.0 * T, tt, zz))
                x3 = float(np.interp(t - 3.0 * T, tt, zz))
                collected[3].append((x0 - 3.0 * x1 + 3.0 * x2 - x3) / (T ** 3))
                accepted_here = True

            if accepted_here:
                diagnostics["accepted_samples"] += 1

    for order in range(1, _FULL_ORDER + 1):
        arr = np.asarray(collected[order], dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            continue
        center = float(np.median(arr))
        mean = float(np.mean(arr))
        median_abs = float(np.median(np.abs(arr)))
        scale = 1.4826 * median_abs
        centers[order] = center
        means[order] = mean
        scales[order] = max(scale, 0.0)
        samples[order] = int(arr.size)
        diagnostics["median_abs"][order] = median_abs

    return centers, scales, samples, means, diagnostics

def _student_score(err: float, variance: float, nu: float) -> float:
    """Robust proper-ish predictive score; lower is better."""
    v = max(float(variance), 1e-18)
    e2 = float(err) * float(err)
    nu = max(float(nu), 1.0)
    return math.log(v) + (nu + 1.0) * math.log1p(e2 / (nu * v))


def _evaluate(points, q: float, level_q: float, prior_T: float, nu: float,
              *, max_forecasts: int = 1000, derivative_profile=None):
    if len(points) < 8:
        return math.inf, {}, -math.inf, math.inf
    sigmas = [max(float(p[2]), 1e-12) for p in points]
    sigma0 = float(np.median(sigmas)) if sigmas else 1.0
    f = _make_filter(q, level_q, prior_T, nu, sigma0, derivative_profile=derivative_profile)
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


def fit_process_noise(train_points, nu: float, derivative_profile=None):
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
            score, _errors, _ll, _se = _evaluate(
                points, q, level_q, prior_T, nu, derivative_profile=derivative_profile
            )
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
    _s, errors, _ll, _se = _evaluate(
        points, q, level_q, prior_T, nu, max_forecasts=1200,
        derivative_profile=derivative_profile,
    )
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

    # First identify a coarse dynamics timescale without a derivative prior.
    # Then learn derivative plausibility on that same physical horizon and
    # refit Q once.  Finally recompute the published derivative profile at the
    # final fitted timescale so diagnostics and reset variances describe exactly
    # the scale transported by the Bayesian model.
    q0, level_q0, T0, _score0, _boundary0, _rounds0 = fit_process_noise(
        train, nu, derivative_profile=None
    )
    derivative_profile_seed_full = _estimate_derivative_profile(points, timescale_s=T0)
    derivative_profile_seed = derivative_profile_seed_full[:3]
    q, level_q, Tdiag, train_score, boundary, rounds = fit_process_noise(
        train, nu, derivative_profile=derivative_profile_seed
    )
    derivative_profile_full = _estimate_derivative_profile(points, timescale_s=Tdiag)
    derivative_profile = derivative_profile_full[:3]
    dt = max(_median_dt(train), 1e-6)
    prior_T = max(16.0 * dt, 60.0)
    val_points = _representative_points(validation, max_points=6000, blocks=6)
    _score, errors, ll_mean, ll_se = _evaluate(
        val_points, q, level_q, prior_T, nu, max_forecasts=1200,
        derivative_profile=derivative_profile,
    )
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
        derivative_centers=list(derivative_profile[0]),
        derivative_scales=list(derivative_profile[1]),
        derivative_samples=list(derivative_profile_full[2]),
        derivative_means=list(derivative_profile_full[3]),
        derivative_training_diagnostics=dict(derivative_profile_full[4]),
    )
