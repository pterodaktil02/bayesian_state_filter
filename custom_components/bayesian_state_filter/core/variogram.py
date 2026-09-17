"""Robust characteristic-time estimation from the level process.

The estimator deliberately operates on the *level* time series, not on the
velocity state used by the live filter.  For a stationary first-order process
with measurement noise the temporal semivariogram is

    gamma(h) = nugget + process_variance * (1 - exp(-h / tau))

where ``nugget`` is the measurement-noise contribution, ``process_variance``
is the variance of the latent process, and ``tau`` is the characteristic time
of the level itself.

This separation is important: the predictive damped-velocity model has its own
``velocity_tau`` which answers a different question (how long the current
slope remains useful for prediction).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import math
import numpy as np


@dataclass
class CharacteristicTimeEstimate:
    tau: float | None
    p10: float | None
    p90: float | None
    confidence: float
    identifiable: bool
    status: str
    nugget_variance: float
    process_variance: float
    signal_fraction: float
    fit_error: float
    lag_count: int
    pair_count: int
    edge_mass: float
    boundary_limited: bool
    lower_limited: bool = False
    upper_limited: bool = False
    min_lag: float = 0.0
    max_lag: float = 0.0

    def dump(self) -> dict:
        return asdict(self)

    @classmethod
    def load(cls, data):
        allowed = {k: data[k] for k in cls.__dataclass_fields__ if k in data}
        return cls(**allowed)


def _weighted_quantile(values, weights, q: float) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if values.size == 0:
        return float("nan")
    order = np.argsort(values)
    v = values[order]
    w = np.maximum(weights[order], 0.0)
    c = np.cumsum(w)
    if c[-1] <= 0:
        return float(np.median(v))
    c /= c[-1]
    return float(np.interp(float(q), c, v))


def _robust_semivariance(diff: np.ndarray) -> float:
    """Gaussian-consistent semivariance from median absolute increments.

    If D ~ N(0, sigma_D^2), median(|D|)=0.67449*sigma_D.  Therefore
    0.5*(1.4826*median(|D|))^2 estimates 0.5*Var(D), i.e. semivariance.
    Keeping the increment centred on zero intentionally counts persistent
    drift as process motion rather than subtracting it away.
    """
    if diff.size == 0:
        return 0.0
    med_abs = float(np.median(np.abs(diff)))
    return 0.5 * (1.4826 * med_abs) ** 2


def _fit_nonnegative_line(f: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    """Weighted NNLS for y = nugget + sill*f with only two coefficients."""
    one = np.ones_like(f)
    s00 = float(np.sum(w * one))
    s01 = float(np.sum(w * f))
    s11 = float(np.sum(w * f * f))
    b0 = float(np.sum(w * y))
    b1 = float(np.sum(w * f * y))
    det = s00 * s11 - s01 * s01
    candidates: list[tuple[float, float]] = []

    if det > 1e-30:
        nugget = (b0 * s11 - b1 * s01) / det
        sill = (s00 * b1 - s01 * b0) / det
        if nugget >= 0.0 and sill >= 0.0:
            candidates.append((nugget, sill))

    # Boundary nugget=0.
    if s11 > 1e-30:
        candidates.append((0.0, max(0.0, b1 / s11)))
    # Boundary sill=0.
    if s00 > 1e-30:
        candidates.append((max(0.0, b0 / s00), 0.0))
    if not candidates:
        return 0.0, 0.0

    def loss(par):
        n, s = par
        r = y - (n + s * f)
        return float(np.sum(w * r * r))

    return min(candidates, key=loss)


def _lag_rows(points: list[tuple[float, float, float]], *, lag_points: int = 26,
              max_pairs_per_lag: int = 4096) -> list[tuple[float, float, float, float, int]]:
    """Return (lag, gamma, gamma_se, measurement_floor, pairs) rows.

    Points are normally produced on a regular fusion grid, but nearest-time
    matching keeps this tolerant of occasional holes in that grid.
    """
    if len(points) < 40:
        return []
    arr = np.asarray(points, dtype=float)
    times, values, variances = arr[:, 0], arr[:, 1], np.maximum(arr[:, 2], 1e-18)
    dts = np.diff(times)
    positive = dts[dts > 0]
    if positive.size == 0:
        return []
    step = float(np.median(positive))
    span = float(times[-1] - times[0])
    if span <= 4.0 * step:
        return []

    max_lag = max(2.0 * step, span / 3.0)
    raw_lags = np.geomspace(step, max_lag, max(int(lag_points), 10))
    # Quantise requested lags to the underlying time grid so the shortest
    # geometric bins do not collapse to duplicate 1-step comparisons.
    lag_steps = np.unique(np.maximum(1, np.rint(raw_lags / step).astype(int)))
    requested = lag_steps.astype(float) * step

    rows = []
    n = len(times)
    base_i = np.arange(n)
    for h in requested:
        targets = times + h
        j_hi = np.searchsorted(times, targets, side="left")
        j_hi = np.clip(j_hi, 1, n - 1)
        j_lo = j_hi - 1
        err_hi = np.abs(times[j_hi] - targets)
        err_lo = np.abs(times[j_lo] - targets)
        j = np.where(err_lo <= err_hi, j_lo, j_hi)

        actual = times[j] - times
        tolerance = max(1.5 * step, 0.12 * h)
        valid = (j > base_i) & (np.abs(actual - h) <= tolerance)
        ii = base_i[valid]
        jj = j[valid]
        if ii.size < 30:
            continue

        # Deterministic thinning bounds startup CPU while preserving coverage
        # across the whole history instead of taking only the newest pairs.
        if ii.size > max_pairs_per_lag:
            sel = np.linspace(0, ii.size - 1, max_pairs_per_lag).astype(int)
            ii, jj = ii[sel], jj[sel]

        diff = values[jj] - values[ii]
        # Chunk-by-time estimates provide an empirical uncertainty that also
        # penalises non-stationarity; a 7-day trend should not yield fake
        # precision just because there are many samples.
        chunks_n = min(10, max(3, diff.size // 150))
        chunk_indices = np.array_split(np.arange(diff.size), chunks_n)
        chunk_gamma = []
        for ch in chunk_indices:
            if ch.size >= 10:
                chunk_gamma.append(_robust_semivariance(diff[ch]))
        if not chunk_gamma:
            continue
        cg = np.asarray(chunk_gamma, dtype=float)
        gamma = float(np.median(cg))
        if cg.size >= 3:
            mad = 1.4826 * float(np.median(np.abs(cg - gamma)))
            gamma_se = mad / math.sqrt(float(cg.size))
        else:
            gamma_se = 0.10 * max(gamma, 1e-15)
        # Do not let huge pair counts claim absurd precision for a model that
        # is only an approximation to real environmental dynamics.
        gamma_se = max(gamma_se, 0.07 * max(gamma, 1e-15), 1e-15)

        measurement_floor = float(np.median(0.5 * (variances[ii] + variances[jj])))
        realised_lag = float(np.median(times[jj] - times[ii]))
        rows.append((realised_lag, gamma, gamma_se, measurement_floor, int(ii.size)))

    # Merge any accidental duplicate realised lags after nearest-time matching.
    merged = []
    for row in rows:
        if merged and abs(row[0] - merged[-1][0]) <= max(step * 0.05, 1e-6):
            # Keep the row with more pairs (normally identical).
            if row[4] > merged[-1][4]:
                merged[-1] = row
        else:
            merged.append(row)
    return merged


def estimate_characteristic_time(points: list[tuple[float, float, float]], *,
                                 tau_points: int = 48,
                                 tau_min_s: float | None = None,
                                 tau_max_s: float | None = None) -> CharacteristicTimeEstimate | None:
    """Estimate the characteristic time of the latent *level* process.

    A profile fit over tau is used; nugget and process variance are solved by
    non-negative weighted least squares for each tau.  Confidence is reduced
    when the process signal is weak, the posterior is boundary-censored, or
    the history does not extend far enough beyond the fitted knee.
    """
    rows = _lag_rows(points)
    if len(rows) < 6:
        return None

    arr = np.asarray(rows, dtype=float)
    h, gamma, se, noise_floor, pair_counts = arr.T
    hmin, hmax = float(np.min(h)), float(np.max(h))
    # Search deliberately extends below the first measured lag and beyond the
    # largest lag.  If evidence accumulates there the result is marked as a
    # bound rather than pretending that the boundary value was measured.
    tmin = max(float(tau_min_s), 1e-3) if tau_min_s is not None else max(hmin / 4.0, 1e-3)
    tmax = max(float(tau_max_s), tmin * 1.01) if tau_max_s is not None else max(hmax * 3.0, tmin * 100.0)
    grid = np.geomspace(tmin, tmax, max(int(tau_points), 16))

    base_w = 1.0 / np.maximum(se, 1e-15) ** 2
    losses = np.zeros(len(grid), dtype=float)
    params = np.zeros((len(grid), 2), dtype=float)
    nu = 4.0
    for i, tau in enumerate(grid):
        f = 1.0 - np.exp(-h / tau)
        nugget, sill = _fit_nonnegative_line(f, gamma, base_w)
        params[i] = (nugget, sill)
        r = (gamma - (nugget + sill * f)) / np.maximum(se, 1e-15)
        # Student-t loss over lag bins makes one pathological lag harmless.
        losses[i] = float(np.sum((nu + 1.0) * np.log1p((r * r) / nu)))

    logw = -0.5 * (losses - float(np.min(losses)))
    weights = np.exp(np.clip(logw, -745.0, 0.0))
    sw = float(np.sum(weights))
    if sw <= 0 or not math.isfinite(sw):
        weights = np.ones_like(weights) / len(weights)
    else:
        weights /= sw

    p10 = _weighted_quantile(grid, weights, 0.10)
    p50 = _weighted_quantile(grid, weights, 0.50)
    p90 = _weighted_quantile(grid, weights, 0.90)
    best_idx = int(np.argmax(weights))
    # Use the profile parameters nearest the posterior median for diagnostics.
    med_idx = int(np.argmin(np.abs(np.log(grid) - math.log(max(p50, 1e-12)))))
    nugget, sill = map(float, params[med_idx])

    edge_mass = float(weights[0] + weights[-1])
    lower_limited = bool(weights[0] >= 0.20 or p10 <= grid[1])
    upper_limited = bool(weights[-1] >= 0.20 or p90 >= grid[-2])
    boundary_limited = lower_limited or upper_limited or edge_mass >= 0.25

    full_log_span = max(math.log(grid[-1] / grid[0]), 1e-12)
    post_width = max(math.log(max(p90, p10 * 1.000001) / max(p10, 1e-12)), 0.0)
    width_conf = max(0.0, min(1.0, 1.0 - post_width / full_log_span))
    entropy = -float(np.sum(weights * np.log(np.maximum(weights, 1e-300))))
    entropy_conf = 1.0 - entropy / max(math.log(len(weights)), 1e-12)
    posterior_conf = max(0.0, min(1.0, 0.55 * width_conf + 0.45 * entropy_conf))

    total_var = max(nugget + sill, 1e-18)
    signal_fraction = max(0.0, min(1.0, sill / total_var))
    # Weak latent motion cannot identify a time constant, even with millions of
    # noisy samples.  Ramp confidence smoothly rather than imposing a hard
    # process/noise ratio threshold.
    signal_conf = max(0.0, min(1.0, (signal_fraction - 0.02) / 0.28))

    # A characteristic time is only well observed if the lag range extends
    # past its knee.  About 3*tau reaches 95% of the exponential plateau.
    coverage_conf = max(0.0, min(1.0, hmax / max(3.0 * p50, 1e-12)))

    fmed = 1.0 - np.exp(-h / max(p50, 1e-12))
    pred = nugget + sill * fmed
    norm_resid = np.abs(gamma - pred) / np.maximum(se, 1e-15)
    fit_error = float(np.median(norm_resid))
    fit_conf = 1.0 / (1.0 + max(fit_error - 1.0, 0.0))

    boundary_conf = max(0.0, 1.0 - min(edge_mass, 1.0))
    # Geometric-like combination: every term represents a necessary condition,
    # but square roots avoid making confidence needlessly tiny for one merely
    # mediocre component.
    confidence = posterior_conf
    confidence *= math.sqrt(max(signal_conf, 0.0))
    confidence *= math.sqrt(max(coverage_conf, 0.0))
    confidence *= math.sqrt(max(fit_conf, 0.0))
    confidence *= boundary_conf
    confidence = max(0.0, min(1.0, confidence))

    # If the fitted process amplitude is negligible, tau is fundamentally
    # unidentifiable.  Returning None is more honest than exposing an arbitrary
    # profile minimum for a flat/noise-only signal.
    insufficient_signal = signal_fraction < 0.05
    identifiable = (
        not insufficient_signal
        and not boundary_limited
        and len(rows) >= 8
        and confidence >= 0.35
    )

    if insufficient_signal:
        status = "insufficient_signal"
        tau_out = p10_out = p90_out = None
        confidence = min(confidence, 0.10)
    elif lower_limited:
        status = "below_resolution"
        tau_out, p10_out, p90_out = float(p50), float(p10), float(p90)
    elif upper_limited:
        status = "longer_than_history"
        tau_out, p10_out, p90_out = float(p50), float(p10), float(p90)
    elif identifiable:
        status = "identified"
        tau_out, p10_out, p90_out = float(p50), float(p10), float(p90)
    else:
        status = "uncertain"
        tau_out, p10_out, p90_out = float(p50), float(p10), float(p90)

    return CharacteristicTimeEstimate(
        tau=tau_out,
        p10=p10_out,
        p90=p90_out,
        confidence=float(confidence),
        identifiable=bool(identifiable),
        status=status,
        nugget_variance=max(float(nugget), 0.0),
        process_variance=max(float(sill), 0.0),
        signal_fraction=float(signal_fraction),
        fit_error=float(fit_error),
        lag_count=int(len(rows)),
        pair_count=int(sum(int(r[4]) for r in rows)),
        edge_mass=float(edge_mass),
        boundary_limited=bool(boundary_limited),
        lower_limited=bool(lower_limited),
        upper_limited=bool(upper_limited),
        min_lag=float(hmin),
        max_lag=float(hmax),
    )
