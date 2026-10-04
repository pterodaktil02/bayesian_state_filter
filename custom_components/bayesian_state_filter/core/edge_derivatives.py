from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np


@dataclass
class EdgeDerivativeEstimate:
    order: int
    points: int
    span_s: float
    rate: float | None
    rate_sigma: float | None
    curvature: float | None
    curvature_sigma: float | None
    jerk: float | None
    jerk_sigma: float | None
    residual_sigma: float
    iterations: int

    def dump(self) -> dict:
        return self.__dict__.copy()


@dataclass
class MultiScaleDerivativeConsensus:
    value: float
    sigma: float
    fit_sigma: float
    scale_sigma: float
    worst_disagreement_z: float
    estimates: int


def combine_multiscale_derivative(estimates, value_attr: str, sigma_attr: str):
    """Combine nested causal-window estimates without fake precision.

    The short/medium/long windows are strongly correlated, so ordinary
    inverse-variance combination would underestimate uncertainty.  We use the
    inverse-variance weighted mean only for the consensus *value*.  The
    within-fit uncertainty is conservatively floored at the best individual
    sigma, then an explicit between-scale term is added in quadrature.

    This makes a straight local trend precise when all scales agree, while a
    changing slope/curvature automatically widens the witness uncertainty
    instead of forcing one arbitrary bandwidth to be the truth.
    """
    valid = []
    for est in (estimates or {}).values():
        if est is None:
            continue
        value = getattr(est, value_attr, None)
        sigma = getattr(est, sigma_attr, None)
        try:
            value = float(value)
            sigma = float(sigma)
        except (TypeError, ValueError):
            continue
        if not (math.isfinite(value) and math.isfinite(sigma) and sigma > 0.0):
            continue
        valid.append((value, sigma))

    if not valid:
        return None

    weights = np.asarray([1.0 / (sig * sig) for _v, sig in valid], dtype=float)
    values = np.asarray([v for v, _sig in valid], dtype=float)
    wsum = float(np.sum(weights))
    if not math.isfinite(wsum) or wsum <= 0.0:
        return None

    center = float(np.sum(weights * values) / wsum)

    # The windows are nested and therefore correlated.  Do not claim the
    # 1/sqrt(sum(w)) precision of independent measurements; the consensus fit
    # sigma can be no smaller than the best constituent fit sigma.
    fit_sigma = min(sig for _v, sig in valid)
    scale_var = float(np.sum(weights * (values - center) ** 2) / wsum)
    scale_sigma = math.sqrt(max(scale_var, 0.0))
    total_sigma = math.sqrt(fit_sigma * fit_sigma + scale_sigma * scale_sigma)

    worst_z = 0.0
    if len(valid) >= 2:
        for i in range(len(valid)):
            for j in range(i + 1, len(valid)):
                v1, s1 = valid[i]
                v2, s2 = valid[j]
                denom = math.sqrt(s1 * s1 + s2 * s2)
                if denom > 0.0:
                    worst_z = max(worst_z, abs(v1 - v2) / denom)

    return MultiScaleDerivativeConsensus(
        value=center,
        sigma=total_sigma,
        fit_sigma=float(fit_sigma),
        scale_sigma=float(scale_sigma),
        worst_disagreement_z=float(worst_z),
        estimates=len(valid),
    )


def _median_dt(times: np.ndarray) -> float:
    if len(times) < 2:
        return 0.0
    d = np.diff(times)
    d = d[(d > 0.0) & np.isfinite(d)]
    return float(np.median(d)) if d.size else 0.0


def robust_causal_local_polynomial(
    points,
    *,
    max_points: int = 512,
    window_s: float | None = None,
    max_order: int = 3,
    min_points: int = 5,
    min_span_s: float | None = None,
    tukey_c: float = 4.685,
    max_iter: int = 6,
) -> EdgeDerivativeEstimate | None:
    """Estimate right-edge derivatives with robust variance-aware WLS.

    Points are ``(timestamp, value, variance)`` triples. The fit is causal:
    every sample lies at or before the evaluation time, which is the timestamp
    of the newest point.

    Measurement precision contributes ``1/variance``. A Tukey-biweight IRLS factor is
    then applied to standardized residuals so one bad endpoint sample cannot
    create an arbitrarily large derivative witness.
    """
    if points is None:
        return None

    min_points = max(int(min_points), 5)
    if min_span_s is not None:
        try:
            min_span_s = float(min_span_s)
        except (TypeError, ValueError):
            min_span_s = None
        if min_span_s is not None and (not math.isfinite(min_span_s) or min_span_s <= 0.0):
            min_span_s = None

    rows = []
    for p in list(points)[-max(int(max_points), 4):]:
        try:
            t, y, var = float(p[0]), float(p[1]), float(p[2])
        except (TypeError, ValueError, IndexError):
            continue
        if not (math.isfinite(t) and math.isfinite(y) and math.isfinite(var)):
            continue
        if var <= 0.0:
            continue
        rows.append((t, y, var))

    if len(rows) < min_points:
        return None

    rows.sort(key=lambda r: r[0])
    if window_s is not None:
        try:
            window_s = float(window_s)
        except (TypeError, ValueError):
            window_s = None
        if window_s is not None and math.isfinite(window_s) and window_s > 0.0:
            cutoff = float(rows[-1][0]) - window_s
            rows = [r for r in rows if float(r[0]) >= cutoff]

    dedup = []
    for row in rows:
        if dedup and abs(row[0] - dedup[-1][0]) <= 1e-9:
            dedup[-1] = row
        else:
            dedup.append(row)
    rows = dedup
    if len(rows) < min_points:
        return None

    t = np.asarray([r[0] for r in rows], dtype=float)
    y = np.asarray([r[1] for r in rows], dtype=float)
    var = np.asarray([r[2] for r in rows], dtype=float)

    span = float(t[-1] - t[0])
    dt_med = _median_dt(t)
    if not math.isfinite(span) or span <= 0.0 or dt_med <= 0.0:
        return None
    if span < 4.0 * dt_med:
        return None
    if min_span_s is not None and span < min_span_s:
        return None

    order = min(int(max_order), 3, len(rows) - 2)
    if order < 1:
        return None
    if order >= 3 and len(rows) < 9:
        order = 2

    t0 = float(t[-1])
    scale_t = max(span, dt_med, 1e-6)
    s = (t - t0) / scale_t

    cols = [np.ones_like(s), s]
    if order >= 2:
        cols.append(0.5 * s * s)
    if order >= 3:
        cols.append((s * s * s) / 6.0)
    A = np.column_stack(cols)

    meas_w = 1.0 / np.maximum(var, np.finfo(float).tiny)
    robust_w = np.ones_like(meas_w)

    # The newest sample has maximal leverage in a causal endpoint fit. Seed its
    # robust weight from a one-step extrapolation using only preceding points;
    # otherwise a bad endpoint can bend the polynomial toward itself and leave
    # an artificially small in-sample residual.
    if len(rows) >= order + 4:
        A0 = A[:-1]
        y0 = y[:-1]
        w0 = meas_w[:-1]
        try:
            sw0 = np.sqrt(w0)
            beta0, *_ = np.linalg.lstsq(A0 * sw0[:, None], y0 * sw0, rcond=None)
            resid0 = y0 - A0 @ beta0
            med0 = float(np.median(resid0))
            sigma0 = max(
                1.4826 * float(np.median(np.abs(resid0 - med0))),
                math.sqrt(float(np.median(var[:-1]))),
                1e-12,
            )
            endpoint_resid = abs(float(y[-1] - A[-1] @ beta0 - med0))
            endpoint_denom = math.sqrt(float(var[-1]) + sigma0 * sigma0)
            endpoint_u = endpoint_resid / max(endpoint_denom, 1e-12)
            if endpoint_u >= float(tukey_c):
                robust_w[-1] = 0.0
            else:
                r = endpoint_u / float(tukey_c)
                robust_w[-1] = (1.0 - r * r) ** 2
        except np.linalg.LinAlgError:
            pass

    beta = None
    iterations = 0
    residual_sigma = 0.0

    for it in range(max(int(max_iter), 1)):
        iterations = it + 1
        w = meas_w * robust_w
        sqrtw = np.sqrt(np.maximum(w, 0.0))
        Aw = A * sqrtw[:, None]
        yw = y * sqrtw
        try:
            beta_new, *_ = np.linalg.lstsq(Aw, yw, rcond=None)
        except np.linalg.LinAlgError:
            return None
        if beta_new.size != A.shape[1] or not np.all(np.isfinite(beta_new)):
            return None

        resid = y - A @ beta_new
        med = float(np.median(resid))
        mad_sigma = 1.4826 * float(np.median(np.abs(resid - med)))
        measurement_floor = math.sqrt(float(np.median(var)))
        residual_sigma = max(mad_sigma, measurement_floor, 1e-12)

        # Ordinary residuals hide high-leverage endpoint outliers because the
        # polynomial can bend toward them. Studentize by the weighted hat
        # leverage so a bad newest point is not allowed to explain itself.
        try:
            gram_inv = np.linalg.pinv(Aw.T @ Aw, rcond=1e-12)
            leverage = np.sum((Aw @ gram_inv) * Aw, axis=1)
            leverage = np.clip(leverage, 0.0, 0.98)
        except np.linalg.LinAlgError:
            leverage = np.zeros_like(resid)
        denom = np.sqrt(var + residual_sigma * residual_sigma) * np.sqrt(
            np.maximum(1.0 - leverage, 0.02)
        )
        u = np.abs(resid - med) / np.maximum(denom, 1e-12)
        r = u / max(float(tukey_c), 1e-12)
        new_robust = np.zeros_like(u)
        mask = r < 1.0
        new_robust[mask] = (1.0 - r[mask] * r[mask]) ** 2

        if beta is not None:
            delta = float(np.max(np.abs(beta_new - beta)))
            scale_beta = max(float(np.max(np.abs(beta_new))), 1.0)
            if delta <= 1e-9 * scale_beta:
                beta = beta_new
                robust_w = new_robust
                break
        beta = beta_new
        robust_w = new_robust

    if beta is None:
        return None

    w = meas_w * robust_w
    normal = A.T @ (w[:, None] * A)
    try:
        cov_scaled = np.linalg.pinv(normal, rcond=1e-12)
    except np.linalg.LinAlgError:
        cov_scaled = np.full((A.shape[1], A.shape[1]), np.nan)

    typical_var = max(float(np.median(var)), np.finfo(float).tiny)
    inflation = max(1.0, (residual_sigma * residual_sigma) / typical_var)
    cov_scaled = cov_scaled * inflation

    def deriv(k: int):
        if k > order:
            return None, None
        value = float(beta[k]) / (scale_t ** k)
        vv = float(cov_scaled[k, k]) / (scale_t ** (2 * k))
        sigma = math.sqrt(max(vv, 0.0)) if math.isfinite(vv) else None
        return value, sigma

    rate, rate_sigma = deriv(1)
    curvature, curvature_sigma = deriv(2)
    jerk, jerk_sigma = deriv(3)

    return EdgeDerivativeEstimate(
        order=order,
        points=len(rows),
        span_s=span,
        rate=rate,
        rate_sigma=rate_sigma,
        curvature=curvature,
        curvature_sigma=curvature_sigma,
        jerk=jerk,
        jerk_sigma=jerk_sigma,
        residual_sigma=float(residual_sigma),
        iterations=int(iterations),
    )
