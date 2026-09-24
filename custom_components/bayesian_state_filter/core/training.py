"""Robust multi-source history calibration and dynamics pre-training."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from bisect import bisect_right
from collections import deque
import math
import statistics
import time
import numpy as np

from .dynamics import DynamicsBank, DynamicsEstimate
from .variogram import CharacteristicTimeEstimate, estimate_characteristic_time


def _median(values, default=0.0):
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(statistics.median(vals)) if vals else float(default)


def _mad(values, center=None):
    vals = [float(v) for v in values if math.isfinite(float(v))]
    if not vals:
        return 0.0
    c = _median(vals) if center is None else float(center)
    return 1.4826 * _median([abs(v - c) for v in vals])


def _quantile(values, q):
    vals = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not vals:
        return 0.0
    if len(vals) == 1:
        return vals[0]
    p = max(0.0, min(1.0, float(q))) * (len(vals) - 1)
    lo, hi = int(math.floor(p)), int(math.ceil(p))
    if lo == hi:
        return vals[lo]
    f = p - lo
    return vals[lo] * (1.0 - f) + vals[hi] * f


@dataclass
class SourceCalibration:
    bias: float = 0.0
    sigma: float = 0.1
    median_dt: float = 60.0
    typical_abs_level: float = 1.0
    samples: int = 0
    calibration_samples: int = 0
    calibration_span: float = 0.0
    calibration_pairs: int = 0
    # Snapshot of the Recorder/history evidence used at startup.  The legacy
    # calibration_* fields above remain the current working evidence and may
    # later be replaced by online pairwise calibration statistics.
    startup_calibration_samples: int = 0
    startup_calibration_span: float = 0.0
    startup_calibration_pairs: int = 0
    startup_sigma: float = 0.0
    calibration_window_s: float = 0.0
    outliers: int = 0
    updates: int = 0

    @property
    def outlier_rate(self):
        return float(self.outliers) / max(int(self.updates), 1)

    def variance(self, value: float, noise_mode: str = "gaussian") -> float:
        s2 = max(self.sigma * self.sigma, 1e-12)
        if noise_mode == "poisson":
            k = s2 / max(self.typical_abs_level, 1e-9)
            return max(k * max(abs(float(value)), 1e-9), 1e-12)
        return s2

    def dump(self):
        return asdict(self)

    @classmethod
    def load(cls, data):
        allowed = {k: data[k] for k in cls.__dataclass_fields__ if k in data}
        return cls(**allowed)


@dataclass
class TrainingResult:
    sources: dict[str, SourceCalibration]
    # Predictive damped-velocity model.  Its tau is *not* the physical/level
    # characteristic time; it is the memory of the local slope.
    dynamics: DynamicsEstimate | None
    bank: DynamicsBank | None
    # Characteristic time of the level process inferred from a robust temporal
    # variogram.
    characteristic: CharacteristicTimeEstimate | None
    fused_points: list[tuple[float, float, float]]
    grid_step: float
    history_span: float
    # Compact pairwise evidence used to make startup -> online calibration a
    # continuous rolling-window transition without retaining millions of raw
    # pair residuals in Home Assistant memory.
    startup_pair_rows: list[tuple[str, str, float, int, float]] | None = None
    calibration_window_s: float = 0.0
    # Dedicated local-dynamics grid.  This mirrors the proven trend-filter
    # bootstrap: common overlap, linear interpolation, natural cadence and only
    # a generic 150k computational ceiling.  Kept separate from the
    # hold-last-value grid used by source calibration/characteristic-time work.
    dynamics_points: list[tuple[float, float, float]] | None = None


@dataclass
class BiasAnchorResult:
    """Diagnostics for the gauge chosen for relative source biases."""
    mode: str
    shift: float
    model_centers: dict[str, float] | None = None
    model_weights: dict[str, float] | None = None


def _huber_psi(u: float, delta: float = 1.345) -> float:
    u = float(u)
    d = max(float(delta), 1e-9)
    return max(-d, min(d, u))


def _passport_anchor_center(calibrations, source_models, model_accuracy, *, huber_delta=1.345):
    """Return a robust absolute-bias gauge from device-model accuracy priors.

    Each device model contributes one family-level center, regardless of how
    many physical sensors of that model are present.  This prevents ten
    identical sensors from becoming ten independent votes about absolute
    accuracy when their datasheet systematic error is shared.

    The family centers are combined with a heteroscedastic Huber M-estimator:

        sum_f psi((m_f - g) / a_f) / a_f = 0

    where m_f is the robust median bias of the model family and a_f is the
    configured datasheet absolute-accuracy scale.
    """
    source_models = source_models or {}
    model_accuracy = model_accuracy or {}
    grouped = {}
    for src, cal in calibrations.items():
        model = source_models.get(src)
        if not model or model not in model_accuracy:
            continue
        try:
            b = float(cal.bias)
            a = float(model_accuracy[model])
        except (TypeError, ValueError):
            continue
        if math.isfinite(b) and math.isfinite(a) and a > 0:
            grouped.setdefault(str(model), []).append(b)
    if not grouped:
        return None, {}, {}

    centers = {model: _median(vals) for model, vals in grouped.items()}
    scales = {model: float(model_accuracy[model]) for model in centers}
    if len(centers) == 1:
        model = next(iter(centers))
        return centers[model], centers, {model: 1.0 / (scales[model] ** 2)}

    # The score is monotone decreasing in g, so bisection gives the unique
    # minimizer of the convex Huber objective without an arbitrary optimizer.
    max_scale = max(scales.values())
    lo = min(centers.values()) - 20.0 * max_scale
    hi = max(centers.values()) + 20.0 * max_scale

    def score(g):
        total = 0.0
        for model, m in centers.items():
            a = scales[model]
            total += _huber_psi((m - g) / a, huber_delta) / a
        return total

    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if score(mid) > 0:
            lo = mid
        else:
            hi = mid
    g = 0.5 * (lo + hi)

    weights = {}
    for model, m in centers.items():
        a = scales[model]
        u = (m - g) / a
        robust = 1.0 if abs(u) <= huber_delta else huber_delta / max(abs(u), 1e-12)
        weights[model] = robust / (a * a)
    return float(g), centers, weights


def normalize_bias_gauge(calibrations, *, mode="median", source_models=None,
                         model_accuracy=None, residuals=None, huber_delta=1.345):
    """Remove the unidentifiable common bias mode.

    ``median`` keeps v0.3.2 behavior. ``mean`` is a linear sum-to-zero gauge.
    ``passport`` uses one robust family vote per configured device model,
    scaled by that model's datasheet absolute accuracy.
    """
    mode = str(mode or "median").strip().lower()
    finite = [float(c.bias) for c in calibrations.values()
              if math.isfinite(float(c.bias))]
    if not finite:
        return BiasAnchorResult(mode=mode, shift=0.0)

    model_centers = None
    model_weights = None
    if mode == "mean":
        gauge = float(sum(finite) / len(finite))
    elif mode == "passport":
        gauge, model_centers, model_weights = _passport_anchor_center(
            calibrations, source_models, model_accuracy, huber_delta=huber_delta
        )
        if gauge is None:
            raise ValueError("passport bias anchor has no sources with valid model absolute_accuracy")
    else:
        mode = "median"
        gauge = float(_median(finite, 0.0))

    if not math.isfinite(gauge) or abs(gauge) <= 1e-15:
        return BiasAnchorResult(mode=mode, shift=0.0,
                                model_centers=model_centers, model_weights=model_weights)

    for cal in calibrations.values():
        if math.isfinite(float(cal.bias)):
            cal.bias = float(cal.bias) - gauge

    # Live residuals target source bias in the current gauge.  When corrected
    # values move by +g, raw-minus-reference residuals move by -g.
    if residuals is not None:
        for src, dq in list(residuals.items()):
            if dq:
                residuals[src] = deque((float(t), float(v) - gauge) for t, v in dq)

    return BiasAnchorResult(mode=mode, shift=float(gauge),
                            model_centers=model_centers, model_weights=model_weights)


def _clean_history(seq):
    out = []
    for t, z in seq:
        try:
            t, z = float(t), float(z)
        except (TypeError, ValueError):
            continue
        if math.isfinite(t) and math.isfinite(z):
            out.append((t, z))
    out.sort(key=lambda p: p[0])
    # Deduplicate exact timestamps: last value wins.
    dedup = []
    for p in out:
        if dedup and abs(p[0] - dedup[-1][0]) < 1e-9:
            dedup[-1] = p
        else:
            dedup.append(p)
    return dedup


def _median_dt(seq):
    dts = [seq[i][0] - seq[i - 1][0] for i in range(1, len(seq)) if seq[i][0] > seq[i - 1][0]]
    return _median(dts, 60.0)


def _temporal_sigma(seq):
    if len(seq) < 3:
        return 0.0
    diffs = [seq[i][1] - seq[i - 1][1] for i in range(1, len(seq))]
    # Conservative fallback only.  First differences contain real process
    # motion and therefore tend to overestimate measurement noise.
    return _mad(diffs) / math.sqrt(2.0)


def _pair_match_tolerance(dt_a: float, dt_b: float, tau: float | None) -> float:
    """Maximum time skew for a pairwise calibration observation.

    Pairwise noise calibration only works if both sensors observed essentially
    the same latent process state.  Use the faster source cadence to set the
    normal matching tolerance and cap it to a small fraction of the observed
    process time scale.
    """
    fast_dt = max(min(float(dt_a), float(dt_b)), 1.0)
    tol = max(2.0 * fast_dt, 5.0)
    if tau is not None:
        try:
            tau_f = float(tau)
            if math.isfinite(tau_f) and tau_f > 0:
                tol = min(tol, max(0.02 * tau_f, 15.0))
        except (TypeError, ValueError):
            pass
    return max(float(tol), 2.0)


def _nearest_pair_residuals(seq_a, seq_b, *, bias_a=0.0, bias_b=0.0,
                            recent_window_s=None, tau=None):
    """Time-align two asynchronous sources with nearest-neighbour matching.

    The *slower* source is used as the anchor.  Each anchor sample is matched
    to the nearest sample of the faster source only when the time skew is tiny
    compared with the process time scale.  Unlike hold-last-value fusion this
    does not charge genuine process motion to the faster source.
    """
    a = _clean_history(seq_a)
    b = _clean_history(seq_b)
    if len(a) < 2 or len(b) < 2:
        return [], 0.0

    dt_a = max(_median_dt(a), 1.0)
    dt_b = max(_median_dt(b), 1.0)
    # Anchor on the slower stream so one slow observation cannot be counted
    # dozens of times merely because its peer updates quickly.
    if dt_a >= dt_b:
        anchors, peers = a, b
        anchor_bias, peer_bias = float(bias_a), float(bias_b)
    else:
        anchors, peers = b, a
        anchor_bias, peer_bias = float(bias_b), float(bias_a)

    end = min(anchors[-1][0], peers[-1][0])
    if recent_window_s is not None and recent_window_s > 0:
        cutoff = end - float(recent_window_s)
        anchors = [p for p in anchors if cutoff <= p[0] <= end]
        peers = [p for p in peers if p[0] >= cutoff - max(dt_a, dt_b) and p[0] <= end + max(dt_a, dt_b)]
    if len(anchors) < 2 or len(peers) < 2:
        return [], 0.0

    peer_times = [p[0] for p in peers]
    tol = _pair_match_tolerance(dt_a, dt_b, tau)
    residuals = []
    used_times = []
    for t, z in anchors:
        idx = bisect_right(peer_times, t)
        candidates = []
        if idx < len(peers):
            candidates.append(peers[idx])
        if idx > 0:
            candidates.append(peers[idx - 1])
        if not candidates:
            continue
        tp, zp = min(candidates, key=lambda q: abs(q[0] - t))
        if abs(tp - t) > tol:
            continue
        r = (float(z) - anchor_bias) - (float(zp) - peer_bias)
        if math.isfinite(r):
            residuals.append(float(r))
            used_times.append(float(t))

    span = used_times[-1] - used_times[0] if len(used_times) >= 2 else 0.0
    return residuals, max(float(span), 0.0)


def _pairwise_variance_rows(histories, calib, *, recent_window_s, tau=None, min_samples=8):
    """Build robust pairwise equations Var(i-j) ~= sigma_i^2 + sigma_j^2."""
    sources = sorted(src for src in calib if histories.get(src))
    rows = []
    source_counts = {src: 0 for src in sources}
    source_span = {src: 0.0 for src in sources}
    for i, src_a in enumerate(sources):
        for src_b in sources[i + 1:]:
            residuals, span = _nearest_pair_residuals(
                histories[src_a], histories[src_b],
                bias_a=calib[src_a].bias, bias_b=calib[src_b].bias,
                recent_window_s=recent_window_s, tau=tau,
            )
            if len(residuals) < int(min_samples):
                continue
            # Pair offsets and slowly varying spatial gradients are location
            # effects, not observation noise.  MAD around the pair median
            # removes the constant offset robustly.
            pair_sigma = _mad(residuals)
            if not math.isfinite(pair_sigma) or pair_sigma <= 0:
                continue
            n = len(residuals)
            rows.append((src_a, src_b, pair_sigma * pair_sigma, n, span))
            source_counts[src_a] += n
            source_counts[src_b] += n
            source_span[src_a] = max(source_span[src_a], span)
            source_span[src_b] = max(source_span[src_b], span)
    return rows, source_counts, source_span


def _solve_source_variances(calib, rows):
    """Solve non-negative source variances from robust pair variances.

    A weak ridge to the provisional variances makes the problem well-defined
    for only two sources or a star-shaped pair graph, while informative
    pairwise equations dominate as soon as the network is identifiable.
    """
    sources = sorted(calib)
    n = len(sources)
    if n == 0 or not rows:
        return {}
    index = {src: i for i, src in enumerate(sources)}
    A = []
    b = []
    w = []
    for sa, sb, pair_var, count, _span in rows:
        row = np.zeros(n, dtype=float)
        row[index[sa]] = 1.0
        row[index[sb]] = 1.0
        A.append(row)
        b.append(max(float(pair_var), 1e-16))
        # Robust scales from hundreds of serially correlated observations are
        # not hundreds of times more informative than sparse pairs.  Cap the
        # effective count so fast sources cannot dominate the network.
        w.append(float(min(max(int(count), 1), 64)))

    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    w = np.asarray(w, dtype=float)
    prior = np.asarray([max(calib[s].sigma ** 2, 1e-12) for s in sources], dtype=float)

    # A variance component much smaller than the statistical resolution of the
    # pair variances is not actually identified.  Without this guard an
    # inconsistent/noisy pair network can pin one source to zero and make the
    # Bayesian filter absurdly overconfident.  The floor is half the median
    # first-order variance uncertainty of pairs touching that source.  Count is
    # capped because pair residuals are serially correlated in real sensors.
    resolution = {src: [] for src in sources}
    for sa, sb, pair_var, count, _span in rows:
        n_eff = min(max(int(count), 2), 64)
        se_var = float(pair_var) * math.sqrt(2.0 / max(n_eff - 1, 1))
        resolution[sa].append(se_var)
        resolution[sb].append(se_var)
    lower = np.asarray([
        max(0.5 * _median(resolution[s], 0.0), 1e-12) for s in sources
    ], dtype=float)

    # Weak prior only resolves null-space directions.
    ridge_weight = max(float(np.median(w)) * 0.02, 0.05)
    Aw = A * np.sqrt(w)[:, None]
    bw = b * np.sqrt(w)
    Ar = np.eye(n, dtype=float) * math.sqrt(ridge_weight)
    br = prior * math.sqrt(ridge_weight)
    M = np.vstack([Aw, Ar])
    y = np.concatenate([bw, br])

    x = np.maximum(prior.copy(), lower)
    try:
        spectral = float(np.linalg.norm(M, ord=2))
    except Exception:
        spectral = float(np.linalg.norm(M))
    lipschitz = max(2.0 * spectral * spectral, 1e-12)
    for _ in range(600):
        grad = 2.0 * M.T.dot(M.dot(x) - y)
        x_new = np.maximum(x - grad / lipschitz, lower)
        if np.max(np.abs(x_new - x)) <= 1e-10 * max(1.0, float(np.max(x))):
            x = x_new
            break
        x = x_new
    return {src: float(x[index[src]]) for src in sources}

def _make_grid(histories, step, freshness):
    starts = [s[0][0] for s in histories.values() if s]
    ends = [s[-1][0] for s in histories.values() if s]
    if not starts or not ends:
        return []
    start, end = min(starts), max(ends)
    if end <= start:
        return []

    ts_by_src = {k: [p[0] for p in seq] for k, seq in histories.items()}
    grid = []
    t = math.ceil(start / step) * step
    # Cap at 150k points: preserve local dynamics while bounding one-time bootstrap cost.
    n_est = int((end - t) / step) + 1
    if n_est > 150000:
        step *= math.ceil(n_est / 150000)
        t = math.ceil(start / step) * step

    while t <= end + 1e-9:
        snap = {}
        for src, seq in histories.items():
            if not seq:
                continue
            times = ts_by_src[src]
            idx = bisect_right(times, t) - 1
            if idx >= 0 and t - seq[idx][0] <= freshness[src]:
                snap[src] = seq[idx][1]
        if snap:
            grid.append((t, snap))
        t += step
    return grid


def _build_fused(grid, calib):
    """Fuse a temporal grid using fixed per-source bias/sigma calibrations."""
    fused = []
    for t, snap in grid:
        vals = [(src, z - calib[src].bias) for src, z in snap.items() if src in calib]
        if not vals:
            continue
        zs = [z for _, z in vals]
        zmed = _median(zs)
        # Robust local source rejection.  For <=2 sources there is no majority,
        # so retain both and rely on their calibrated measurement variances.
        if len(zs) >= 3:
            sc = _mad(zs, center=zmed)
            floor = _median([calib[src].sigma for src, _ in vals], 1e-6)
            sc = max(sc, floor, 1e-9)
            kept = [(src, z) for src, z in vals if abs(z - zmed) <= 4.685 * sc]
            if kept:
                vals = kept
        zf = _median([z for _, z in vals])
        source_vars = [calib[src].sigma ** 2 for src, _ in vals]
        if len(source_vars) == 1:
            vf = max(source_vars[0], 1e-12)
        else:
            vf = max(1.57 * _median(source_vars, 1e-4) / len(vals),
                     0.25 * _median(source_vars, 1e-4), 1e-12)
        fused.append((float(t), float(zf), float(vf)))
    return fused



def _build_dynamics_fused(histories, calib, max_points=150000):
    """Build the prototype-equivalent fused history for x-v-a-j training.

    Returns (t, level, variance).  Source bias/sigma come from the production
    calibrator, but temporal fusion is deliberately identical in spirit to
    bayesian_trend_filter v0.6.3: interpolate every calibrated source on the
    common overlap at its natural ensemble cadence, with only a generic point
    ceiling.
    """
    usable = {src: _clean_history(seq) for src, seq in histories.items() if src in calib and seq}
    usable = {src: seq for src, seq in usable.items() if seq}
    if not usable:
        return []
    start = max(seq[0][0] for seq in usable.values())
    end = min(seq[-1][0] for seq in usable.values())
    if end <= start:
        return []
    span = end - start
    native_dt = _median([max(float(calib[src].median_dt), 1e-6) for src in usable], 60.0)
    step = max(native_dt, span / max(int(max_points) - 1, 1), 1e-6)
    grids = np.arange(start, end + 0.5 * step, step, dtype=float)
    corrected = []
    variances = []
    for src, seq in usable.items():
        tt = np.asarray([x[0] for x in seq], dtype=float)
        xx = np.asarray([x[1] for x in seq], dtype=float) - float(calib[src].bias)
        corrected.append(np.interp(grids, tt, xx))
        variances.append(max(float(calib[src].sigma) ** 2, 1e-12))
    if not corrected:
        return []
    matrix = np.vstack(corrected)
    fused = np.median(matrix, axis=0)
    med_var = _median(variances, 1e-4)
    fused_var = med_var if len(corrected) == 1 else max(0.25 * med_var, med_var / len(corrected), 1e-12)
    return [(float(t), float(v), float(fused_var)) for t, v in zip(grids, fused)]

def _calibration_window_s(characteristic, *, dts, span):
    """Choose the recent source-calibration window used at startup.

    This mirrors OnlineSourceCalibrator: source quality is a local property,
    while relative bias and process dynamics may use the full recorder history.
    """
    median_dt = _median(dts.values(), 60.0)
    tau = None
    if characteristic is not None and characteristic.tau is not None:
        try:
            candidate = float(characteristic.tau)
            if math.isfinite(candidate) and candidate > 0:
                tau = candidate
        except (TypeError, ValueError):
            pass
    window = max(3.0 * tau, 10.0 * median_dt, 3600.0) if tau is not None else max(6.0 * 3600.0, 10.0 * median_dt)
    if span > 0:
        window = min(window, span)
    return max(float(window), 60.0)


def _recalibrate_recent_sigmas(histories, calib, *, recent_window_s, tau=None):
    """Estimate per-source observation sigma from time-aligned sensor pairs.

    Pairwise differences cancel the common latent process without relying on
    high-order temporal differences of a single source.  Therefore serially
    correlated / internally filtered sensors do not collapse to a fictitious
    near-zero sigma.
    """
    rows, source_counts, source_span = _pairwise_variance_rows(
        histories, calib, recent_window_s=recent_window_s, tau=tau, min_samples=8
    )
    solved = _solve_source_variances(calib, rows)
    pair_counts = {src: 0 for src in calib}
    for sa, sb, _var, _count, _span in rows:
        pair_counts[sa] = pair_counts.get(sa, 0) + 1
        pair_counts[sb] = pair_counts.get(sb, 0) + 1
    for src, c in calib.items():
        if src in solved:
            floor = max(c.typical_abs_level * 1e-6, 1e-8)
            c.sigma = max(math.sqrt(max(solved[src], 0.0)), floor)
        c.calibration_samples = int(source_counts.get(src, 0))
        c.calibration_span = float(source_span.get(src, 0.0))
        c.calibration_pairs = int(pair_counts.get(src, 0))
        c.startup_calibration_samples = c.calibration_samples
        c.startup_calibration_span = c.calibration_span
        c.startup_calibration_pairs = c.calibration_pairs
        c.startup_sigma = float(c.sigma)
    return rows

def calibrate_history(histories: dict[str, list[tuple[float, float]]], *,
                      tau_points: int = 16, forget_time_s: float | None = None,
                      tau_min_s: float | None = None, tau_max_s: float | None = None,
                      characteristic_tau_min_s: float | None = None,
                      characteristic_tau_max_s: float | None = None,
                      bias_anchor: str = "median", source_models=None,
                      model_accuracy=None, huber_delta: float = 1.345) -> TrainingResult:
    histories = {k: _clean_history(v) for k, v in histories.items()}
    histories = {k: v for k, v in histories.items() if v}
    if not histories:
        return TrainingResult({}, None, None, None, [], 60.0, 0.0)

    dts = {k: max(_median_dt(v), 1.0) for k, v in histories.items()}
    step = max(1.0, min(300.0, _median(dts.values(), 60.0)))
    freshness = {k: max(3.0 * dts[k], 3.0 * step) for k in histories}
    grid = _make_grid(histories, step, freshness)
    # _make_grid may enlarge its cadence to cap startup work at 20k points.
    # Propagate that realised cadence to all downstream estimators and to the
    # live rolling-history sampler.
    if len(grid) >= 2:
        realised_dts = [grid[i][0] - grid[i - 1][0] for i in range(1, len(grid)) if grid[i][0] > grid[i - 1][0]]
        if realised_dts:
            step = max(_median(realised_dts, step), 1.0)

    # Pass 1: spatial reference without assuming source quality.
    residuals = {k: [] for k in histories}
    for _, snap in grid:
        if len(snap) < 2:
            continue
        ref = _median(snap.values())
        for src, z in snap.items():
            residuals[src].append(z - ref)

    calib = {}
    for src, seq in histories.items():
        bias = _median(residuals[src], 0.0) if residuals[src] else 0.0
        # Provisional sigma is deliberately conservative and is used only for
        # the preliminary fusion/time-scale fit.  The published per-source
        # sigma is replaced below by time-aligned pairwise calibration.
        sigma = _mad(residuals[src], center=bias) if residuals[src] else _temporal_sigma(seq)
        if sigma <= 0:
            sigma = _temporal_sigma(seq)
        level_scale = _median([abs(z) for _, z in seq], 1.0)
        numerical_floor = max(level_scale * 1e-6, 1e-8)
        sigma = max(sigma, numerical_floor)
        calib[src] = SourceCalibration(
            bias=float(bias), sigma=float(sigma), median_dt=float(dts[src]),
            typical_abs_level=max(float(level_scale), 1e-9), samples=len(seq),
        )

    normalize_bias_gauge(
        calib, mode=bias_anchor, source_models=source_models,
        model_accuracy=model_accuracy, huber_delta=huber_delta,
    )

    # Pass 2a: provisional fusion using the long-history calibration.  The
    # long history is excellent for relative bias, but its spatial residual
    # spread can be badly inflated by old operating regimes, spatial thermal
    # gradients, SysID runs, maintenance incidents, etc.  We therefore use it
    # only to obtain a preliminary level-process time scale.
    fused = _build_fused(grid, calib)

    if len(fused) < 20:
        span = fused[-1][0] - fused[0][0] if len(fused) >= 2 else 0.0
        return TrainingResult(calib, None, None, None, fused, step, span)

    span = fused[-1][0] - fused[0][0]

    # Estimate a preliminary time scale, then recalibrate source *noise* from
    # a recent window comparable to the online calibrator (about 3 tau).
    # Bias remains a long-history quantity; sigma is intentionally local so a
    # restart does not resurrect obsolete week-old operating regimes.
    preliminary_characteristic = estimate_characteristic_time(
        fused,
        tau_points=max(int(tau_points) * 3, 32),
        tau_min_s=characteristic_tau_min_s,
        tau_max_s=characteristic_tau_max_s,
    )
    recent_window = _calibration_window_s(
        preliminary_characteristic, dts=dts, span=span
    )
    startup_pair_rows = _recalibrate_recent_sigmas(
        histories, calib, recent_window_s=recent_window,
        tau=(preliminary_characteristic.tau if preliminary_characteristic is not None else None),
    )
    for c in calib.values():
        c.calibration_window_s = float(recent_window)

    # Rebuild observation variances with the recent source-noise calibration,
    # then fit the published characteristic time and predictive dynamics.
    fused = _build_fused(grid, calib)
    characteristic = estimate_characteristic_time(
        fused,
        tau_points=max(int(tau_points) * 3, 32),
        tau_min_s=characteristic_tau_min_s,
        tau_max_s=characteristic_tau_max_s,
    )

    # v0.3: source calibration and level-process characteristic time are
    # trained here. Full [x,v,a,j] q/timescale identification is performed by
    # core.gated_training on the same fused history, so the legacy damped-
    # velocity DynamicsBank is intentionally not built.
    dynamics_points = _build_dynamics_fused(histories, calib, max_points=150000)
    return TrainingResult(
        calib, None, None, characteristic, fused, step, span,
        startup_pair_rows=list(startup_pair_rows),
        calibration_window_s=float(recent_window),
        dynamics_points=dynamics_points,
    )


class OnlineSourceCalibrator:
    """Slow bias tracker plus rolling pairwise noise calibration.

    Startup pairwise variance evidence is retained as a compact prior and ages
    out over the same calibration window that produced the startup sigma.  This
    avoids the previous discontinuity where seven days of Recorder evidence
    could be replaced by the first few seconds/minutes of live pairs after a
    restart or YAML reload.
    """

    def __init__(self, calibrations: dict[str, SourceCalibration], *,
                 startup_pair_rows=None, calibration_window_s: float | None = None,
                 bias_anchor: str = "median", source_models=None, model_accuracy=None,
                 huber_delta: float = 1.345):
        self.calibrations = calibrations
        self.bias_anchor = str(bias_anchor or "median").strip().lower()
        self.source_models = dict(source_models or {})
        self.model_accuracy = {str(k): float(v) for k, v in (model_accuracy or {}).items()}
        self.huber_delta = float(huber_delta)
        self.last_anchor = BiasAnchorResult(mode=self.bias_anchor, shift=0.0)
        self.cache = {}  # src -> (t, raw value)
        # Cross-source snapshot residuals are used only for relative bias.
        self.residuals = {src: deque() for src in calibrations}
        # Live close-in-time residuals.  Historical evidence is kept compactly
        # in startup_pair_rows below rather than as millions of Python tuples.
        self.pair_residuals: dict[tuple[str, str], deque] = {}
        self.last_pair_sample: dict[tuple[str, str], tuple[float, float]] = {}
        self.startup_pair_rows: dict[tuple[str, str], tuple[str, str, float, int, float]] = {}
        for row in startup_pair_rows or []:
            sa, sb, pair_var, count, span = row
            key = self._pair_key(sa, sb)
            self.startup_pair_rows[key] = (
                key[0], key[1], float(pair_var), int(count), float(span)
            )
        try:
            w = float(calibration_window_s) if calibration_window_s is not None else 0.0
        except (TypeError, ValueError):
            w = 0.0
        self.calibration_window_s = w if math.isfinite(w) and w > 0 else 0.0
        self._online_start_time: float | None = None

        # Online evidence is tracked separately so diagnostics show both the
        # Recorder/history basis and what has accumulated since startup.
        self.live_calibration_samples = {src: 0 for src in calibrations}
        self.live_calibration_span = {src: 0.0 for src in calibrations}
        self.live_calibration_pairs = {src: 0 for src in calibrations}
        self._last_updated_source: str | None = None

        # Adaptive full-calibration scheduler. Every observation only updates
        # cheap O(1) drift statistics; O(history) median/MAD/variance fitting
        # is deferred to a scheduled refit.
        self._last_refit_ts: float = 0.0
        self._refit_runs: int = 0
        self._refit_failures: int = 0
        self._refit_last_ms: float = 0.0
        self._refit_max_ms: float = 0.0
        self._refit_last_points: int = 0
        self._calibration_mode: str = "stable"
        self._drift_score: float = 0.0
        self._drift: dict[str, dict[str, float]] = {}


    def dump_compact(self, now: float, tau: float) -> dict:
        """Persist compact rolling calibration evidence.

        The potentially large live pair-residual deques are collapsed into the
        same pair-variance rows used by startup calibration.  After restart
        these rows become the new historical portion of the rolling window, so
        learning continues without rereading the full Recorder archive.
        """
        dynamic_window = max(
            3.0 * float(tau),
            10.0 * _median([c.median_dt for c in self.calibrations.values()], 60.0),
            3600.0,
        )
        window = self.calibration_window_s or dynamic_window
        live_rows, _counts, _spans = self._pair_rows_live(float(now), window)
        rows = self._combine_startup_and_live_rows(live_rows, now=float(now), window=window)
        return {
            "startup_pair_rows": [list(r) for r in rows],
            "calibration_window_s": float(window),
            "cache": {k: [float(v[0]), float(v[1])] for k, v in self.cache.items()},
            "last_updated_source": self._last_updated_source,
            "snapshot_time": float(now),
            "last_refit_ts": float(self._last_refit_ts),
            "refit_runs": int(self._refit_runs),
            "refit_failures": int(self._refit_failures),
            "refit_last_ms": float(self._refit_last_ms),
            "refit_max_ms": float(self._refit_max_ms),
            "refit_last_points": int(self._refit_last_points),
            "calibration_mode": str(self._calibration_mode),
            "drift_score": float(self._drift_score),
            "drift": {k: dict(v) for k, v in self._drift.items()},
        }

    @classmethod
    def load_compact(cls, data: dict | None, calibrations: dict[str, SourceCalibration], *,
                     bias_anchor: str = "median", source_models=None, model_accuracy=None,
                     huber_delta: float = 1.345):
        if not data:
            return cls(calibrations, bias_anchor=bias_anchor, source_models=source_models,
                       model_accuracy=model_accuracy, huber_delta=huber_delta)
        obj = cls(
            calibrations,
            startup_pair_rows=data.get("startup_pair_rows") or [],
            calibration_window_s=data.get("calibration_window_s"),
            bias_anchor=bias_anchor, source_models=source_models,
            model_accuracy=model_accuracy, huber_delta=huber_delta,
        )
        for src, pair in (data.get("cache") or {}).items():
            try:
                if src in calibrations:
                    obj.cache[src] = (float(pair[0]), float(pair[1]))
            except Exception:
                continue
        obj._last_updated_source = data.get("last_updated_source")
        # Compact rows represent the rolling window at checkpoint time.  Age
        # that historical evidence across downtime/catch-up exactly as if the
        # process had never restarted.
        try:
            obj._online_start_time = float(data.get("snapshot_time"))
        except (TypeError, ValueError):
            obj._online_start_time = None

        try:
            checkpoint_ts = float(data.get("snapshot_time", 0.0) or 0.0)
        except (TypeError, ValueError):
            checkpoint_ts = 0.0
        try:
            obj._last_refit_ts = float(
                data.get("last_refit_ts", checkpoint_ts) or checkpoint_ts
            )
        except (TypeError, ValueError):
            obj._last_refit_ts = checkpoint_ts

        obj._refit_runs = int(data.get("refit_runs", 0) or 0)
        obj._refit_failures = int(data.get("refit_failures", 0) or 0)
        obj._refit_last_ms = float(data.get("refit_last_ms", 0.0) or 0.0)
        obj._refit_max_ms = float(data.get("refit_max_ms", 0.0) or 0.0)
        obj._refit_last_points = int(data.get("refit_last_points", 0) or 0)
        obj._calibration_mode = str(
            data.get("calibration_mode", "stable") or "stable"
        )
        obj._drift_score = float(data.get("drift_score", 0.0) or 0.0)

        raw_drift = data.get("drift") or {}
        if isinstance(raw_drift, dict):
            for src, row in raw_drift.items():
                if src not in calibrations or not isinstance(row, dict):
                    continue
                try:
                    baseline_outlier = min(
                        max(
                            float(
                                row.get(
                                    "outlier_baseline",
                                    calibrations[src].outlier_rate,
                                )
                            ),
                            0.0,
                        ),
                        0.5,
                    )
                    obj._drift[src] = {
                        "abs_z": float(row.get("abs_z", 0.8)),
                        "outlier": float(
                            row.get("outlier", baseline_outlier)
                        ),
                        "outlier_baseline": baseline_outlier,
                        "weight": float(row.get("weight", 1.0)),
                        "last_t": float(row.get("last_t", checkpoint_ts)),
                    }
                except (TypeError, ValueError):
                    continue
        return obj

    def seed_startup_evidence(self, rows, calibration_window_s: float | None = None):
        """Seed compact startup evidence when history training arrives later."""
        if rows:
            self.startup_pair_rows.clear()
            for row in rows:
                sa, sb, pair_var, count, span = row
                key = self._pair_key(sa, sb)
                self.startup_pair_rows[key] = (
                    key[0], key[1], float(pair_var), int(count), float(span)
                )
        if calibration_window_s is not None:
            try:
                w = float(calibration_window_s)
            except (TypeError, ValueError):
                w = 0.0
            if math.isfinite(w) and w > 0:
                self.calibration_window_s = w
        # New startup evidence defines a new rolling-window origin.
        self._online_start_time = None

    def ensure_source(self, src: str, value: float, t: float):
        if src not in self.calibrations:
            self.calibrations[src] = SourceCalibration(
                bias=0.0, sigma=max(abs(value) * 1e-3, 1e-6),
                median_dt=60.0, typical_abs_level=max(abs(value), 1e-9), samples=0,
            )
            self.residuals[src] = deque()
            self.live_calibration_samples[src] = 0
            self.live_calibration_span[src] = 0.0
            self.live_calibration_pairs[src] = 0
        baseline_outlier = min(max(float(c.outlier_rate), 0.0), 0.5)
        self._drift.setdefault(
            src,
            {
                "abs_z": 0.8,
                "outlier": baseline_outlier,
                "outlier_baseline": baseline_outlier,
                "weight": 1.0,
                "last_t": float(t),
            },
        )
        self.cache[src] = (float(t), float(value))
        self._last_updated_source = src

    @staticmethod
    def _pair_key(a: str, b: str):
        return (a, b) if a < b else (b, a)

    def normalize_bias_gauge(self):
        self.last_anchor = normalize_bias_gauge(
            self.calibrations, mode=self.bias_anchor,
            source_models=self.source_models, model_accuracy=self.model_accuracy,
            residuals=self.residuals, huber_delta=self.huber_delta,
        )
        return self.last_anchor

    def _record_close_pairs(self, now: float, tau: float, updated_source: str):
        if updated_source not in self.cache:
            return False
        c0 = self.calibrations[updated_source]
        t0, z0 = self.cache[updated_source]
        added = False
        for peer, (tp, zp) in self.cache.items():
            if peer == updated_source:
                continue
            cp = self.calibrations[peer]
            # Record a pair only when the slower source updates.  This prevents
            # one slow sample from being reused by every update of a fast peer.
            if c0.median_dt < cp.median_dt * 0.95:
                continue
            if abs(c0.median_dt - cp.median_dt) <= 0.05 * max(c0.median_dt, cp.median_dt):
                # Equal-rate sources: deterministic tie-break avoids duplicates.
                if updated_source > peer:
                    continue
            tol = _pair_match_tolerance(c0.median_dt, cp.median_dt, tau)
            if abs(t0 - tp) > tol:
                continue
            key = self._pair_key(updated_source, peer)
            stamp = (float(t0), float(tp))
            if self.last_pair_sample.get(key) == stamp:
                continue
            # Bias does not affect pair variance after robust centering, but
            # correcting it keeps stored numbers numerically well centred.
            r = (z0 - c0.bias) - (zp - cp.bias)
            if not math.isfinite(r):
                continue
            self.pair_residuals.setdefault(key, deque()).append((float(now), float(r)))
            self.last_pair_sample[key] = stamp
            added = True
        return added

    def _pair_rows_live(self, now: float, window: float):
        cutoff = now - window
        rows = []
        counts = {src: 0 for src in self.calibrations}
        spans = {src: 0.0 for src in self.calibrations}
        for key, dq in list(self.pair_residuals.items()):
            while dq and dq[0][0] < cutoff:
                dq.popleft()
            if len(dq) < 8:
                continue
            vals = [v for _, v in dq]
            ps = _mad(vals)
            if not math.isfinite(ps) or ps <= 0:
                continue
            span = max(dq[-1][0] - dq[0][0], 0.0)
            sa, sb = key
            n = len(dq)
            rows.append((sa, sb, ps * ps, n, span))
            counts[sa] += n
            counts[sb] += n
            spans[sa] = max(spans[sa], span)
            spans[sb] = max(spans[sb], span)
        return rows, counts, spans

    def _combine_startup_and_live_rows(self, live_rows, *, now: float, window: float):
        """Blend startup and live pair evidence by rolling-window time coverage.

        The startup row summarizes the historical part of [now-window, now].
        As wall-clock time advances after startup, that historical portion ages
        out linearly while live rows fill the same interval.  Once one full
        window has elapsed, startup evidence contributes exactly zero.
        """
        live_map = {self._pair_key(r[0], r[1]): r for r in live_rows}
        if not self.startup_pair_rows:
            return list(live_rows)

        if self._online_start_time is None:
            self._online_start_time = float(now)
        elapsed = max(float(now) - self._online_start_time, 0.0)
        w = max(float(window), 1.0)
        old_fraction = max(1.0 - min(elapsed / w, 1.0), 0.0)

        combined = []
        for key in sorted(set(self.startup_pair_rows) | set(live_map)):
            sr = self.startup_pair_rows.get(key)
            lr = live_map.get(key)

            if sr is None:
                combined.append(lr)
                continue
            if old_fraction <= 0.0:
                if lr is not None:
                    combined.append(lr)
                continue
            if lr is None:
                sa, sb, svar, scount, sspan = sr
                remaining_count = int(round(old_fraction * max(int(scount), 1)))
                if remaining_count >= 8:
                    remaining_span = max(float(sspan) - elapsed, 0.0)
                    combined.append((
                        sa, sb, float(svar), remaining_count, remaining_span
                    ))
                continue

            sa, sb, svar, scount, sspan = sr
            _la, _lb, lvar, lcount, lspan = lr
            # Live evidence is weighted by the fraction of the rolling window
            # it actually spans.  This prevents a dense 50-second burst from
            # outweighing seven days of startup evidence simply because it has
            # many samples.
            live_fraction = min(max(float(lspan), 0.0) / w, 1.0)
            denom = old_fraction + live_fraction
            if denom <= 0.0:
                continue
            pair_var = (
                old_fraction * float(svar) + live_fraction * float(lvar)
            ) / denom
            # Retain the fraction of startup observations that is still
            # inside the rolling window, then add the actual live observations.
            # _solve_source_variances caps this count at 64 internally.
            count = max(1, int(round(
                old_fraction * max(int(scount), 1) + max(int(lcount), 1)
            )))
            remaining_startup_span = max(float(sspan) - elapsed, 0.0)
            span = min(w, remaining_startup_span + max(float(lspan), 0.0))
            combined.append((sa, sb, float(pair_var), count, float(span)))
        return combined

    def _combined_source_evidence(self, live_counts, live_spans, pair_rows, *, now, window):
        """Return diagnostics for the current startup+live working evidence."""
        if self._online_start_time is None:
            self._online_start_time = float(now)
        elapsed = max(float(now) - self._online_start_time, 0.0)
        w = max(float(window), 1.0)
        old_fraction = max(1.0 - min(elapsed / w, 1.0), 0.0)

        pair_counts = {src: 0 for src in self.calibrations}
        for sa, sb, _var, _count, _span in pair_rows:
            pair_counts[sa] = pair_counts.get(sa, 0) + 1
            pair_counts[sb] = pair_counts.get(sb, 0) + 1

        counts = {}
        spans = {}
        for src, c in self.calibrations.items():
            live_span = float(live_spans.get(src, 0.0))
            if old_fraction > 0.0:
                counts[src] = int(round(
                    old_fraction * int(c.startup_calibration_samples)
                    + int(live_counts.get(src, 0))
                ))
                remaining_startup_span = max(
                    float(c.startup_calibration_span) - elapsed, 0.0
                )
                spans[src] = min(w, remaining_startup_span + live_span)
            else:
                counts[src] = int(live_counts.get(src, 0))
                spans[src] = live_span
        return counts, spans, pair_counts

    def observe_innovation(
        self, src: str, t: float, z_score: float, weight: float = 1.0
    ):
        """Update cheap O(1) drift statistics for one source."""
        if src not in self.calibrations:
            return
        try:
            z = abs(float(z_score))
            w = float(weight)
            now = float(t)
        except (TypeError, ValueError):
            return
        if not math.isfinite(z) or not math.isfinite(w) or not math.isfinite(now):
            return

        c = self.calibrations[src]
        baseline_outlier = min(max(float(c.outlier_rate), 0.0), 0.5)
        d = self._drift.setdefault(
            src,
            {
                "abs_z": 0.8,
                "outlier": baseline_outlier,
                "outlier_baseline": baseline_outlier,
                "weight": 1.0,
                "last_t": now,
            },
        )
        last_t = float(d.get("last_t", now))
        dt = max(now - last_t, 0.0)

        # About 15 minutes effective memory. Sparse sources naturally receive
        # larger updates when they finally report.
        tau_ewma = 900.0
        alpha = 1.0 - math.exp(-dt / tau_ewma) if dt > 0 else 0.02
        alpha = min(max(alpha, 0.01), 0.35)

        d["abs_z"] = (
            (1.0 - alpha) * float(d.get("abs_z", 0.8)) + alpha * z
        )
        outlier = 1.0 if z > 3.0 or w < 0.25 else 0.0
        d["outlier"] = (
            (1.0 - alpha) * float(d.get("outlier", 0.0)) + alpha * outlier
        )
        d["weight"] = (
            (1.0 - alpha) * float(d.get("weight", 1.0)) + alpha * w
        )
        d["last_t"] = now
        self._update_drift_mode()

    def _update_drift_mode(self):
        score = 0.0
        for d in self._drift.values():
            abs_z = max(float(d.get("abs_z", 0.8)), 0.0)
            outlier = min(max(float(d.get("outlier", 0.0)), 0.0), 1.0)
            weight = min(max(float(d.get("weight", 1.0)), 0.0), 2.0)

            # Correctly scaled Gaussian innovations have E|z| ~= 0.8.
            z_term = max(abs_z - 1.0, 0.0) / 0.8
            baseline_outlier = min(
                max(float(d.get("outlier_baseline", 0.0)), 0.0), 0.5
            )
            # Drift means deterioration relative to this sensor's own normal
            # outlier behaviour. A sensor that historically sits at 4% must
            # not be permanently classified as WATCH merely for remaining at 4%.
            outlier_excess = max(outlier - baseline_outlier, 0.0)
            outlier_scale = max(0.03, baseline_outlier * 2.0)
            outlier_term = outlier_excess / outlier_scale

            weight_term = max(0.75 - weight, 0.0) / 0.50
            score = max(score, z_term, outlier_term, weight_term)

        self._drift_score = float(score)
        old = self._calibration_mode

        # Fast escalation, slower relaxation.
        if old == "unstable":
            if score < 0.7:
                self._calibration_mode = "watch" if score >= 0.35 else "stable"
        elif old == "watch":
            if score >= 1.5:
                self._calibration_mode = "unstable"
            elif score < 0.35:
                self._calibration_mode = "stable"
        else:
            if score >= 1.5:
                self._calibration_mode = "unstable"
            elif score >= 0.6:
                self._calibration_mode = "watch"

    def _refit_interval(self, window: float) -> float:
        """Return adaptive heavy-refit cadence.

        Stable has NO upper clamp: roughly two refits per calibration window.
        Watch/unstable may accelerate, but remain guarded from rapid polling.
        """
        w = max(float(window), 1.0)
        if self._calibration_mode == "unstable":
            # Long windows may accelerate to at most once per 2 hours.
            return max(min(w / 12.0, 7200.0), 900.0)
        if self._calibration_mode == "watch":
            # At most once per 6 hours while suspicious.
            return max(min(w / 6.0, 21600.0), 1800.0)
        # Stable: exactly the intended slow policy, about twice per window.
        return max(w / 2.0, 3600.0)

    def scheduler_diagnostics(self, now: float, tau: float) -> dict:
        dynamic_window = max(
            3.0 * float(tau),
            10.0 * _median(
                [c.median_dt for c in self.calibrations.values()], 60.0
            ),
            3600.0,
        )
        window = self.calibration_window_s or dynamic_window
        interval = self._refit_interval(window)
        age = (
            max(float(now) - self._last_refit_ts, 0.0)
            if self._last_refit_ts
            else None
        )
        return {
            "mode": self._calibration_mode,
            "drift_score": float(self._drift_score),
            "drift_sources": {
                src: {
                    "abs_z": round(float(d.get("abs_z", 0.0)), 3),
                    "outlier": round(float(d.get("outlier", 0.0)), 4),
                    "outlier_baseline": round(
                        float(d.get("outlier_baseline", 0.0)), 4
                    ),
                    "weight": round(float(d.get("weight", 1.0)), 3),
                }
                for src, d in self._drift.items()
            },
            "window_s": float(window),
            "interval_s": float(interval),
            "last_refit_age_s": age,
            "next_refit_due_s": (
                0.0 if age is None else max(interval - age, 0.0)
            ),
            "refit_runs": int(self._refit_runs),
            "refit_failures": int(self._refit_failures),
            "refit_last_ms": float(self._refit_last_ms),
            "refit_max_ms": float(self._refit_max_ms),
            "refit_last_points": int(self._refit_last_points),
            "residual_points": sum(len(dq) for dq in self.residuals.values()),
            "pair_residual_points": sum(
                len(dq) for dq in self.pair_residuals.values()
            ),
        }

    def update_snapshot(
        self, now: float, tau: float, updated_source: str | None = None
    ):
        """Accumulate evidence cheaply and run O(history) work only when due."""
        if len(self.cache) < 2:
            return False
        if self._online_start_time is None:
            self._online_start_time = float(now)

        fresh = {}
        for src, (t, z) in self.cache.items():
            c = self.calibrations[src]
            max_age = max(3.0 * c.median_dt, 0.20 * tau, 5.0)
            if now - t <= max_age:
                fresh[src] = z
        if len(fresh) < 2:
            return False

        corrected = [
            z - self.calibrations[src].bias for src, z in fresh.items()
        ]
        ref = _median(corrected)
        dynamic_window = max(
            3.0 * tau,
            10.0 * _median(
                [self.calibrations[s].median_dt for s in fresh], 60.0
            ),
            3600.0,
        )
        window = self.calibration_window_s or dynamic_window
        cutoff = now - window

        # Cheap accumulation only; no median/MAD over the whole window.
        for src, z in fresh.items():
            dq = self.residuals.setdefault(src, deque())
            dq.append((now, z - ref))
            while dq and dq[0][0] < cutoff:
                dq.popleft()

        src_now = updated_source or self._last_updated_source
        if src_now:
            self._record_close_pairs(now, tau, src_now)

        for dq in self.pair_residuals.values():
            while dq and dq[0][0] < cutoff:
                dq.popleft()

        interval = self._refit_interval(window)
        if self._last_refit_ts and now - self._last_refit_ts < interval:
            return False

        return self._full_refit(now, window)

    def _full_refit(self, now: float, window: float):
        """Run the expensive rolling bias/noise calibration pass."""
        started = time.perf_counter()
        self._refit_runs += 1
        self._last_refit_ts = float(now)
        point_count = (
            sum(len(dq) for dq in self.residuals.values())
            + sum(len(dq) for dq in self.pair_residuals.values())
        )
        self._refit_last_points = int(point_count)

        try:
            for src, dq in self.residuals.items():
                if src not in self.calibrations or len(dq) < 8:
                    continue
                c = self.calibrations[src]
                vals = [v for _, v in dq]
                b = _median(vals)
                alpha = min(
                    0.05, max(c.median_dt / max(window, 1.0), 0.005)
                )
                c.bias = (1.0 - alpha) * c.bias + alpha * b

            self.normalize_bias_gauge()

            live_rows, live_counts, live_spans = self._pair_rows_live(
                now, window
            )
            live_pair_counts = {src: 0 for src in self.calibrations}
            for sa, sb, _var, _count, _span in live_rows:
                live_pair_counts[sa] = live_pair_counts.get(sa, 0) + 1
                live_pair_counts[sb] = live_pair_counts.get(sb, 0) + 1

            for src in self.calibrations:
                self.live_calibration_samples[src] = int(
                    live_counts.get(src, 0)
                )
                self.live_calibration_span[src] = float(
                    live_spans.get(src, 0.0)
                )
                self.live_calibration_pairs[src] = int(
                    live_pair_counts.get(src, 0)
                )

            rows = self._combine_startup_and_live_rows(
                live_rows, now=now, window=window
            )
            solved = _solve_source_variances(self.calibrations, rows)

            if solved:
                counts, spans, pair_counts = self._combined_source_evidence(
                    live_counts, live_spans, rows, now=now, window=window
                )
                for src, var in solved.items():
                    c = self.calibrations[src]
                    target = math.sqrt(max(var, 1e-12))
                    alpha = min(
                        0.05,
                        max(c.median_dt / max(window, 1.0), 0.005),
                    )
                    floor = max(c.typical_abs_level * 1e-6, 1e-9)
                    c.sigma = max(
                        (1.0 - alpha) * c.sigma + alpha * target,
                        floor,
                    )
                    c.calibration_samples = int(counts.get(src, 0))
                    c.calibration_span = float(spans.get(src, 0.0))
                    c.calibration_pairs = int(pair_counts.get(src, 0))

            return True
        except Exception:
            self._refit_failures += 1
            raise
        finally:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            self._refit_last_ms = float(elapsed_ms)
            self._refit_max_ms = max(
                self._refit_max_ms, float(elapsed_ms)
            )