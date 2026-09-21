"""Robust multi-source history calibration and dynamics pre-training."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from bisect import bisect_right
from collections import deque
import math
import statistics
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

def _normalize_bias_gauge(calibrations, residuals=None):
    """Anchor relative source biases to a unique, robust zero point.

    Pairwise/source-residual calibration identifies only bias differences.
    Without an explicit gauge all biases can drift together by an arbitrary
    constant, shifting the absolute ensemble level while leaving every
    pairwise residual unchanged.

    The filter fuses sources by a robust median, so use the matching gauge:

        median(bias_i) = 0

    If live residual deques are supplied they are shifted by the same amount
    so their stored targets stay in the new gauge.

    Returns the removed common offset.  Corrected measurements would move by
    the same signed amount.
    """
    finite = [
        float(c.bias)
        for c in calibrations.values()
        if math.isfinite(float(c.bias))
    ]
    if not finite:
        return 0.0
    gauge = float(_median(finite, 0.0))
    if not math.isfinite(gauge) or abs(gauge) <= 1e-15:
        return 0.0

    for c in calibrations.values():
        if math.isfinite(float(c.bias)):
            c.bias = float(c.bias) - gauge

    if residuals is not None:
        for src, dq in list(residuals.items()):
            if dq:
                residuals[src] = deque((float(t), float(v) - gauge) for t, v in dq)

    return gauge


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
                      characteristic_tau_max_s: float | None = None) -> TrainingResult:
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

    # Bias is identifiable only up to a common additive constant.  Anchor the
    # startup solution to the same robust centre used by source fusion.
    _normalize_bias_gauge(calib)

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
                 startup_pair_rows=None, calibration_window_s: float | None = None):
        self.calibrations = calibrations
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
        }

    @classmethod
    def load_compact(cls, data: dict | None, calibrations: dict[str, SourceCalibration]):
        if not data:
            return cls(calibrations)
        obj = cls(
            calibrations,
            startup_pair_rows=data.get("startup_pair_rows") or [],
            calibration_window_s=data.get("calibration_window_s"),
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
        self.cache[src] = (float(t), float(value))
        self._last_updated_source = src

    @staticmethod
    def _pair_key(a: str, b: str):
        return (a, b) if a < b else (b, a)

    def normalize_bias_gauge(self) -> float:
        """Enforce median(source bias) == 0 and keep residual history aligned."""
        return _normalize_bias_gauge(self.calibrations, self.residuals)

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

    def update_snapshot(self, now: float, tau: float, updated_source: str | None = None):
        if len(self.cache) < 2:
            return 0.0
        if self._online_start_time is None:
            self._online_start_time = float(now)
        fresh = {}
        for src, (t, z) in self.cache.items():
            c = self.calibrations[src]
            max_age = max(3.0 * c.median_dt, 0.20 * tau, 5.0)
            if now - t <= max_age:
                fresh[src] = z
        if len(fresh) < 2:
            return 0.0

        corrected = [z - self.calibrations[src].bias for src, z in fresh.items()]
        ref = _median(corrected)
        dynamic_window = max(
            3.0 * tau,
            10.0 * _median([self.calibrations[s].median_dt for s in fresh], 60.0),
            3600.0,
        )
        # When startup history exists, keep the exact calibration horizon that
        # produced startup sigma.  This makes restart/reload continuous.  With
        # no startup history we retain the previous dynamic-window behavior.
        window = self.calibration_window_s or dynamic_window
        cutoff = now - window

        # Bias remains a robust cross-source quantity.  Bias adaptation keeps
        # its existing live-window behavior; the continuity fix in 0.2.1.7 is
        # specifically for pairwise sigma evidence.
        for src, z in fresh.items():
            dq = self.residuals.setdefault(src, deque())
            dq.append((now, z - ref))
            while dq and dq[0][0] < cutoff:
                dq.popleft()
            c = self.calibrations[src]
            if len(dq) >= 8:
                vals = [v for _, v in dq]
                b = _median(vals)
                alpha = min(0.05, max(c.median_dt / max(window, 1.0), 0.005))
                c.bias = (1.0 - alpha) * c.bias + alpha * b

        # Relative biases have one unconstrained common mode.  Re-anchor that
        # mode after every live update so the absolute ensemble cannot drift
        # away from the raw robust centre while pairwise differences remain
        # unchanged.
        gauge_shift = self.normalize_bias_gauge()

        src_now = updated_source or self._last_updated_source
        if not src_now or not self._record_close_pairs(now, tau, src_now):
            return gauge_shift

        live_rows, live_counts, live_spans = self._pair_rows_live(now, window)
        live_pair_counts = {src: 0 for src in self.calibrations}
        for sa, sb, _var, _count, _span in live_rows:
            live_pair_counts[sa] = live_pair_counts.get(sa, 0) + 1
            live_pair_counts[sb] = live_pair_counts.get(sb, 0) + 1
        for src in self.calibrations:
            self.live_calibration_samples[src] = int(live_counts.get(src, 0))
            self.live_calibration_span[src] = float(live_spans.get(src, 0.0))
            self.live_calibration_pairs[src] = int(live_pair_counts.get(src, 0))

        rows = self._combine_startup_and_live_rows(live_rows, now=now, window=window)
        solved = _solve_source_variances(self.calibrations, rows)
        if not solved:
            return gauge_shift

        counts, spans, pair_counts = self._combined_source_evidence(
            live_counts, live_spans, rows, now=now, window=window
        )
        for src, var in solved.items():
            c = self.calibrations[src]
            # rows already carry the correct startup/live time weighting; keep
            # the existing sigma smoother and only change the evidence it sees.
            target = math.sqrt(max(var, 1e-12))
            alpha = min(0.05, max(c.median_dt / max(window, 1.0), 0.005))
            floor = max(c.typical_abs_level * 1e-6, 1e-9)
            c.sigma = max((1.0 - alpha) * c.sigma + alpha * target, floor)
            c.calibration_samples = int(counts.get(src, 0))
            c.calibration_span = float(spans.get(src, 0.0))
            c.calibration_pairs = int(pair_counts.get(src, 0))

        return gauge_shift

