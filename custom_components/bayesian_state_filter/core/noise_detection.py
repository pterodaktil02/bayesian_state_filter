from __future__ import annotations

from dataclasses import dataclass
import math
import statistics


@dataclass
class NoiseDetectionResult:
    """History-derived observation-noise description.

    ``family`` describes the stochastic variance law. Quantization is an
    independent observation property and therefore never appears as a family.
    A Poisson-like source can be quantized just as a Gaussian source can.
    """

    family: str = "gaussian"
    confidence: float = 0.0
    quantization_step: float | None = None
    quantization_confidence: float = 0.0
    poisson_scale: float | None = None
    level_variance_fraction: float | None = None
    level_variance_confidence: float = 0.0
    level_variance_reference: float | None = None
    level_variance_reason: str | None = None
    level_variance_boundary_limited: bool = False
    level_variance_span_z: float | None = None
    reason: str = "fallback"

    def dump(self) -> dict:
        return {
            "family": self.family,
            "confidence": float(self.confidence),
            "quantization_step": self.quantization_step,
            "quantization_confidence": float(self.quantization_confidence),
            "poisson_scale": self.poisson_scale,
            "level_variance_fraction": self.level_variance_fraction,
            "level_variance_confidence": float(self.level_variance_confidence),
            "level_variance_reference": self.level_variance_reference,
            "level_variance_reason": self.level_variance_reason,
            "level_variance_boundary_limited": bool(self.level_variance_boundary_limited),
            "level_variance_span_z": self.level_variance_span_z,
            "reason": self.reason,
        }

    @classmethod
    def load(cls, data):
        if not isinstance(data, dict):
            return None
        family = str(data.get("family", "gaussian"))
        if family == "quantized_gaussian":
            family = "gaussian"
        if family not in {"gaussian", "poisson"}:
            return None
        qstep = data.get("quantization_step")
        qconf = data.get("quantization_confidence")
        if qconf is None and qstep is not None:
            qconf = data.get("confidence", 0.0)
        return cls(
            family=family,
            confidence=float(data.get("confidence", 0.0) or 0.0),
            quantization_step=(None if qstep is None else float(qstep)),
            quantization_confidence=float(qconf or 0.0),
            poisson_scale=(None if data.get("poisson_scale") is None else float(data["poisson_scale"])),
            level_variance_fraction=(None if data.get("level_variance_fraction") is None
                                     else float(data["level_variance_fraction"])),
            level_variance_confidence=float(data.get("level_variance_confidence", 0.0) or 0.0),
            level_variance_reference=(None if data.get("level_variance_reference") is None
                                      else float(data["level_variance_reference"])),
            level_variance_reason=(None if data.get("level_variance_reason") is None
                                   else str(data["level_variance_reason"])),
            level_variance_boundary_limited=bool(
                data.get("level_variance_boundary_limited", False)
            ),
            level_variance_span_z=(None if data.get("level_variance_span_z") is None
                                   else float(data["level_variance_span_z"])),
            reason=str(data.get("reason", "restored")),
        )


def _finite_values(seq, max_points=20000):
    vals = []
    for row in seq or []:
        try:
            z = float(row[1])
        except (TypeError, ValueError, IndexError):
            continue
        if math.isfinite(z):
            vals.append(z)
    if len(vals) > max_points:
        step = len(vals) / float(max_points)
        vals = [vals[min(int(i * step), len(vals) - 1)] for i in range(max_points)]
    return vals


def _alignment_fraction(values, step):
    if not values or step <= 0:
        return 0.0
    origin = statistics.median(values)
    tol = max(step * 2e-6, 1e-9)
    good = 0
    for v in values:
        k = round((v - origin) / step)
        if abs((v - origin) - k * step) <= tol:
            good += 1
    return good / len(values)


def detect_quantization(values):
    """Detect a meaningful value lattice without inferring a noise family."""
    if len(values) < 64:
        return None, 0.0
    unique = len(set(values))
    repeated_fraction = 1.0 - unique / len(values)
    if repeated_fraction < 0.05:
        return None, 0.0

    candidates = [100.0, 50.0, 20.0, 10.0, 5.0, 2.0, 1.0,
                  0.5, 0.2, 0.1, 0.05, 0.02, 0.01,
                  0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]
    for step in candidates:
        aligned = _alignment_fraction(values, step)
        if aligned < 0.995:
            continue
        bins = len({round(v / step) for v in values})
        if bins < 4:
            continue
        confidence = min(
            1.0,
            0.65 * aligned + 0.35 * min(repeated_fraction / 0.25, 1.0),
        )
        return float(step), float(confidence)
    return None, 0.0


def _pearson(xs, ys):
    if len(xs) != len(ys) or len(xs) < 3:
        return 0.0
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    sx = sum((x - mx) ** 2 for x in xs)
    sy = sum((y - my) ** 2 for y in ys)
    if sx <= 0 or sy <= 0:
        return 0.0
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(sx * sy)


def _sample_skewness(values):
    """Population-style skewness used only as a long-history diagnostic."""
    n = len(values)
    if n < 3:
        return 0.0
    mean = statistics.mean(values)
    var = sum((x - mean) ** 2 for x in values) / n
    if var <= 0:
        return 0.0
    sigma = math.sqrt(var)
    return sum(((x - mean) / sigma) ** 3 for x in values) / n


def _stationary_poisson_evidence(values, quant_step):
    """Look for scaled-Poisson moment structure when level span is small.

    A stationary record cannot prove Var(X) proportional to E[X] from level
    dependence alone.  However, a scaled Poisson variable Y=k*N obeys two
    simultaneous moment relations:

        Var(Y) = k * E[Y]
        skew(Y) = sqrt(k / E[Y]) = sigma(Y) / E[Y]

    The second relation gives extra information that a stationary Gaussian
    source does not have.  This is deliberately conservative: it requires a
    long positive history, positive measurable skewness, and close agreement
    between observed and Poisson-predicted skewness.

    Quantization is not used as proof of Poisson noise; if present it only
    contributes a small amount of confidence after the moment test passes.
    """
    n = len(values)
    if n < 512:
        return {"status": "insufficient_stationary_history", "confidence": 0.0}
    if min(values) < 0:
        return {"status": "negative_values", "confidence": 0.75}

    mean = statistics.mean(values)
    if mean <= 0:
        return {"status": "nonpositive_level", "confidence": 0.75}

    var = statistics.pvariance(values)
    if not math.isfinite(var) or var <= 0:
        return {"status": "insufficient_variance", "confidence": 0.0}

    sigma = math.sqrt(var)
    scale = var / mean
    predicted_skew = sigma / mean
    observed_skew = _sample_skewness(values)

    # For Gaussian data, the standard error of sample skewness is roughly
    # sqrt(6/n).  Requiring >2 sigma positive skew makes this fallback much
    # harder to trigger on ordinary stationary Gaussian sensors.
    skew_se = math.sqrt(6.0 / n)
    skew_z = observed_skew / max(skew_se, 1e-12)

    if observed_skew <= 0 or skew_z < 2.0:
        return {
            "status": "stationary_skew_not_count_like",
            "confidence": min(0.55, max(0.20, 0.20 + 0.10 * max(skew_z, 0.0))),
            "scale": float(scale),
            "observed_skew": float(observed_skew),
            "predicted_skew": float(predicted_skew),
            "skew_z": float(skew_z),
        }

    mismatch = abs(observed_skew - predicted_skew) / max(predicted_skew, 1e-12)
    if mismatch > 0.45:
        return {
            "status": "stationary_moment_mismatch",
            "confidence": 0.45,
            "scale": float(scale),
            "observed_skew": float(observed_skew),
            "predicted_skew": float(predicted_skew),
            "skew_z": float(skew_z),
            "relative_skew_mismatch": float(mismatch),
        }

    match = max(0.0, 1.0 - mismatch / 0.45)
    significance = min(1.0, max(0.0, (skew_z - 2.0) / 2.0))
    quant_bonus = 0.05 if quant_step is not None else 0.0
    confidence = min(0.85, 0.58 + 0.17 * match + 0.05 * significance + quant_bonus)

    return {
        "status": "stationary_poisson",
        "confidence": float(confidence),
        "scale": float(scale),
        "observed_skew": float(observed_skew),
        "predicted_skew": float(predicted_skew),
        "skew_z": float(skew_z),
        "relative_skew_mismatch": float(mismatch),
    }




def _rolling_median(values, half_window=15):
    """Centered rolling median used only for slow local level estimation."""
    out = [None] * len(values)
    n = len(values)
    for i in range(n):
        lo = max(0, i - half_window)
        hi = min(n, i + half_window + 1)
        if hi - lo >= max(7, half_window):
            out[i] = float(statistics.median(values[lo:hi]))
    return out


def _local_variance_law(seq):
    """Estimate whether fast observation variance grows with local level.

    Only the one-dimensional history is used.  Locally linear process motion
    is removed by predicting each interior sample from its two neighbours.
    The fitted diagnostic law is

        R(x) = R_ref * ((1-p) + p*x/x_ref)

    where p=0 is constant measurement variance and p=1 is the pure
    level-dependent limit.  This result is diagnostic-only for now.
    """
    clean = []
    for row in seq or []:
        try:
            t, z = float(row[0]), float(row[1])
        except (TypeError, ValueError, IndexError):
            continue
        if math.isfinite(t) and math.isfinite(z):
            clean.append((t, z))
    clean.sort(key=lambda p: p[0])
    if len(clean) < 512:
        return {"status": "insufficient_history", "confidence": 0.0}

    if len(clean) > 20000:
        step = len(clean) / 20000.0
        clean = [clean[min(int(i * step), len(clean) - 1)] for i in range(20000)]

    dts = [
        clean[i][0] - clean[i - 1][0]
        for i in range(1, len(clean))
        if clean[i][0] > clean[i - 1][0]
    ]
    if not dts:
        return {"status": "insufficient_timestamps", "confidence": 0.0}
    median_dt = max(float(statistics.median(dts)), 1e-9)

    values = [z for _, z in clean]
    levels = _rolling_median(values, half_window=15)
    rows = []
    for i in range(1, len(clean) - 1):
        level = levels[i]
        if level is None or not math.isfinite(level):
            continue
        t0, z0 = clean[i - 1]
        t1, z1 = clean[i]
        t2, z2 = clean[i + 1]
        left = t1 - t0
        right = t2 - t1
        span = t2 - t0
        if left <= 0 or right <= 0 or span <= 0:
            continue
        if left > 6.0 * median_dt or right > 6.0 * median_dt:
            continue
        w = left / span
        predicted = (1.0 - w) * z0 + w * z2
        residual = z1 - predicted
        factor2 = 1.0 + (1.0 - w) ** 2 + w ** 2
        rows.append((float(level), float((residual * residual) / factor2)))

    if len(rows) < 384:
        return {"status": "insufficient_local_residuals", "confidence": 0.0}

    rows.sort(key=lambda p: p[0])
    n = len(rows)
    centers, variances = [], []
    median_square_normal = 0.4549364231195727
    for b in range(8):
        lo = b * n // 8
        hi = (b + 1) * n // 8
        chunk = rows[lo:hi]
        if len(chunk) < 32:
            continue
        centers.append(float(statistics.median(x for x, _ in chunk)))
        med_sq = float(statistics.median(v for _, v in chunk))
        variances.append(max(med_sq / median_square_normal, 1e-15))

    if len(centers) < 6:
        return {"status": "insufficient_variance_bins", "confidence": 0.0}

    x_ref = float(statistics.median(centers))
    if not math.isfinite(x_ref) or x_ref <= 0:
        return {"status": "nonpositive_reference_level", "confidence": 0.0}

    ordered = sorted(centers)
    p10 = ordered[int(0.10 * (len(ordered) - 1))]
    p90 = ordered[int(0.90 * (len(ordered) - 1))]
    level_span = max(float(p90 - p10), 0.0)
    relative_span = level_span / x_ref
    corr = _pearson(centers, variances)

    # The rolling level estimate is itself noisy.  A stationary quantized or
    # count-like source can otherwise manufacture several apparent "levels"
    # from measurement noise alone.  Estimate the uncertainty of a 31-sample
    # rolling median and require the observed level span to exceed it by a
    # comfortable margin before claiming that R(x) is identifiable.
    overall_med_sq = float(statistics.median(v for _, v in rows))
    overall_variance = max(overall_med_sq / median_square_normal, 1e-15)
    measurement_sigma = math.sqrt(overall_variance)
    median_window_n = 31.0
    # Asymptotic stddev(median) for Gaussian samples: sqrt(pi/2)*sigma/sqrt(n).
    level_est_sigma = math.sqrt(math.pi / 2.0) * measurement_sigma / math.sqrt(median_window_n)
    span_z = level_span / max(level_est_sigma, 1e-12)

    def fit_for_fraction(p):
        shape = [(1.0 - p) + p * (x / x_ref) for x in centers]
        denom = sum(v * v for v in shape)
        if denom <= 0:
            return None
        scale = sum(v * r for v, r in zip(shape, variances)) / denom
        pred = [scale * v for v in shape]
        mse = sum((r - q) ** 2 for r, q in zip(variances, pred)) / len(pred)
        return float(scale), float(mse)

    constant_fit = fit_for_fraction(0.0)
    if constant_fit is None:
        return {"status": "fit_failed", "confidence": 0.0}
    constant_mse = constant_fit[1]
    best = (constant_mse, 0.0, constant_fit[0])
    for j in range(1, 101):
        p = j / 100.0
        fit = fit_for_fraction(p)
        if fit is not None and fit[1] < best[0]:
            best = (fit[1], p, fit[0])

    best_mse, level_fraction, ref_variance = best
    improvement = (
        max(0.0, (constant_mse - best_mse) / constant_mse)
        if constant_mse > 1e-30 else 0.0
    )

    # Two different identifiability failures must not be conflated:
    # 1) the apparent local-level movement is not large compared with the
    #    uncertainty of the level estimate itself (latent span not resolved);
    # 2) the latent movement is very well resolved, but is too small relative
    #    to the absolute reference level to separate constant and proportional
    #    variance terms reliably.
    if span_z < 6.0:
        return {
            "status": "latent_span_insufficient",
            "confidence": 0.20,
            "level_fraction": None,
            "reference_level": x_ref,
            "reference_variance": float(constant_fit[0]),
            "relative_span": relative_span,
            "span_z": float(span_z),
            "corr": float(corr),
            "improvement": improvement,
            "boundary_limited": False,
        }

    # With less than four percent relative level span, constant and
    # level-dependent variance are too collinear to separate reliably even
    # when the level change itself is measured with overwhelming S/N.
    if relative_span < 0.04:
        return {
            "status": "relative_level_span_insufficient",
            "confidence": 0.20,
            "level_fraction": None,
            "reference_level": x_ref,
            "reference_variance": float(constant_fit[0]),
            "relative_span": relative_span,
            "span_z": float(span_z),
            "corr": float(corr),
            "improvement": improvement,
            "boundary_limited": False,
        }

    if corr < 0.75 or improvement < 0.05 or level_fraction < 0.10:
        confidence = min(0.85, max(0.45, 0.70 - 0.25 * max(corr, 0.0)))
        return {
            "status": "constant_variance",
            "confidence": float(confidence),
            "level_fraction": 0.0,
            "reference_level": x_ref,
            "reference_variance": float(constant_fit[0]),
            "relative_span": relative_span,
            "span_z": float(span_z),
            "corr": float(corr),
            "improvement": improvement,
            "boundary_limited": False,
        }

    boundary_limited = bool(level_fraction <= 0.0 or level_fraction >= 1.0)
    confidence = min(
        0.85,
        0.45
        + 0.20 * min(max((corr - 0.75) / 0.25, 0.0), 1.0)
        + 0.15 * min(improvement / 0.15, 1.0)
        + 0.05 * min(relative_span / 0.15, 1.0),
    )
    return {
        "status": "level_dependent_variance",
        "confidence": float(confidence),
        "level_fraction": float(level_fraction),
        "reference_level": x_ref,
        "reference_variance": float(ref_variance),
        "relative_span": relative_span,
        "span_z": float(span_z),
        "corr": float(corr),
        "improvement": improvement,
        "boundary_limited": boundary_limited,
    }

def _poisson_evidence(values, quant_step):
    """Assess whether observation variance follows a Poisson-like law.

    Primary route: detect variance growth with signal level.
    Fallback route: for long but nearly stationary positive series, test the
    scaled-Poisson moment relation between variance and skewness.
    """
    if len(values) < 256:
        return {"status": "insufficient_history", "confidence": 0.0}
    if min(values) < 0:
        return {"status": "negative_values", "confidence": 0.75}

    ordered = sorted(values)
    med = statistics.median(ordered)
    if med <= 0:
        return {"status": "nonpositive_level", "confidence": 0.75}
    p10 = ordered[int(0.10 * (len(ordered) - 1))]
    p90 = ordered[int(0.90 * (len(ordered) - 1))]
    span_required = 0.15 * med
    if quant_step is not None:
        span_required = max(span_required, 8.0 * quant_step)

    if (p90 - p10) < span_required:
        stationary = _stationary_poisson_evidence(values, quant_step)
        if stationary.get("status") == "stationary_poisson":
            stationary["level_span"] = float(p90 - p10)
            stationary["required_span"] = float(span_required)
            return stationary
        return {
            "status": "insufficient_level_span",
            "confidence": 0.20,
            "level_span": float(p90 - p10),
            "required_span": float(span_required),
            "stationary_test": stationary,
        }

    pairs = []
    for a, b in zip(values[:-1], values[1:]):
        level = 0.5 * (a + b)
        if level <= 0:
            continue
        proxy = 0.5 * (b - a) ** 2
        pairs.append((level, proxy))
    if len(pairs) < 200:
        return {"status": "insufficient_pairs", "confidence": 0.1}

    pairs.sort(key=lambda p: p[0])
    nbins = 6
    centers, vars_ = [], []
    n = len(pairs)
    for i in range(nbins):
        lo = i * n // nbins
        hi = (i + 1) * n // nbins
        chunk = pairs[lo:hi]
        if len(chunk) < 20:
            continue
        lv = [x for x, _ in chunk]
        pv = [y for _, y in chunk]
        centers.append(statistics.median(lv))
        vars_.append(statistics.median(pv))
    if len(centers) < 4 or max(vars_, default=0.0) <= 0:
        return {"status": "insufficient_variance_bins", "confidence": 0.1}

    corr = _pearson(centers, vars_)
    ratios = [v / l for l, v in zip(centers, vars_) if l > 0 and v > 0]
    if len(ratios) < 4:
        return {"status": "insufficient_variance_bins", "confidence": 0.1}

    rmed_raw = statistics.median(ratios)
    rmed = rmed_raw / 0.4549364231195727
    mad = statistics.median(abs(r - rmed_raw) for r in ratios)
    rel_mad = mad / max(abs(rmed_raw), 1e-12)

    if corr >= 0.80 and rel_mad <= 0.55:
        confidence = min(
            1.0,
            max(0.0, (corr - 0.75) / 0.25) * max(0.0, 1.0 - rel_mad),
        )
        if confidence >= 0.55:
            return {
                "status": "poisson",
                "confidence": float(confidence),
                "scale": float(rmed),
                "corr": float(corr),
                "rel_mad": float(rel_mad),
            }

    gaussian_conf = min(0.90, max(0.55, 0.80 - 0.35 * max(corr, 0.0)))
    return {
        "status": "variance_not_level_dependent",
        "confidence": float(gaussian_conf),
        "corr": float(corr),
        "rel_mad": float(rel_mad),
    }


def _consensus_quantization(per_source):
    quantized = [(row[2], row[3]) for row in per_source if row[2] is not None]
    if len(quantized) < max(1, math.ceil(len(per_source) * 0.6)):
        return None, 0.0
    steps = [s for s, _ in quantized]
    confs = [c for _, c in quantized]
    med_step = float(statistics.median(steps))
    if min(steps) <= 0 or max(steps) / min(steps) > 2.0:
        return None, 0.0
    return med_step, float(statistics.median(confs))


def detect_noise_model(histories: dict[str, list[tuple[float, float]]]) -> NoiseDetectionResult:
    """Infer stochastic family and quantization as orthogonal properties."""
    per_source = []
    for src, seq in (histories or {}).items():
        vals = _finite_values(seq)
        if len(vals) < 64:
            continue
        step, qconf = detect_quantization(vals)
        pe = _poisson_evidence(vals, step)
        lv = _local_variance_law(seq)
        per_source.append((src, vals, step, qconf, pe, lv))

    if not per_source:
        return NoiseDetectionResult(reason="insufficient_history")

    qstep, qconf = _consensus_quantization(per_source)

    local_rows = [
        row[5] for row in per_source
        if row[5].get("status") in {
            "level_dependent_variance",
            "constant_variance",
            "insufficient_level_span",
        }
    ]
    lv_fraction = None
    lv_confidence = 0.0
    lv_reference = None
    lv_reason = None
    lv_boundary = False
    lv_span_z = None
    if local_rows:
        dependent = [
            r for r in local_rows
            if r.get("status") == "level_dependent_variance"
        ]
        needed_local = max(1, math.ceil(len(per_source) * 0.6))
        if len(dependent) >= needed_local:
            lv_fraction = float(statistics.median(
                float(r.get("level_fraction", 0.0)) for r in dependent
            ))
            lv_confidence = float(statistics.median(
                float(r.get("confidence", 0.0)) for r in dependent
            ))
            refs = [
                float(r["reference_level"]) for r in dependent
                if r.get("reference_level") is not None
            ]
            lv_reference = float(statistics.median(refs)) if refs else None
            lv_reason = "level_dependent_local_residuals"
            lv_boundary = any(bool(r.get("boundary_limited", False)) for r in dependent)
            span_vals = [float(r["span_z"]) for r in dependent if r.get("span_z") is not None]
            lv_span_z = float(statistics.median(span_vals)) if span_vals else None
        elif all(
            r.get("status") in {"latent_span_insufficient", "relative_level_span_insufficient"}
            for r in local_rows
        ):
            lv_fraction = None
            lv_confidence = 0.20
            refs = [
                float(r["reference_level"]) for r in local_rows
                if r.get("reference_level") is not None
            ]
            lv_reference = float(statistics.median(refs)) if refs else None
            statuses_local = [r.get("status") for r in local_rows]
            if all(s == "latent_span_insufficient" for s in statuses_local):
                lv_reason = "latent_span_insufficient"
            elif all(s == "relative_level_span_insufficient" for s in statuses_local):
                lv_reason = "relative_level_span_insufficient"
            else:
                # Mixed multi-source case: prefer the more conservative
                # statement that the latent span itself is not resolved if any
                # source fails that test.
                lv_reason = (
                    "latent_span_insufficient"
                    if "latent_span_insufficient" in statuses_local
                    else "relative_level_span_insufficient"
                )
            span_vals = [float(r["span_z"]) for r in local_rows if r.get("span_z") is not None]
            lv_span_z = float(statistics.median(span_vals)) if span_vals else None
        else:
            constant = [
                r for r in local_rows
                if r.get("status") == "constant_variance"
            ]
            if constant:
                lv_fraction = 0.0
                lv_confidence = float(statistics.median(
                    float(r.get("confidence", 0.0)) for r in constant
                ))
                refs = [
                    float(r["reference_level"]) for r in constant
                    if r.get("reference_level") is not None
                ]
                lv_reference = float(statistics.median(refs)) if refs else None
                lv_reason = "local_variance_constant"
                span_vals = [float(r["span_z"]) for r in constant if r.get("span_z") is not None]
                lv_span_z = float(statistics.median(span_vals)) if span_vals else None

    poisson_statuses = {"poisson", "stationary_poisson"}
    poisson = [row[4] for row in per_source if row[4].get("status") in poisson_statuses]
    needed = max(1, math.ceil(len(per_source) * 0.6))
    if len(poisson) >= needed:
        stationary_only = all(p.get("status") == "stationary_poisson" for p in poisson)
        return NoiseDetectionResult(
            family="poisson",
            confidence=float(statistics.median(p["confidence"] for p in poisson)),
            poisson_scale=float(statistics.median(p["scale"] for p in poisson)),
            quantization_step=qstep,
            quantization_confidence=qconf,
            level_variance_fraction=lv_fraction,
            level_variance_confidence=lv_confidence,
            level_variance_reference=lv_reference,
            level_variance_reason=lv_reason,
            level_variance_boundary_limited=lv_boundary,
            level_variance_span_z=lv_span_z,
            reason=("stationary_scaled_poisson" if stationary_only
                    else "variance_scales_with_level"),
        )

    statuses = [row[4].get("status") for row in per_source]
    confs = [float(row[4].get("confidence", 0.0)) for row in per_source]
    if statuses and all(s == "insufficient_level_span" for s in statuses):
        reason = "insufficient_level_span"
        confidence = float(statistics.median(confs)) if confs else 0.2
    elif any(s == "variance_not_level_dependent" for s in statuses):
        reason = "variance_not_level_dependent"
        usable = [float(row[4].get("confidence", 0.0)) for row in per_source
                  if row[4].get("status") == "variance_not_level_dependent"]
        confidence = float(statistics.median(usable)) if usable else 0.6
    elif any(s in {"negative_values", "nonpositive_level"} for s in statuses):
        reason = "not_count_like"
        confidence = 0.75
    else:
        reason = "insufficient_poisson_evidence"
        confidence = 0.25

    return NoiseDetectionResult(
        family="gaussian",
        confidence=confidence,
        quantization_step=qstep,
        quantization_confidence=qconf,
        level_variance_fraction=lv_fraction,
        level_variance_confidence=lv_confidence,
        level_variance_reference=lv_reference,
        level_variance_reason=lv_reason,
        level_variance_boundary_limited=lv_boundary,
        level_variance_span_z=lv_span_z,
        reason=reason,
    )
