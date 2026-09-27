from __future__ import annotations

from dataclasses import dataclass
import math
import statistics


@dataclass
class NoiseDetectionResult:
    """History-derived observation-noise description.

    ``family`` describes the stochastic variance law.  Quantization is an
    independent observation property and therefore never appears as a family.
    A Poisson-like source can be quantized just as a Gaussian source can.
    """

    family: str = "gaussian"
    confidence: float = 0.0
    quantization_step: float | None = None
    quantization_confidence: float = 0.0
    poisson_scale: float | None = None
    reason: str = "fallback"

    def dump(self) -> dict:
        return {
            "family": self.family,
            "confidence": float(self.confidence),
            "quantization_step": self.quantization_step,
            "quantization_confidence": float(self.quantization_confidence),
            "poisson_scale": self.poisson_scale,
            "reason": self.reason,
        }

    @classmethod
    def load(cls, data):
        if not isinstance(data, dict):
            return None
        family = str(data.get("family", "gaussian"))
        # v1 briefly exposed quantized_gaussian as a family.  Treat it as a
        # Gaussian stochastic model with independent quantization metadata.
        if family == "quantized_gaussian":
            family = "gaussian"
        if family not in {"gaussian", "poisson"}:
            return None
        qstep = data.get("quantization_step")
        qconf = data.get("quantization_confidence")
        if qconf is None and qstep is not None:
            # Migration of the dev.6 schema, where family confidence also
            # represented lattice confidence.
            qconf = data.get("confidence", 0.0)
        return cls(
            family=family,
            confidence=float(data.get("confidence", 0.0) or 0.0),
            quantization_step=(None if qstep is None else float(qstep)),
            quantization_confidence=float(qconf or 0.0),
            poisson_scale=(None if data.get("poisson_scale") is None else float(data["poisson_scale"])),
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


def _poisson_evidence(values, quant_step):
    """Assess whether local observation variance scales with signal level.

    Quantization is *not* evidence for or against Poisson noise.  It is used
    only to decide whether the available level span is large enough to resolve
    a variance-vs-level relationship above the value lattice.
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
        return {
            "status": "insufficient_level_span",
            "confidence": 0.20,
            "level_span": float(p90 - p10),
            "required_span": float(span_required),
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

    # Broad level coverage was available, but the expected variance/level law
    # was not observed.  This is positive evidence for the conservative
    # Gaussian fallback rather than merely an absence of data.
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
    # Do not claim one ensemble-wide lattice when member resolutions differ by
    # more than a factor of two.
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
        per_source.append((src, vals, step, qconf, pe))

    if not per_source:
        return NoiseDetectionResult(reason="insufficient_history")

    qstep, qconf = _consensus_quantization(per_source)

    poisson = [row[4] for row in per_source if row[4].get("status") == "poisson"]
    needed = max(1, math.ceil(len(per_source) * 0.6))
    if len(poisson) >= needed:
        return NoiseDetectionResult(
            family="poisson",
            confidence=float(statistics.median(p["confidence"] for p in poisson)),
            poisson_scale=float(statistics.median(p["scale"] for p in poisson)),
            quantization_step=qstep,
            quantization_confidence=qconf,
            reason="variance_scales_with_level",
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
        reason=reason,
    )
