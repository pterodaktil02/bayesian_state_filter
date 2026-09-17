"""History calibration regression tests."""

import math

import numpy as np

from custom_components.bayesian_state_filter.core.training import calibrate_history


def test_history_calibration_recovers_relative_biases():
    rng = np.random.default_rng(12345)
    times = np.arange(0.0, 6 * 3600.0, 30.0)
    latent = 22.0 + 0.4 * np.sin(times / 1800.0)

    offsets = {
        "sensor.a": -0.6,
        "sensor.b": 0.0,
        "sensor.c": 0.6,
    }
    histories = {}
    for source, offset in offsets.items():
        noise = rng.normal(0.0, 0.03, size=len(times))
        histories[source] = list(zip(times, latent + offset + noise))

    result = calibrate_history(histories, tau_points=8)

    assert set(result.sources) == set(offsets)
    assert abs(result.sources["sensor.a"].bias + 0.6) < 0.1
    assert abs(result.sources["sensor.b"].bias) < 0.1
    assert abs(result.sources["sensor.c"].bias - 0.6) < 0.1
    for calibration in result.sources.values():
        assert calibration.sigma > 0
        assert math.isfinite(calibration.sigma)


def _asynchronous_histories(seed=42):
    rng = np.random.default_rng(seed)
    span = 18 * 3600.0

    def latent(t):
        return 22.0 + 0.5 * np.sin(t / 1800.0) + 0.2 * np.sin(t / 7000.0)

    spec = {
        "sensor.fast": (15.0, 0.0, 0.06),
        "sensor.bmp": (60.0, 1.1, 0.35),
        "sensor.qing": (120.0, -0.05, 0.15),
        "sensor.cup": (1200.0, -1.3, 0.08),
        "sensor.door": (1170.0, -0.27, 0.10),
    }
    histories = {}
    for source, (dt, bias, sigma) in spec.items():
        times = np.arange(0.0, span, dt, dtype=float)
        if source != "sensor.fast":
            times = times + rng.normal(0.0, min(2.0, dt * 0.01), size=times.size)
        values = np.asarray([latent(t) for t in times]) + bias
        values = values + rng.normal(0.0, sigma, size=times.size)
        histories[source] = list(zip(times, values))
    return histories


def test_pairwise_sigma_is_finite_nonzero_and_network_is_connected():
    result = calibrate_history(_asynchronous_histories(), tau_points=12)

    assert len(result.fused_points) > 100
    for source, calibration in result.sources.items():
        assert math.isfinite(calibration.sigma)
        assert calibration.sigma > 1e-3
        # With five healthy asynchronous sources each source should have
        # evidence from multiple peers, even though individual sample counts
        # differ strongly by cadence.
        assert calibration.calibration_pairs >= 2, source
        assert calibration.calibration_samples >= 8, source


def test_startup_calibration_is_repeatable_for_identical_history():
    histories = _asynchronous_histories(seed=7)
    first = calibrate_history(histories, tau_points=12)
    second = calibrate_history(histories, tau_points=12)

    for source in first.sources:
        a = first.sources[source]
        b = second.sources[source]
        assert a.bias == b.bias
        assert a.sigma == b.sigma
        assert a.calibration_samples == b.calibration_samples
        assert a.calibration_pairs == b.calibration_pairs


def test_correlated_fast_source_does_not_collapse_sigma_to_zero():
    rng = np.random.default_rng(20260917)
    span = 12 * 3600.0

    def latent(t):
        return 22.0 + 0.4 * np.sin(t / 2200.0)

    histories = {}
    times = np.arange(0.0, span, 15.0)
    rho = 0.97
    stationary_sigma = 0.06
    innovation_sigma = stationary_sigma * math.sqrt(1.0 - rho * rho)
    e = 0.0
    values = []
    for t in times:
        e = rho * e + rng.normal(0.0, innovation_sigma)
        values.append(latent(t) + e)
    histories["sensor.fast"] = list(zip(times, values))

    for source, dt, bias, sigma in [
        ("sensor.mid", 60.0, 0.2, 0.08),
        ("sensor.slow", 120.0, -0.3, 0.10),
        ("sensor.s1", 600.0, 0.6, 0.12),
        ("sensor.s2", 900.0, -0.6, 0.15),
    ]:
        t = np.arange(0.0, span, dt)
        y = [latent(x) + bias + rng.normal(0.0, sigma) for x in t]
        histories[source] = list(zip(t, y))

    result = calibrate_history(histories, tau_points=8)
    fast_sigma = result.sources["sensor.fast"].sigma

    # Regression guard for the failed source-local high-order difference
    # approach, which could produce ~1e-3 or smaller for correlated sensors.
    assert fast_sigma > 0.02
    assert fast_sigma < 0.25


def test_pairwise_calibration_recovers_known_sigmas_on_stationary_latent():
    rng = np.random.default_rng(321)
    span = 12 * 3600.0
    spec = {
        "sensor.a": (15.0, 0.0, 0.05),
        "sensor.b": (30.0, 0.4, 0.08),
        "sensor.c": (45.0, -0.3, 0.12),
        "sensor.d": (60.0, 0.7, 0.16),
        "sensor.e": (75.0, -0.5, 0.20),
    }
    histories = {}
    for source, (dt, bias, sigma) in spec.items():
        times = np.arange(0.0, span, dt, dtype=float)
        times = times + rng.normal(0.0, min(0.5, dt * 0.005), size=times.size)
        values = 22.0 + bias + rng.normal(0.0, sigma, size=times.size)
        histories[source] = list(zip(times, values))

    result = calibrate_history(histories, tau_points=12)

    for source, (_dt, _bias, sigma_true) in spec.items():
        estimated = result.sources[source].sigma
        # Robust pairwise MAD is intentionally conservative and the samples
        # are asynchronous. Require recovery within 25% (or 0.03 absolute)
        # rather than testing an unrealistically exact estimator.
        tolerance = max(0.03, 0.25 * sigma_true)
        assert abs(estimated - sigma_true) < tolerance, (
            source, sigma_true, estimated
        )
        assert result.sources[source].calibration_pairs == 4


def test_history_calibration_is_order_invariant():
    histories = _asynchronous_histories(seed=19)
    shuffled = {}
    rng = np.random.default_rng(20)
    for source, seq in histories.items():
        seq = list(seq)
        rng.shuffle(seq)
        shuffled[source] = seq

    ordered_result = calibrate_history(histories, tau_points=12)
    shuffled_result = calibrate_history(shuffled, tau_points=12)

    for source in ordered_result.sources:
        assert ordered_result.sources[source].bias == shuffled_result.sources[source].bias
        assert ordered_result.sources[source].sigma == shuffled_result.sources[source].sigma
