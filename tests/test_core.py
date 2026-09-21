"""Core regression tests for Bayesian State Filter 0.3.x."""

import math

import numpy as np

from custom_components.bayesian_state_filter.core.filter import CoreFilter
from custom_components.bayesian_state_filter.core.noise_models import GaussianNoise
from custom_components.bayesian_state_filter.core.process_noise import IntegratedWienerProcessNoise
from custom_components.bayesian_state_filter.core.state_models import AdaptivePolynomialStateModel
from custom_components.bayesian_state_filter.core.types import Observation
from custom_components.bayesian_state_filter.core.updaters import StudentTUpdater


def make_filter(q=1e-12, timescale=600.0):
    return CoreFilter(
        state_model=AdaptivePolynomialStateModel(3),
        noise_model=GaussianNoise(sigma=0.05),
        updater=StudentTUpdater(nu=4.0, min_weight=0.05),
        process_noise=IntegratedWienerProcessNoise(order=3, q=q),
        prior_timescale_s=timescale,
    )


def test_student_t_inlier_can_exceed_unit_weight_and_outlier_is_downweighted():
    filt = make_filter(q=0.0)
    filt.reset(20.0, t=0.0, variance=0.05**2)

    inlier = filt.step(Observation(t=10.0, z=20.0, variance=0.05**2))
    assert 1.0 < inlier.diag["weight"] <= 1.25 + 1e-12

    before = float(filt.x[0])
    outlier = filt.step(Observation(t=20.0, z=25.0, variance=0.05**2))
    assert outlier.diag["weight"] < 0.1
    assert abs(float(filt.x[0]) - before) < 1.0


def test_full_state_posterior_stays_finite_symmetric_and_positive_diagonal():
    filt = make_filter(q=1e-15, timescale=120.0)
    t = 0.0
    for i in range(300):
        t += 2.0
        value = 20.0 + 0.1 * math.sin(i / 20.0)
        out = filt.step(Observation(t=t, z=value, variance=0.04**2))
        assert math.isfinite(out.y_mean)
        assert math.isfinite(out.y_var)
        assert out.y_var > 0.0
        assert np.all(np.isfinite(filt.x))
        assert np.all(np.isfinite(filt.P))
        assert np.allclose(filt.P, filt.P.T, rtol=0.0, atol=1e-12)
        assert np.all(np.diag(filt.P) > 0.0)


def test_derivative_weights_are_hierarchical():
    model = AdaptivePolynomialStateModel(3)
    x = np.array([1000.0, 0.2, 0.01, 0.001], dtype=float)
    P = np.diag([1.0, 0.01, 0.0004, 0.0001])
    w = model.effective_weights(x, P, 1.0)
    assert 1.0 >= w[1] >= w[2] >= w[3] >= 0.0


def test_gated_mean_and_covariance_use_same_transition():
    model = AdaptivePolynomialStateModel(3)
    x = np.array([1000.0, 0.2, -0.03, 0.01], dtype=float)
    P = np.array([
        [0.04, 0.002, 0.0, 0.0],
        [0.002, 0.01, 0.001, 0.0],
        [0.0, 0.001, 0.0025, 0.0001],
        [0.0, 0.0, 0.0001, 0.001],
    ], dtype=float)
    Q = IntegratedWienerProcessNoise(order=3, q=1e-12).Q(2.5)
    Fw, _ = model.weighted_transition(x, P, 2.5)
    xp, Pp = model.predict(x, P, 2.5, Q)
    assert np.allclose(xp, Fw @ x)
    assert np.allclose(Pp, Fw @ P @ Fw.T + Q)


def test_backdated_observation_is_rejected():
    filt = make_filter()
    filt.step(Observation(t=100.0, z=20.0, variance=0.05**2))
    try:
        filt.step(Observation(t=99.0, z=20.1, variance=0.05**2))
    except ValueError as exc:
        assert "time ordered" in str(exc)
    else:
        raise AssertionError("backdated observation must be rejected")


def test_equal_timestamp_uses_minimum_positive_dt():
    filt = make_filter()
    filt.step(Observation(t=100.0, z=20.0, variance=0.05**2))
    out = filt.step(Observation(t=100.0, z=20.01, variance=0.05**2))
    assert math.isclose(out.dt, 1e-6, rel_tol=0.0, abs_tol=1e-15)
    assert filt.t_last == 100.0


def test_state_checkpoint_roundtrip_preserves_full_4d_state():
    first = make_filter(q=3e-14, timescale=45.0)
    first.reset(1000.0, t=0.0, variance=0.02**2)
    for i in range(1, 50):
        first.step(Observation(t=float(i), z=1000.0 + 0.001 * i, variance=0.02**2))

    saved = first.dump_state()
    second = make_filter(q=0.0, timescale=999.0)
    assert second.load_state(saved)
    assert np.array_equal(first.x, second.x)
    assert np.array_equal(first.P, second.P)
    assert first.t_last == second.t_last
    assert first.q_process == second.q_process
    assert first.tau == second.tau
