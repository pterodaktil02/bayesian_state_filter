"""Core regression tests for Bayesian State Filter."""

import math

from custom_components.bayesian_state_filter.core.filter import CoreFilter
from custom_components.bayesian_state_filter.core.noise_models import GaussianNoise
from custom_components.bayesian_state_filter.core.process_noise import DampedAccelerationProcessNoise
from custom_components.bayesian_state_filter.core.state_models import LevelVelocityModel
from custom_components.bayesian_state_filter.core.types import Observation
from custom_components.bayesian_state_filter.core.updaters import StudentTUpdater


def make_filter():
    tau = 600.0
    return CoreFilter(
        state_model=LevelVelocityModel(tau=tau),
        noise_model=GaussianNoise(sigma=0.05),
        updater=StudentTUpdater(nu=4.0),
        process_noise=DampedAccelerationProcessNoise(q_acc=1e-8, tau=tau),
    )


def test_student_t_downweights_large_outlier():
    filt = make_filter()
    t = 0.0
    filt.step(Observation(t=t, z=20.0, variance=0.05**2))

    for _ in range(20):
        t += 10.0
        out = filt.step(Observation(t=t, z=20.0, variance=0.05**2))
        assert out.diag["weight"] == 1.0

    before = float(filt.x[0])
    t += 10.0
    outlier = filt.step(Observation(t=t, z=25.0, variance=0.05**2))

    assert outlier.diag["weight"] < 0.1
    assert abs(float(filt.x[0]) - before) < 1.0


def test_posterior_uncertainty_stays_finite_and_positive():
    filt = make_filter()
    t = 0.0
    for i in range(100):
        t += 10.0
        value = 20.0 + 0.1 * math.sin(i / 10.0)
        out = filt.step(Observation(t=t, z=value, variance=0.04**2))
        assert math.isfinite(out.y_mean)
        assert out.y_var > 0.0
        assert math.isfinite(out.y_var)


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

    assert math.isclose(out.dt, 1e-3, rel_tol=0.0, abs_tol=1e-12)
    assert filt.t_last == 100.0
