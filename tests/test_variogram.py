"""Characteristic-time estimator tests."""

import math

import numpy as np

from custom_components.bayesian_state_filter.core.variogram import estimate_characteristic_time


def test_characteristic_time_on_synthetic_ou_process():
    rng = np.random.default_rng(20260916)
    dt = 30.0
    tau_true = 900.0
    a = math.exp(-dt / tau_true)
    process_sigma = 0.5
    innovation_sigma = process_sigma * math.sqrt(1.0 - a * a)

    x = 0.0
    points = []
    for i in range(2400):  # 20 h
        x = a * x + rng.normal(0.0, innovation_sigma)
        measurement_sigma = 0.03
        y = x + rng.normal(0.0, measurement_sigma)
        points.append((i * dt, y, measurement_sigma**2))

    estimate = estimate_characteristic_time(points, tau_points=48)

    assert estimate is not None
    assert estimate.tau is not None
    assert estimate.identifiable
    assert 300.0 < estimate.tau < 2500.0
