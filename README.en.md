# Bayesian State Filter for Home Assistant

Version: **0.4.1**

[Русское описание](README.md)

`bayesian_state_filter` is a Home Assistant custom integration for fusing multiple asynchronous numeric sensors into one robust estimate of a shared latent scalar process.

Instead of simple averaging, the filter jointly estimates:

- process level;
- local rate of change;
- acceleration;
- jerk;
- relative source bias;
- characteristic observation variance of each source;
- uncertainty of the current estimate;
- confidence in the derivative terms of the dynamic model.

Typical use cases include temperature, atmospheric and duct pressure, humidity, background radiation, and other continuous or quasi-continuous quantities.

> This is unrelated to Home Assistant's built-in `bayesian` integration, which estimates event probability and produces a binary sensor. This project estimates a continuous numeric state.

## State model

The state is always:

```text
[x, v, a, j]
```

where `x` is level, `v` is rate, `a` is acceleration and `j` is jerk.

The model uses an integrated Wiener process. Confidence gating is applied to the mean prediction, while covariance is propagated through the full kinematic transition. This prevents weakly supported derivatives from driving the mean while preserving their observability through cross-covariances.

### Derivative confidence

For every derivative, the filter computes the posterior z-score:

```text
z_d = |d| / sigma_d
```

and base confidence:

```text
c(z) = erf(|z| / sqrt(2))
```

Since 0.4.1, higher-order derivatives are suppressed more strongly when their evidence is weak:

```text
g_k(c) = exp(-k * (1-c) / c)

w_v = g_1(c_v)
w_a = w_v * g_2(c_a)
w_j = w_a * g_3(c_j)
```

Weak acceleration and jerk therefore collapse rapidly toward zero influence, while strongly supported derivatives pass smoothly without hard thresholds or mode switches.

## Robust multi-source fusion

In multi-source mode the integration keeps the latest valid state of every source and rebuilds a common estimate at the current time whenever a source updates.

For every source it uses:

- latest measurement;
- `bias`;
- characteristic observation variance `sigma^2`;
- typical reporting interval `median_dt_s`;
- age of the latest measurement.

A slow-reporting source is not considered inaccurate merely because it reports slowly. No extra age uncertainty is added before the source exceeds its normal reporting cadence. Beyond that point the uncertainty of what the source would read now grows:

```text
R_eff = R_sensor + R_age
```

This distinction is important for battery-powered and event-driven sensors: reporting cadence and measurement accuracy are different properties.

After source values are transported to a common time, the integration performs robust variance-aware fusion. A single fused observation with its own variance then updates the state filter.

## Source calibration

### Bias

`bias` is treated as a slow metrology parameter. It is not relearned from the current process motion.

In 0.4.1 bias is estimated from the full `history_days` horizon, but only from the quietest parts of that history. Dynamic sections are excluded because sensors can respond differently to the same physical change and that transient disagreement is not a zero-offset error.

A coarse long-horizon time grid is used for bias fitting. This preserves the slow metrology information while making week-long recalibration practical.

Applying a new set of source biases preserves the current fused level, so recalibration itself does not inject an artificial step into the output state.

Available `bias_anchor` modes:

- `median` - robust relative zero gauge;
- `mean` - linear sum-to-zero gauge;
- `passport` - absolute prior anchor derived from model datasheet accuracy, with one robust vote per model family regardless of the number of identical physical sensors.

### Sigma

Source `sigma` is also treated as a slow property of the measurement channel. It is learned from long Recorder history and is not continuously relearned from the live window.

Multi-source setups use time-aligned pairwise calibration. A single-source setup uses a local-linear residual estimator so ordinary process slope is not misclassified as measurement noise.

Current process scatter does not immediately change `sigma_sensor`; runtime changes only age-related uncertainty and the resulting fused ensemble variance.

## Dynamics identification

Version 0.4.1 no longer derives process noise from one fitted time constant.

It identifies separately:

```text
q_process
level_q_process
```

from predictive residuals at multiple horizons:

```text
1, 2, 4, 8, 16, 32 natural steps
```

For long histories, several contiguous native-cadence blocks are retained across the full horizon. This preserves fast process behavior without making a week of 1 Hz data an unnecessarily large optimization problem.

`gated_timescale_s` is now a secondary diagnostic derived from the fitted process noise rather than a control parameter. The separate slow `characteristic_time_s` diagnostic may be `null` when no characteristic time is identifiable above the noise floor.

## Automatic noise detection

`noise_model: auto` treats stochastic family and quantization as independent properties.

The active stochastic family is reported as:

```text
noise_model: gaussian | poisson
```

Detected quantization is reported separately through:

```text
quantization_step
quantization_sigma
quantization_confidence
```

Quantization by itself is not evidence for Poisson noise. A discretized pressure sensor can therefore remain `gaussian` while still reporting `quantization_step: 1`.

Automatic scaled-Poisson detection uses two evidence paths:

1. When history spans a sufficiently broad range of levels, the detector tests whether local variance grows with level: `Var(X) ~ k * E[X]`. Stable positive variance scaling yields `reason: variance_scales_with_level`.
2. When the signal is nearly stationary and level span is insufficient, the detector uses a scaled-Poisson moment test. For `X = kN` with Poisson `N`, `skew(X) = sigma / mean`. The fallback requires a long positive history, statistically significant positive skewness, and agreement between observed `skew` and `sigma/mean`. A successful test yields `reason: stationary_scaled_poisson`.

If broad level coverage is available but variance does not grow with level, the detector selects Gaussian with `reason: variance_not_level_dependent`. Negative values are strong evidence against a count-like model and yield `reason: not_count_like`.

For count-derived and scaled-count signals, the Poisson-like runtime model uses variance proportional to the current level. The inferred `variance_scale` is exposed in noise-model diagnostics.

## Fast restart and checkpoints

The integration writes a checkpoint to Home Assistant Store every **30 minutes**.

The checkpoint contains `[x, v, a, j]`, covariance, process-noise parameters, source calibration, diagnostics and Recorder watermarks.

After restart:

```text
load checkpoint
-> read Recorder tail after the saved watermark
-> replay it through the normal filter path
-> switch to live mode
```

A full bootstrap is required only when the checkpoint is missing or incompatible. Version 0.4.1 uses a new checkpoint schema because process-noise semantics changed.

Heavy Recorder reads and calibration work stay outside the per-sample hot path; CPU-heavy work is dispatched through the Home Assistant executor.

## Installation

Copy:

```text
custom_components/bayesian_state_filter
```

to:

```text
/config/custom_components/bayesian_state_filter
```

Restart Home Assistant after replacing Python files.

YAML-only changes can be reloaded with:

```text
bayesian_state_filter.reload
```

## Minimal configuration

```yaml
sensor:
  - platform: bayesian_state_filter
    name: "Room temperature"
    ensemble:
      sources:
        - sensor.temperature_1
        - sensor.temperature_2
        - sensor.temperature_3
```

All sources must measure the same physical quantity in compatible units. Unit conversion is not performed automatically.

## Recommended configuration

```yaml
sensor:
  - platform: bayesian_state_filter
    name: "Room temperature"

    ensemble:
      sources:
        - sensor.temperature_1
        - sensor.temperature_2
        - sensor.temperature_3
        - sensor.temperature_4
      min_sources: 2

    bayes:
      noise_model: auto
      history_days: 7
      save_every_s: 1800
      student_nu: 4
      characteristic_refit_s: 21600
      diagnostics: compact
```

Main options:

| Option | Default | Purpose |
|---|---:|---|
| `history_days` | `7` | Recorder history used for calibration and dynamics identification |
| `save_every_s` | `1800` | Checkpoint interval |
| `student_nu` | `4` | Student-t degrees of freedom |
| `characteristic_refit_s` | `21600` | Slow characteristic-time diagnostic refit interval |
| `warmup_refit_s` | `21600` | Minimum retry interval for dynamics training |
| `bias_anchor` | `median` | `median`, `mean` or `passport` |
| `noise_model` | `auto` | `auto`, `gaussian` or `poisson` |
| `diagnostics` | `compact` | `compact`, `full`, `debug`, `verbose` |

### `passport` anchoring example

```yaml
sensor:
  - platform: bayesian_state_filter
    name: "Pressure ensemble"

    ensemble:
      sources:
        - entity_id: sensor.pressure_1
          model: bmp280_pressure
        - entity_id: sensor.pressure_2
          model: bmp280_pressure
        - entity_id: sensor.pressure_3
          model: bmp390_pressure

      models:
        bmp280_pressure:
          absolute_accuracy: 1.0
        bmp390_pressure:
          absolute_accuracy: 0.5

    bayes:
      bias_anchor: passport
```

Multiple sensors of the same model do not create multiple independent absolute-reference votes; one model family contributes one robust vote.

## Main attributes

```text
stddev
rate_per_hour
curvature_per_hour2
jerk_per_hour3

rate_weight
curvature_weight
jerk_weight

rate_z
curvature_z
jerk_z

gated_timescale_s
gated_local_rmse
gated_local_rmse_step1
gated_local_rmse_step2

characteristic_time_s
characteristic_time_confidence
characteristic_time_status
characteristic_time_identifiable
```

Per-source diagnostics under `source_health` include:

```text
model
bias
sigma
median_dt_s
outlier_rate
last_z_score
robust_weight
```

Rate, curvature and jerk are exposed in per-hour units for readability; the internal model uses seconds.

## Practical notes

- `bias` and `sigma` are slow source parameters; current physical motion should not relearn them every second.
- Under `median`/`mean` anchoring, a common systematic offset shared by all sources is not identifiable. `passport` adds a prior, not a physical reference standard.
- A source is not penalized merely for reporting slowly. Age increases uncertainty about its current value.
- Large derivative values are not meaningful on their own; inspect `*_z` and `*_weight` as well.
- When dynamics are not identifiable above noise, derivative weights should collapse toward zero and the filter naturally becomes a robust level estimator.
- `characteristic_time_s: null` with `insufficient_signal` is a valid result: the filter should not invent a process timescale that is not observable.

## Layout

```text
custom_components/
  bayesian_state_filter/
    __init__.py
    const.py
    manifest.json
    sensor.py
    services.yaml
    strings.json
    translations/
    core/
      dynamics.py
      filter.py
      gated_training.py
      noise_detection.py
      noise_models.py
      process_noise.py
      state_models.py
      training.py
      types.py
      updaters.py
      variogram.py
```

## License

GNU GPL-3.0-only.
