# Bayesian State Filter for Home Assistant

Version: **0.5.0-dev.10**

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

### Derivative plausibility

The derivatives `v/a/j` are part of the state vector itself and directly affect level prediction. Their operational weight is therefore based on how typical their magnitude is for the process, not on how strongly they differ from zero.

The filter learns local-polynomial derivatives from long Recorder history. For every derivative order it estimates a robust historical scale:

```text
sigma_hist = 1.4826 * MAD(d)
```

Training windows that cross confirmed level regime changes are excluded. Historical segmentation uses the same semantics as live recovery: roughly 6 sigma deviation, three same-direction confirmations and a compact new plateau.

The production plausibility gate is centred at zero:

```text
z = |d| / sigma_hist

w = exp(-z^2 / 2),  z < 5
w = 0,              z >= 5
```

A quiet process with derivative near zero therefore receives maximum weight, while abnormally large derivatives are suppressed.

Higher orders inherit all lower-order penalties:

```text
W_v = w_v
W_a = w_v * w_a
W_j = w_v * w_a * w_j
```

As dynamics become implausible, the effective model degrades monotonically:

```text
x-v-a-j -> x-v-a -> x-v -> x
```

Posterior derivative z-scores and covariance remain available in debug diagnostics, but they no longer act as permission for unbounded extrapolation.

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

The checkpoint contains `[x, v, a, j]`, covariance, process-noise parameters, source calibration and Recorder watermarks.

After restart the filter catches up internally before exposing its first state:

```text
load checkpoint / perform full bootstrap
-> silently replay Recorder tail
-> silently apply the freshest current source states
-> synchronize the latent state to now
-> only then publish the entity
```

A data gap caused by restart should therefore appear as increased uncertainty, not as an artificial level step.

A publication fallback is also available. If the Bayesian level diverges strongly from the direct raw/fused observation while there is evidence of broken dynamics or a developing regime change, the entity temporarily publishes the direct observation. The latent Bayesian filter continues updating in the background and resumes publication after stable reconvergence.

A full bootstrap is required only when the checkpoint is missing or incompatible. Heavy Recorder reads and calibration work stay outside the per-sample hot path; CPU-heavy work is dispatched through the Home Assistant executor.

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
      diagnostics: normal
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
| `diagnostics` | `normal` | `minimal`, `normal` or `debug`; legacy `compact/full/verbose` are accepted as aliases |

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

The default `normal` diagnostics expose a compact operational view:

```yaml
stddev: ...

filter:
  mode: tracking
  noise_model: gaussian
  noise_confidence: ...

dynamics:
  rate:
    value_per_hour: ...
    weight: ...
  curvature:
    value_per_hour2: ...
    weight: ...
  jerk:
    value_per_hour3: ...
    weight: ...
  plausibility:
    rate_sigma_per_hour: ...
    curvature_sigma_per_hour2: ...
    jerk_sigma_per_hour3: ...
    rate_mean_over_sigma: ...
    curvature_mean_over_sigma: ...
    jerk_mean_over_sigma: ...

regime:
  candidate: false
  last_jump: ...

fallback:
  active: false

sources:
  configured: ...
  active: ...
  unhealthy: ...

last_update:
  source: ...
  innovation_z: ...
  clipped: false
  dt_s: ...
```

`*_mean_over_sigma` reports the mean of the learned derivative distribution relative to its robust scale. A small non-zero value is normal for quantized and irregularly sampled processes.

`minimal` keeps only the most important operational state. `debug` additionally exposes the full laboratory diagnostics: posterior derivative uncertainty and z-scores, calibration, source health, noise-model internals, timescales, startup catch-up, regime-change and fallback details.

## Practical notes

- `bias` and `sigma` are slow source parameters; current physical motion should not relearn them every second.
- Under `median`/`mean` anchoring, a common systematic offset shared by all sources is not identifiable. `passport` adds a prior, not a physical reference standard.
- A source is not penalized merely for reporting slowly. Age increases uncertainty about its current value.
- A derivative near zero is useful evidence of a quiet process and receives maximum plausibility weight.
- Abnormally large derivatives are suppressed relative to their historical robust-MAD scale; at 5 sigma the local weight is zero.
- Confirmed level regime changes are not learned as huge rate/acceleration/jerk events: training windows that cross them are excluded.
- As dynamics become implausible, the model degrades hierarchically from `x-v-a-j` toward `x`.
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
