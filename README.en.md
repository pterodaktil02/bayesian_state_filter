# Bayesian State Filter for Home Assistant

Version: **0.5.0-dev.38**

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

### Derivatives and the independent edge witness

The derivatives `v/a/j` are part of the state vector and directly affect level prediction. Starting with 0.5.0-dev.38 their operational weight is driven primarily by agreement between the Bayesian state and an independent causal witness built only from direct raw/fused observations.

Long Recorder history still provides robust historical scales

```text
sigma_hist = 1.4826 * MAD(d)
```

but those scales now describe novelty more than correctness. A real new process may legitimately lie many historical sigmas outside the training distribution and still be accepted when direct observations confirm it.

The edge witness is a robust multiscale causal local-polynomial estimate:

```text
short:   >= 25 points and >= 0.5 * process_timescale
medium:  >= 40 points and >= 1.0 * process_timescale
long:    >= 55 points and >= 2.0 * process_timescale
```

Rate and curvature use consistent estimates across multiple scales. Jerk requires two cubic estimates, medium and long. Every fit window must satisfy both the point-count and physical-time constraints.

When a witness is valid, each derivative order uses

```text
z_edge = |d_bayes - d_edge| / sigma_edge
```

with the local coupling weight

```text
w = 1,                         z_edge <= 1
w = exp(-0.5 * (z_edge-1)^2),  1 < z_edge < 5
w = 0,                         z_edge >= 5
```

Agreement within one witness sigma gets full trust. Beyond that the weight decays smoothly and reaches zero at 5 sigma. A large derivative is not penalized merely for being historically unusual when the independent witness confirms it.

Higher orders remain hierarchical:

```text
W_v = w_v
W_a = w_v * w_a
W_j = w_v * w_a * w_j
```

so the effective model still degrades monotonically:

```text
x-v-a-j -> x-v-a -> x-v -> x
```

If no quality-passing edge witness is available for an order, the historical plausibility gate is used as a fallback. This keeps startup and post-gap behavior conservative until a new causal segment becomes observable.

### Derivative recovery

Destructive derivative recovery is triggered by persistent Bayes-to-edge disagreement rather than by one unusual sample:

```text
ENTER grace:  z_edge > 5
CANCEL grace: z_edge < 3
```

Recovery requires at least five fresh edge updates and at least one `process_timescale_s` since the candidate began. When it fires, the first divergent derivative and the tail above it are reconditioned from witness hints.

Historical 5-sigma recovery remains a fallback when no independent witness is available.

### Observation gaps

The edge witness is causal and must never fit across an interval in which the process was unobserved. Therefore

```text
gap > process_timescale_s
=> start a new edge segment
```

Old edge history is excluded from the fit, unfinished recovery grace is cleared, external derivative weights are removed and historical fallback is used temporarily. The Bayesian latent state itself is not reset.

Edge history is persisted independently in the checkpoint. Older or insufficient checkpoints are backfilled once from recent direct/fused Recorder observations.
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

The checkpoint contains `[x, v, a, j]`, covariance, process-noise parameters, source calibration, Recorder watermarks and the independent `edge_history`.

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
    rate_edge_z: ...
    curvature_edge_z: ...
    jerk_edge_z: ...
    rate_edge_local_weight: ...
    curvature_edge_local_weight: ...
    jerk_edge_local_weight: ...

  recovery:
    count: ...
    process_timescale_s: ...
    edge_divergence:
      enter_sigma: 5
      cancel_sigma: 3

  edge_witness:
    process_timescale_s: ...
    gap_limit_s: ...
    segment_points: ...
    segment_span_s: ...

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
- Historical robust-MAD scale describes primarily how unusual current dynamics are, not whether they are correct.
- With a valid edge witness, `v/a/j` weights are driven by Bayes-to-observation agreement: full weight through 1 sigma, then Gaussian decay to zero at 5 sigma.
- Historical plausibility is used as a fallback while the edge witness is unavailable.
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
      edge_derivatives.py
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
