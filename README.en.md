# Bayesian State Filter for Home Assistant

Version: **0.4.0**

[Русское описание](README.md)

`bayesian_state_filter` is a Home Assistant custom integration for fusing multiple asynchronous numeric sensors into one robust estimate of a shared latent scalar process.

Instead of simple averaging, it jointly estimates:

- process level;
- local rate of change;
- acceleration;
- jerk;
- relative bias of each source;
- effective observation standard deviation (`sigma`) of each source;
- confidence in each incoming measurement;
- confidence in the derivative terms of the dynamic model.

Typical use cases include temperature, pressure, humidity, background radiation, and other continuous or quasi-continuous quantities measured by several sources in compatible units.

> This is unrelated to Home Assistant's built-in `bayesian` integration, which estimates event probability and produces a binary sensor. This project estimates a continuous numeric state.

## State model

The state is always:

```text
[x, v, a, j]
```

where `x` is level, `v` is rate, `a` is acceleration and `j` is jerk.

The model uses an integrated Wiener process driven by snap noise. The same confidence-gated transition matrix is applied to both the state mean and covariance. This prevents poorly observed hidden derivatives from destabilizing the covariance through ungated cross-couplings.

### Derivative confidence

For every derivative, the filter computes a posterior z-score:

```text
z_d = |d| / sigma_d
```

It is mapped to posterior confidence as:

```text
c(z) = erf(|z| / sqrt(2))
```

Effective weights are hierarchical:

```text
w_v = c_v
w_a = c_v * c_a
w_j = c_v * c_a * c_j
```

Therefore:

```text
1 >= w_v >= w_a >= w_j >= 0
```

If rate is not confidently observed, acceleration and jerk cannot have stronger influence on the predicted trajectory.

### Robust multi-source fusion

Each incoming measurement updates the common state independently; sources are not averaged before the Bayesian update.

For every source, the integration estimates:

- relative `bias`;
- effective observation standard deviation `sigma`;
- typical update interval;
- innovation (measurement residual);
- innovation z-score;
- Student-t robust weight;
- outlier statistics.

The Student-t updater may assign a weight slightly above 1 to a well-aligned inlier and smoothly downweight outliers.

### Dynamics identification

`q/timescale` are identified from Recorder history on the process's natural time grid. Validation uses 1-2 natural process steps instead of an arbitrary long forecast horizon.

RMSE is a diagnostic of the fitted dynamics, not a definition of derivative confidence. Confidence is derived only from the posterior state and covariance.

### Fast restart and checkpoints

The integration writes a checkpoint to Home Assistant Store every **30 minutes**.

The checkpoint contains:

- `[x, v, a, j]`;
- covariance `P`;
- learned `q/timescale`;
- source calibration;
- characteristic-time diagnostics;
- Recorder watermarks;
- checkpoint schema version.

After restart:

```text
load checkpoint
-> read Recorder tail after the saved watermark
-> replay the tail through the normal filter path
-> switch to live mode
```

A full multi-day bootstrap is only required when the checkpoint is missing, corrupted or schema-incompatible. Online source calibration is not recomputed on every sample; expensive refits are scheduled adaptively from drift-monitor state.

## Online source calibration

Each observation performs the normal filter update plus a cheap drift monitor. Re-estimation of source `bias/sigma` over the historical window runs separately and adaptively: infrequently for stable sources and more often when drift is detected.

Drift is evaluated relative to each source's own baseline rather than a single global threshold. Scheduler state is persisted in the checkpoint.

Three common-bias gauges are available through `bias_anchor`:

- `median` - robust relative zero gauge;
- `mean` - linear sum-to-zero gauge;
- `passport` - an absolute prior-based anchor derived from model datasheet accuracy, with one robust vote per model family regardless of the number of identical physical sensors.

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
| `history_days` | `7` | Recorder history used for startup calibration and dynamics identification |
| `save_every_s` | `1800` | Checkpoint interval |
| `student_nu` | `4` | Student-t degrees of freedom |
| `characteristic_refit_s` | `21600` | Refitting interval for the slow level characteristic time |
| `warmup_refit_s` | `21600` | Minimum retry interval for dynamics warmup while it is still unidentified |
| `bias_anchor` | `median` | Common-bias anchor: `median`, `mean` or `passport` |
| `noise_model` | `auto` | `auto`, `gaussian` or `poisson` |
| `diagnostics` | `compact` | `compact`, `full`, `debug`, `verbose` |

### `passport` anchoring example

When `bias_anchor: passport` is used, sources that participate in the absolute anchor must declare a model, and that model must provide `absolute_accuracy`:

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

Multiple sensors of the same model do not create multiple independent absolute-reference votes; the model family contributes one robust vote.

## Main attributes

The resulting sensor exposes, among others:

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
```

Rate, curvature, and jerk are exposed in per-hour units for readability, while the internal state uses per-second time units.

Per-source diagnostics are available under `source_health`:

```text
model
bias
sigma
median_dt_s
outlier_rate
last_z_score
robust_weight
```

`characteristic_time_s` remains a separate slow-timescale estimate of the level process. It is not expected to match `gated_timescale_s`, which belongs to the local state-space model.

## Practical notes

- With `median` or `mean` anchoring, `bias` is relative: a systematic error shared by all sources is not identifiable from the ensemble alone. `passport` adds an absolute prior from model datasheet accuracy, but it is still a prior, not a physical reference standard.
- `sigma` is the effective observation standard deviation with respect to the latent state, not the sensor datasheet accuracy.
- A high `robust_weight` indicates an inlier; a low one indicates an outlier or temporary disagreement with the common process.
- Large `curvature_per_hour2` or `jerk_per_hour3` values alone are not necessarily problematic because local derivatives are rescaled from seconds to hours. Inspect them together with derivative weights and local RMSE.
- When dynamics are not identifiable above the noise floor, derivative weights should collapse toward zero and the model naturally behaves as a level estimator.

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
      filter.py
      gated_training.py
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
