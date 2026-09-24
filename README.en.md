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
- effective observation sigma of each source;
- confidence in each incoming measurement;
- confidence in the derivative terms of the dynamic model.

Typical use cases include temperature, pressure, humidity, radiation background and other continuous or quasi-continuous quantities measured by several sources in compatible units.

> This is unrelated to Home Assistant's built-in `bayesian` integration, which estimates event probability and produces a binary sensor. This project estimates a continuous numeric state.

## What changed in 0.4.0

The main 0.4.0 change is a hot-path performance fix in online source
calibration. Previously, every incoming observation could trigger a
`bias/sigma` re-estimation over a growing history window. With fast sources
this turned an almost unchanged metrology estimate into an O(history)
operation on every sample and could eventually consume the Home Assistant
event-loop CPU.

Each sample now performs only the normal filter update plus a cheap O(1)
drift monitor. The full O(history) calibration pass is scheduled adaptively:

```text
stable   -> approximately once per calibration_window / 2
watch    -> more often when drift becomes noticeable
unstable -> more often again under clear degradation
```

Drift is evaluated relative to each source's own established outlier baseline,
so a sensor that normally sits around a 4% outlier rate is not permanently
classified as suspicious merely for exceeding a global fixed threshold.

Adaptive scheduler state is persisted in the checkpoint. A normal restart from
a compatible checkpoint neither repeats the full multi-day bootstrap nor
forces an immediate heavy refit.

`source_health` was also reduced to compact runtime fields so a large nested
diagnostic payload is not serialized on every sensor update. Detailed CPU and
background-work diagnostics are exposed separately under `cpu_diag_*`.

0.4.0 also adds `bias_anchor`:

- `median` - the previous robust zero gauge;
- `mean` - a linear sum-to-zero gauge;
- `passport` - an absolute anchor from model datasheet accuracy, with one
  robust vote per model family regardless of how many identical physical
  sensors are present.

## What changed in 0.3.2

Fixed the unidentifiable common mode of source `bias` calibration. Pairwise
calibration determines only bias differences, so without an explicit gauge all
source biases could drift together by the same constant and move the absolute
ensemble level without changing any cross-sensor residual.

Startup and live calibration now enforce:

```text
median(bias_i) = 0
```

This matches the robust median fusion used by the filter: relative source
corrections are preserved, while the common bias zero point can no longer
wander. Existing 0.3.1 checkpoints are migrated without a full retrain: the
common bias offset is removed and the stored level plus level-history are
shifted by the same amount.

## What changed in 0.3.1

Version 0.3 ports the four-dimensional model validated in the experimental `bayesian_trend_filter 0.6.3` into the production filter.

The state is always:

```text
[x, v, a, j]
```

where `x` is level, `v` is rate, `a` is acceleration and `j` is jerk.

The model uses an integrated Wiener process driven by snap noise. The same confidence-gated transition matrix is used for both the state mean and covariance. This prevents poorly observed hidden derivatives from destabilizing the covariance through ungated cross-couplings.

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

For every source the integration estimates:

- relative `bias`;
- effective `sigma`;
- typical update cadence;
- innovation;
- innovation z-score;
- Student-t robust weight;
- outlier statistics.

The Student-t updater may assign a weight slightly above 1 to a well-aligned inlier and smoothly downweights outliers.

### Dynamics identification

`q/timescale` are identified from Recorder history on the natural process cadence. Validation uses 1-2 natural process steps instead of an arbitrary long forecast horizon.

RMSE is a diagnostic of the fitted dynamics, not a definition of derivative confidence. Confidence is derived only from the posterior state and covariance.

### Fast restart

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

A full multi-day bootstrap is only required when the checkpoint is missing, corrupted or schema-incompatible.

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
      characteristic_refit_s: 1800
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
| `bias_anchor` | `median` | Common-bias gauge: `median`, `mean` or `passport` |
| `noise_model` | `auto` | `auto`, `gaussian` or `poisson` |
| `diagnostics` | `compact` | `compact`, `full`, `debug`, `verbose` |

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

Rate, curvature and jerk are exposed in per-hour units for readability, while the internal state uses per-second time units.

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

- `bias` is relative; without an external reference the filter cannot determine a systematic error shared by all sources.
- `sigma` is the effective observation error with respect to the latent state, not the sensor datasheet accuracy.
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
