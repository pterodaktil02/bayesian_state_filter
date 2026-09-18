# Bayesian State Filter for Home Assistant

Version: **0.2.1.7**

[Русский README](README.md)

> **Do not confuse this project with Home Assistant's built-in [`bayesian`](https://www.home-assistant.io/integrations/bayesian/) integration.** The built-in integration estimates the probability of an event from a set of observations and exposes a **binary** `on/off` sensor. Even when numeric observations are used, its result is still a `binary_sensor`. `bayesian_state_filter` is specifically intended for **numeric sensors and continuous-valued quantities**: it fuses asynchronous measurements of the same physical quantity and produces a continuous numeric estimate, for example temperature, humidity or pressure.

`bayesian_state_filter` is a Home Assistant custom sensor platform that fuses multiple asynchronous numeric sensors into one robust Bayesian estimate of a shared latent state.

It learns relative source bias, effective observation uncertainty, source cadence, predictive local dynamics and a separate level-process characteristic time. Live updates use an always-on Student-t robust updater, so a temporarily divergent sensor is downweighted rather than blindly averaged into the result.

> All sources in one filter instance must represent the same physical quantity in compatible units. The integration does not perform unit conversion.

## Installation

Copy:

```text
custom_components/bayesian_state_filter
```

to:

```text
/config/custom_components/bayesian_state_filter
```

After installing or updating the integration files, perform a full Home Assistant restart.

Starting with 0.2.1.7, later YAML edits can be applied without restarting Home Assistant:
call `bayesian_state_filter.reload` or use **Settings -> System -> YAML -> Bayesian State Filter**.
Reload re-reads YAML and recreates this platform's sensor entities; listeners owned by removed
entities are unsubscribed during teardown.

Version 0.2.1.7 uses YAML sensor-platform configuration; there is no Config Flow yet.

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
      save_every_s: 60
      tau_points: 16
      characteristic_refit_s: 1800
      student_nu: 4
      diagnostics: compact
```

# Configuration reference

## Top-level keys

| Key | Type | Default | Values / meaning |
|---|---|---|---|
| `platform` | string | — | Must be `bayesian_state_filter`. |
| `name` | string | `Bayesian Sensor` | Entity name. Use distinct names for multiple filter instances. |
| `ensemble` | mapping | `{}` | Source list and minimum fresh-source count. |
| `bayes` | mapping | `{}` | History, dynamics, robust update and diagnostics settings. |

## `ensemble.sources`

List of numeric Home Assistant entities. Unknown, unavailable, non-numeric and non-finite states are ignored.

The final entity inherits unit/device/state metadata from the first suitable live source. Units are not converted.

## `ensemble.min_sources`

- Type: integer
- Default: `1`
- Minimum: `1`

Minimum number of fresh sources required for a live Bayesian update.

A source is considered fresh approximately for:

```text
max(3 * source_median_dt,
    0.20 * characteristic_time_or_velocity_tau,
    300 s)
```

## `bayes.noise_model`

- Type: string
- Default: `auto`
- Accepted: `auto`, `gaussian`, `poisson`

`auto` is intentionally conservative in 0.2.1.7 and currently selects Gaussian. The public interface is already split into `noise_model_mode` and active `noise_model` so future Poisson auto-detection will not require YAML migration.

Gaussian:

```text
R_i = sigma_i^2
```

Poisson family:

```text
R_i(y) = k_i * |y|
k_i ~= sigma_i^2 / typical_abs_level_i
```

The effective scale is learned internally; users do not configure a separate "scaled Poisson" type.

## `bayes.history_days`

- Type: float
- Default: `7.0` days
- Minimum: `0.1` day

Recorder history span requested for each source at startup. Used for relative bias, pairwise uncertainty calibration, filter pretraining, characteristic-time estimation and predictive dynamics learning.

## `bayes.save_every_s`

- Type: float
- Default: `60`
- Minimum: `5`

Minimum interval between persistence writes to Home Assistant Store.

## `bayes.tau_points`

- Type: integer
- Default: `16`
- Minimum: `8`

Number of logarithmic candidates in the internal predictive velocity-memory tau grid. The level-process characteristic-time fit uses a separate, denser grid.

## `bayes.tau_min_s`

Optional positive float. Lower bound of **predictive velocity memory**, not the public level-process characteristic time.

Automatic lower bound is approximately:

```text
max(4 * fused_grid_step,
    4 * fastest_source_median_dt,
    10 s)
```

## `bayes.tau_max_s`

Optional positive float. Upper bound of predictive velocity memory.

Automatic upper bound is approximately:

```text
max(8 * auto_tau_min,
    history_span / 2)
```

## `bayes.characteristic_tau_min_s`

Optional positive float. Lower search bound for the level-process characteristic time.

Backward-compatible alias: `characteristic_time_min_s`.

## `bayes.characteristic_tau_max_s`

Optional positive float. Upper search bound for the level-process characteristic time.

Backward-compatible alias: `characteristic_time_max_s`.

## `bayes.characteristic_refit_s`

- Type: float
- Default: `1800`
- Minimum: `300`

Minimum interval between online characteristic-time refits.

## `bayes.forget_time_s`

Optional positive float controlling forgetting in the **predictive dynamics bank**.

Default:

```text
max(3 days, 8 * tau_max)
```

This is not the Recorder history window and not the pairwise source-calibration window.

## `bayes.student_nu`

- Type: float
- Default: `4.0`
- Minimum: `1.01`

Student-t degrees of freedom. Lower values have heavier tails and downweight large outliers more aggressively. Larger values approach Gaussian updating.

Robust updating is always enabled in 0.2.1.7; there is no `robust_enable` switch.

## `bayes.diagnostics`

- Default: `compact`
- `compact`: recommended public attribute set
- `full`: laboratory/internal diagnostics
- `debug`, `verbose`: aliases for `full`

Any other value behaves like compact. This setting changes presentation only, not estimator behavior.

# Startup training

At startup the integration:

1. reads Recorder history per source;
2. estimates source cadence;
3. estimates relative source bias;
4. obtains a preliminary level-process time scale;
5. calibrates per-source uncertainty from close-in-time sensor pairs;
6. solves non-negative source variances from `Var(i-j) ~= sigma_i^2 + sigma_j^2`;
7. rebuilds fused history with calibrated source variances;
8. estimates final level-process characteristic time;
9. trains the predictive damped-velocity model;
10. replays fused history through the Bayesian filter.

If Recorder history is unavailable, persisted state is used when available; otherwise the filter bootstraps from live source states and learns as data accumulates.

# Source calibration semantics

`bias` is relative, not absolute metrological calibration:

```text
corrected = raw - bias
```

`sigma` is effective observation uncertainty relative to the shared latent state. It may include sensor noise, quantization, local microclimate, enclosure lag, small alignment errors and other residual source-specific disagreement. It is not manufacturer accuracy.

# Characteristic time

The level-process characteristic time is estimated from a robust temporal semivariogram:

```text
gamma(h) = nugget + process_variance * (1 - exp(-h / tau))
```

Possible statuses:

- `identified`
- `insufficient_signal`
- `below_resolution`
- `longer_than_history`
- `uncertain`
- `unavailable`

The characteristic time is observational. In a closed control loop it describes the observed room + controller + actuator + forcing system, not a pure plant constant.

# Compact entity attributes

Main diagnostics:

- `stddev`
- `filter_mode`
- `noise_model_mode`
- `noise_model`
- `noise_model_params`
- `noise_variance_source`
- `characteristic_time_s`
- `characteristic_time_p10_s`
- `characteristic_time_p90_s`
- `characteristic_time_confidence`
- `characteristic_time_status`
- `characteristic_time_identifiable`
- `characteristic_time_boundary_limited`
- `source_health`

Last-update fields are internally consistent and all refer to the same observation:

- `last_source`
- `measurement_sigma`
- `measurement_variance`
- `innovation`
- `z_score`
- `robust_weight`
- `update_dt_s`

`z_score` is the absolute innovation divided by its expected standard deviation, including both predicted state uncertainty and measurement variance. It is therefore a normalized innovation of the observation model, not simply "sensor error in sigmas".

# `source_health`

Per source:

- `bias`: relative offset; correction is `raw - bias`.
- `sigma`: effective observation standard deviation.
- `median_dt_s`: typical update cadence.
- `outliers`: strong live outliers since current startup.
- `outlier_rate`: `outliers / live_updates`; always interpret with the denominator.
- `history_samples`: raw Recorder samples used at startup.
- `calibration_samples`: current effective pairwise evidence: the still-active startup fraction plus live evidence inside the rolling calibration window. It is not necessarily a count of unique raw source samples.
- `calibration_span_s`: current time span of the working pairwise evidence.
- `calibration_pairs`: number of peer sources currently contributing usable pairwise equations.
- `startup_calibration_samples`: historical pairwise evidence used by startup calibration; immutable after startup.
- `startup_calibration_span_s`: Recorder time span covered by startup pairwise evidence.
- `startup_calibration_pairs`: peer-source count in the startup calibration.
- `startup_sigma`: source sigma immediately after startup calibration.
- `calibration_window_s`: rolling sigma-calibration horizon over which startup evidence ages out.
- `live_updates`: live updates processed since startup.
- `live_calibration_samples`: actual live pairwise residuals accumulated since startup.
- `live_calibration_span_s`: live pairwise evidence time span.
- `live_calibration_pairs`: peer sources already contributing usable live pairs.
- `last_raw_value`: latest raw source value.
- `last_corrected_value`: value after the bias used at that exact update.
- `last_innovation`: latest source innovation.
- `last_z_score`: latest source z-score.
- `last_robust_weight`: latest Student-t weight.

In 0.2.1.7 startup and live sigma calibration form one rolling evidence window.
The first live recalculation no longer discards Recorder evidence. Historical
pairwise evidence ages out over `calibration_window_s` while live pairs fill the
same horizon, so a dense short burst from a fast source cannot replace a
multi-day startup estimate in a few seconds. Raw historical pair residuals are
not kept in memory; startup evidence is retained compactly as pair variance,
pair count and time span.

`startup_calibration_samples` may exceed `history_samples` because one source
sample can contribute to separate pairwise equations with multiple peer sensors.

# Full diagnostics

With `diagnostics: full`, additional research fields include:

- `variance`
- `velocity`
- `velocity_tau_s`
- `process_noise_q`
- `velocity_tau_p10_s`
- `velocity_tau_p90_s`
- `velocity_tau_confidence`
- `velocity_tau_edge_mass`
- `velocity_tau_boundary_limited`
- `characteristic_time_edge_mass`
- `characteristic_nugget_variance`
- `characteristic_process_variance`
- `characteristic_signal_fraction`
- `characteristic_fit_error`
- `characteristic_lag_count`
- `characteristic_pair_count`
- `characteristic_min_lag_s`
- `characteristic_max_lag_s`
- `noise_velocity`
- `innovation_var`
- `p_value`
- `effective_innovation_variance`

`velocity_tau_s` is the predictive memory of the local slope. It is deliberately separate from `characteristic_time_s`.

# Runtime dependency

The integration imports NumPy throughout the mathematical core and explicitly declares `numpy>=1.26.0,<3.0.0` in `manifest.json`. If Home Assistant already has a compatible NumPy version, that installed version satisfies the requirement.

# Time-ordering policy

Live observations must be non-decreasing in time. A genuinely backdated observation is rejected by the core filter instead of silently rewinding state. Equal timestamps are accepted with a minimum internal `dt = 1e-3 s` to keep covariance propagation numerically well-defined. Recorder history is cleaned, sorted by timestamp and exact duplicate timestamps are deduplicated (last value wins) before training.

# Known limitations and roadmap

The following improvements are intentionally deferred from 0.2.1.7 so the already-tested estimator is not mixed with a large behavioral refactor:

- Config Flow / Options Flow while retaining YAML compatibility;
- real `noise_model: auto` detection for Gaussian vs Poisson observation noise;
- splitting the large `sensor.py` into history, diagnostics, persistence and runtime-calibration modules without behavior changes;
- Home Assistant integration tests with a mocked state machine / Recorder;
- explicit fresh/stale source-count diagnostics.

Internal numerical constants such as the minimum Student-t weight, PSD floor and freshness policy are deliberately not user-facing YAML knobs. Key values are named in `const.py` so the policy is auditable without turning the integration into a tuning panel.

# Legacy notes

`robust_enable` is not a documented option in 0.2.1.7. Student-t robustness is always active.

`save_every_s` belongs inside `bayes:`.

An empty `bayes:` block is valid and uses defaults.

# Development tests

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements-dev.txt
pytest -q
```

The test suite covers robust outlier handling, posterior uncertainty, bias calibration, pairwise source variance calibration, recovery of known synthetic source sigmas, history order invariance, live time-ordering/backdating policy, startup repeatability, correlated fast-source noise and synthetic OU characteristic-time estimation.

# Authors and contributions

**Evgeny V. Polupanov** — project originator, requirements author, and practical development lead.

**Grigory P. Timofeev** — engineering and software co-author.

See [`AUTHORS.md`](AUTHORS.md) for the detailed contribution breakdown.

This project is licensed under the **GNU General Public License v3.0 (GPL-3.0-only)**. See [`LICENSE`](LICENSE) for the full text.
