# Changelog

## 0.4.0 - 2026-09-24

### Performance
- Removed the O(history) source recalibration pass from the per-sample hot path.
- Added a cheap O(1) innovation/drift monitor and adaptive heavy-refit scheduling.
- Stable sources now refit at roughly half the calibration window; suspicious or unstable sources refit more often.
- Drift detection is relative to each source's established outlier baseline, avoiding false `watch` mode for naturally noisier sensors.
- Bounded retained level history and reduced hot-path `source_health` payload to lower Home Assistant state/serialization overhead.

### Persistence and diagnostics
- Persisted adaptive calibration scheduler state in checkpoints.
- Restoring an older compatible checkpoint seeds the refit timestamp from the checkpoint time instead of forcing an immediate heavy refit.
- Added CPU/background diagnostics for calibration, warmup, characteristic-time fitting and checkpoint saves.

### Calibration
- Added configurable bias anchoring: `median`, `mean`, and `passport`.
- `passport` uses one robust vote per sensor model weighted by configured absolute accuracy, avoiding duplicate sensors of one model counting as independent absolute references.
- Checkpoint compatibility now includes source/model mapping and bias-anchor semantics.

## 0.3.2 - 2026-09-21

- Fixed the unidentifiable common bias mode by enforcing `median(bias_i) = 0`.
- Applied the same gauge at startup and during live source calibration.
- Kept live residual history in the same gauge after re-anchoring.
- Added migration for 0.3.1 checkpoints: stored level and level-history are shifted consistently instead of forcing a full retrain.
- Added regression tests preventing common-mode bias drift.

## 0.3.1 - 2026-09-20

### Added
- Production four-dimensional `[x, v, a, j]` state model.
- Posterior-confidence derivative gating with hierarchical velocity, acceleration and jerk weights.
- Prototype-equivalent Student-t updater and confidence-gated dynamics training from `bayesian_trend_filter 0.6.3`.
- Dedicated natural-cadence fused history for dynamics identification.
- 1-step and 2-step local RMSE diagnostics.
- 30-minute checkpoints with Recorder-tail replay after restart.
- Russian and English documentation for the new model and restart path.

### Fixed
- Corrected the initial v0.3.0 merge so valid Student-t inliers are no longer capped at weight 1.
- Mean and covariance now use the same confidence-gated transition.
- Dynamics training and startup replay use the same temporal representation.
- Checkpoint schema/store key changed so incompatible v0.3.0 state cannot be restored.

## 0.2.1.7

- Added `bayesian_state_filter.reload` for YAML configuration reloads without a full Home Assistant restart.
- Reload teardown now unregisters state listeners, and entity initialization works both during normal startup and when a YAML platform is recreated while Home Assistant is already running.
- Added integration/service translations so the YAML reload UI shows **Bayesian State Filter** instead of the raw domain name.
- Made startup and live pairwise sigma calibration continuous: Recorder-derived pair evidence is retained compactly and ages out over the same rolling calibration window while live evidence replaces it.
- Prevented a short dense live burst after restart/reload from immediately replacing a multi-day startup sigma estimate.
- Expanded `source_health` with startup/live calibration evidence, `startup_sigma`, and `calibration_window_s`.
- Kept the Bayesian state model, Student-t robust update, characteristic-time estimator and persistence key compatible with 0.2.1.6.

## 0.2.1.6

Review hardening before the first public push:

- declare the NumPy runtime requirement in `manifest.json`;
- name key numerical/freshness constants in `const.py` without changing their values;
- add regression tests for known-sigma pairwise calibration, history order invariance and live time-ordering/backdating policy;
- document roadmap and deferred structural work;

- Added per-source live diagnostics in `source_health`: raw/corrected value, innovation, z-score and robust weight.
- Added explicit `outliers` counter next to `outlier_rate`.
- No change to Bayesian filter math relative to 0.2.1.5.

## 0.2.1.5

- Split configured noise selection policy (`noise_model_mode`) from the actually active noise family (`noise_model`).
- Added consistent last-update diagnostics: `last_source`, `measurement_sigma`, `measurement_variance`, `innovation`, `z_score`, `robust_weight`, `update_dt_s` now refer to the same observation.
- Kept the public interface ready for future automatic Poisson detection. In this release `auto` remains conservative and selects Gaussian noise.

## 0.2.1.4

- Replaced source-local high-order differencing with time-aligned pairwise source calibration.
- Per-source observation variances are estimated from equations of the form `Var(i-j) ~= sigma_i^2 + sigma_j^2` over close-in-time sensor pairs.
- Added a non-zero identifiability floor to prevent a source variance from collapsing spuriously to zero.
- Added `calibration_pairs`, `calibration_samples` and `calibration_span_s` diagnostics.

## 0.2.1.x base

- Multi-source source-aware Bayesian filtering.
- Always-on Student-t robust update.
- Recorder pre-training and persistence.
- Independent level-process characteristic-time estimation using a robust temporal variogram.
- Separate predictive damped-velocity model with learned `velocity_tau` and process noise.
