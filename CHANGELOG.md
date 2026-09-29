# Changelog

## 0.4.1 - 2026-09-27

### Multi-source fusion
- Reworked live multi-source updates around a persistent source window instead of treating the latest source event as the whole ensemble observation.
- Every source is transported to the current time before fusion.
- Slow/event-driven sensors are no longer penalized merely for publishing slowly: extra uncertainty starts only after the source exceeds its normal reporting cadence.
- Source age now increases prediction variance rather than directly changing an arbitrary source weight.
- Added robust variance-aware fusion of the current source snapshot before the common state update.
- Separated state publication cadence from heavy diagnostic attribute refresh so the visible estimate can remain smooth without rebuilding large attribute payloads on every sample.

### Source metrology
- Bias is now treated as a slow source property and is estimated from the full configured `history_days` horizon, using only the quietest process segments.
- Dynamic sections are excluded from bias fitting because source response delays during real physical changes must not be learned as zero-offset errors.
- Added a coarse long-horizon bias grid to keep week-long recalibration computationally bounded.
- Applying a new bias solution preserves the current fused level, avoiding artificial output steps during recalibration.
- Source `sigma` is now also treated as a slow metrology parameter: it is learned from long Recorder history and remains fixed between calibration passes.
- Multi-source sigma calibration uses time-aligned pairwise residuals with bounded anchor sampling.
- Single-source sigma estimation uses local-linear residuals with a quantization floor, avoiding the old first-difference artefact on quantized signals.

### Dynamics
- Replaced single-timescale-driven process-noise fitting with direct multi-horizon identification of both high-order process noise and independent level random-walk diffusion.
- Training scores predictive residuals over 1, 2, 4, 8, 16 and 32 natural steps.
- Long histories keep several contiguous native-cadence blocks distributed over the Recorder horizon so fast dynamics remain visible without optimizing over every raw sample.
- `gated_timescale_s` is now a secondary diagnostic derived from the fitted process noise instead of the quantity that directly determines it.
- Added an independent level random-walk term so the level can follow persistent common motion even when derivative confidence is near zero.
- Changed derivative gating to smooth order-dependent exponential suppression:
  - `w_v = g_1(c_v)`
  - `w_a = w_v * g_2(c_a)`
  - `w_j = w_a * g_3(c_j)`
  - `g_k(c) = exp(-k * (1-c) / c)`
- Higher derivatives therefore require progressively stronger posterior evidence without hard thresholds or state-machine hysteresis.

### Noise model
- Added automatic noise-family diagnostics with Gaussian vs scaled-Poisson classification.
- Quantization is represented independently from the stochastic family through `quantization_step`, `quantization_sigma` and `quantization_confidence`.
- A quantized signal is no longer implicitly classified as Gaussian or Poisson solely because it is quantized.
- Added a stationary scaled-Poisson fallback for low-span signals using the moment identity `skew(X) ~= sigma / mean`, with positive-skew significance and moment-mismatch guards.
- Preserved the broad-span variance-vs-level test as the primary classifier; stationary moment evidence is used only when the level range is too small for that test.
- Added `stationary_scaled_poisson` as an explicit diagnostic reason and retained quantization as an independent observation property.

### Performance and persistence
- Heavy source calibration remains outside the per-sample hot path and is executed through the Home Assistant executor.
- Warmup history is bounded/cleared after training, and checkpoint persistence reuses cached heavy summaries instead of recomputing them on every save.
- Reduced public diagnostics by removing CPU-profiler attributes while keeping process and source-quality information.
- Checkpoint schema was advanced for the new process-noise semantics; the first start after upgrading to 0.4.1 may require one full Recorder bootstrap.
- Noise-detector schema is now part of the checkpoint configuration fingerprint, so a detector upgrade triggers one clean Recorder bootstrap instead of reusing a stale family classification.

### Fixes
- Fixed duplicated residual evidence from appending all fresh sources on every source event.
- Fixed source-event timestamp handling and stale snapshot semantics.
- Fixed `numpy` import in the new robust ensemble fusion path.
- Fixed startup/live calibration inconsistencies that allowed recent process motion to overwrite long-history source statistics.
- Fixed runtime behavior where dynamic derivative terms could create ringing while the level process itself remained too rigid.
- Fixed derivative observability deadlock: confidence gating now controls mean prediction only, while covariance uses the full kinematic transition so `v/a/j` can become observable from level measurements.
- Unified the numerical variance floor used by derivative z-scores and exponential gating, preventing collapsed covariance from falsely opening higher-order derivative gates.
- Made derivative diagnostics use one atomic `x/P` snapshot and grouped public diagnostics into `model`, `calibration`, `dynamics`, `timescales`, `startup`, and `last_update` blocks while retaining flat compatibility aliases.

## 0.4.0 - 2026-09-24

### Root cause

The performance regression was traced to online source calibration rather than the 4D Kalman update itself. The hot path accepted a new observation and then immediately recomputed robust bias/noise statistics over the retained calibration evidence. With several ~1 Hz pressure sources and a multi-day calibration window, the amount of work therefore grew with accumulated history while the information gain from each additional sample was negligible.

In practice this produced a characteristic sawtooth CPU profile: load increased as the calibration evidence grew, eventually saturating the Home Assistant event-loop thread, then dropped when retained state was rebuilt or trimmed and the cycle started again.

The fix separates two timescales that should never have shared one execution path:

- per-sample estimation remains O(1)-like and uses the already learned source calibration;
- expensive O(history) metrology is treated as a background re-certification step, scheduled from drift evidence.

During the investigation, two secondary issues were also found and removed: oversized `source_health` attributes were being rebuilt and serialized on every state update, and the first adaptive scheduler prototype used a fixed outlier threshold that could falsely classify a naturally noisier source as `watch`, recreating a six-hour heavy-refit cadence. The final scheduler uses a per-source historical outlier baseline.

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
- add per-source live diagnostics in `source_health`: raw/corrected value, innovation, z-score and robust weight;
- add explicit `outliers` counter next to `outlier_rate`;
- no change to Bayesian filter math relative to 0.2.1.5.

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
