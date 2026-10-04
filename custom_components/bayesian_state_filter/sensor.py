from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import statistics
import time
import numpy as np
from collections import deque
from functools import partial
from datetime import datetime, timedelta, timezone

from homeassistant.components.sensor import SensorEntity
from homeassistant.helpers.event import async_track_state_change_event
from homeassistant.helpers.start import async_at_started
from homeassistant.helpers.storage import Store

from .const import (
    DOMAIN,
    ATTR_VARIANCE, ATTR_STDDEV, ATTR_VELOCITY, ATTR_NOISE_VELOCITY,
    ATTR_INNOVATION, ATTR_INNOVATION_VAR, ATTR_P_VALUE, ATTR_Z_SCORE,
    ATTR_NOISE_MODEL, ATTR_NOISE_MODEL_MODE, ATTR_NOISE_MODEL_PARAMS,
    ATTR_NOISE_VARIANCE_SOURCE, ATTR_LAST_SOURCE, ATTR_MEASUREMENT_SIGMA,
    ATTR_ROBUST_WEIGHT,
    ATTR_CHARACTERISTIC_TIME, ATTR_CHARACTERISTIC_TIME_P10,
    ATTR_CHARACTERISTIC_TIME_P90, ATTR_CHARACTERISTIC_TIME_CONFIDENCE,
    ATTR_CHARACTERISTIC_TIME_STATUS, ATTR_CHARACTERISTIC_TIME_IDENTIFIABLE,
    ATTR_DYNAMICS_EDGE_MASS, ATTR_DYNAMICS_BOUNDARY_LIMITED,
    ATTR_CHARACTERISTIC_NUGGET_VARIANCE, ATTR_CHARACTERISTIC_PROCESS_VARIANCE,
    ATTR_CHARACTERISTIC_SIGNAL_FRACTION, ATTR_CHARACTERISTIC_FIT_ERROR,
    ATTR_CHARACTERISTIC_LAG_COUNT, ATTR_CHARACTERISTIC_PAIR_COUNT,
    ATTR_CHARACTERISTIC_MIN_LAG, ATTR_CHARACTERISTIC_MAX_LAG,
    ATTR_VELOCITY_TIME, ATTR_VELOCITY_TIME_P10, ATTR_VELOCITY_TIME_P90,
    ATTR_VELOCITY_TIME_CONFIDENCE, ATTR_VELOCITY_TIME_EDGE_MASS,
    ATTR_VELOCITY_TIME_BOUNDARY_LIMITED,
    ATTR_FILTER_MODE, ATTR_PROCESS_NOISE, ATTR_SOURCE_HEALTH,
    ATTR_MEASUREMENT_VARIANCE, ATTR_EFFECTIVE_INNOVATION_VARIANCE,
    ATTR_UPDATE_DT,
    ATTR_RATE_PER_HOUR, ATTR_RATE_STDDEV_PER_HOUR,
    ATTR_CURVATURE_PER_HOUR2, ATTR_CURVATURE_STDDEV_PER_HOUR2,
    ATTR_JERK_PER_HOUR3, ATTR_JERK_STDDEV_PER_HOUR3,
    ATTR_RATE_WEIGHT, ATTR_CURVATURE_WEIGHT, ATTR_JERK_WEIGHT,
    ATTR_RATE_Z, ATTR_CURVATURE_Z, ATTR_JERK_Z,
    ATTR_GATED_TIMESCALE, ATTR_GATED_LOCAL_RMSE,
    ATTR_GATED_LOCAL_RMSE_STEP1, ATTR_GATED_LOCAL_RMSE_STEP2,
    NUMERIC_VARIANCE_FLOOR, STUDENT_T_MIN_WEIGHT,
    FRESHNESS_MEDIAN_DT_MULTIPLIER, FRESHNESS_TAU_FRACTION,
    FRESHNESS_MIN_S, CHARACTERISTIC_REFIT_MIN_S,
    REGIME_CHANGE_Z_THRESHOLD, REGIME_CHANGE_CONFIRMATIONS,
    REGIME_CHANGE_COMPACT_SIGMA,
)
from .core.filter import CoreFilter
from .core.noise_models import GaussianNoise, PoissonLikeNoise
from .core.noise_detection import NoiseDetectionResult, detect_noise_model
from .core.process_noise import IntegratedWienerProcessNoise
from .core.state_models import AdaptivePolynomialStateModel
from .core.gated_training import GatedDynamicsEstimate, train_gated_dynamics
from .core.edge_derivatives import (
    robust_causal_local_polynomial,
    combine_multiscale_derivative,
)
from .core.training import (
    BiasAnchorResult,
    OnlineSourceCalibrator,
    SourceCalibration,
    calibrate_history,
    estimate_biases_from_history,
)
from .core.types import Observation
from .core.updaters import StudentTUpdater
from .core.variogram import CharacteristicTimeEstimate, estimate_characteristic_time

_LOGGER = logging.getLogger(__name__)


async def async_setup_platform(hass, config, async_add_entities, discovery_info=None):
    async_add_entities([BayesianEnsembleSensor(hass, config)])


class BayesianEnsembleSensor(SensorEntity):
    """Bayesian State Filter 0.5.0-dev.24.

    Backward-compatible YAML platform. Multi-source observations are fused
    from the latest per-source estimates at a common time. Older source
    estimates remain usable, but their predictive variance grows with age.
    """

    _attr_has_entity_name = True

    def __init__(self, hass, config):
        self.hass = hass
        name = config.get("name", "Bayesian Sensor")
        self._attr_name = name
        slug = name.lower().replace(" ", "_")
        self._attr_unique_id = f"{DOMAIN}_{slug}"

        ens_cfg = config.get("ensemble", {}) or {}
        raw_sources = list(ens_cfg.get("sources", []))
        self.sources = []
        self._source_models = {}
        for item in raw_sources:
            if isinstance(item, str):
                entity_id = item
                model = None
            elif isinstance(item, dict):
                entity_id = str(item.get("entity_id", "")).strip()
                model = item.get("model")
                model = str(model).strip() if model is not None else None
            else:
                raise ValueError(f"Invalid ensemble source: {item!r}")
            if not entity_id:
                raise ValueError(f"Missing entity_id in ensemble source: {item!r}")
            self.sources.append(entity_id)
            if model:
                self._source_models[entity_id] = model
        self.min_sources = max(int(ens_cfg.get("min_sources", 1)), 1)
        self._single_source = len(self.sources) == 1

        self._model_accuracy = {}
        for model, spec in (ens_cfg.get("models", {}) or {}).items():
            if not isinstance(spec, dict):
                raise ValueError(f"Model {model!r} definition must be a mapping")
            # A model may intentionally omit absolute_accuracy.  Such sources
            # still participate in relative-bias calibration and state updates,
            # but do not vote for the passport absolute-bias anchor.
            if "absolute_accuracy" not in spec or spec.get("absolute_accuracy") is None:
                continue
            accuracy = float(spec["absolute_accuracy"])
            if not math.isfinite(accuracy) or accuracy <= 0:
                raise ValueError(f"Model {model!r} absolute_accuracy must be positive")
            self._model_accuracy[str(model)] = accuracy

        bayes_cfg = config.get("bayes", {}) or {}
        self._bias_anchor = str(bayes_cfg.get("bias_anchor", "median")).strip().lower()
        if self._bias_anchor not in {"median", "mean", "passport"}:
            raise ValueError(f"Unknown bias_anchor={self._bias_anchor!r}")
        self._bias_huber_delta = 1.345
        if self._bias_anchor == "passport":
            anchor_sources = [
                src for src in self.sources
                if self._source_models.get(src) in self._model_accuracy
            ]
            if not anchor_sources:
                raise ValueError(
                    "passport bias_anchor requires absolute_accuracy for at least "
                    "one configured source model"
                )
        self._noise_mode_cfg = str(bayes_cfg.get("noise_model", "auto")).strip().lower()
        if self._noise_mode_cfg not in {"auto", "gaussian", "poisson"}:
            _LOGGER.warning(
                "Bayesian State Filter: unknown noise_model=%r; using auto",
                self._noise_mode_cfg,
            )
            self._noise_mode_cfg = "auto"
        self._history_days = max(float(bayes_cfg.get("history_days", 7.0)), 0.1)
        self._save_every_s = max(float(bayes_cfg.get("save_every_s", 1800.0)), 5.0)
        self._tau_points = max(int(bayes_cfg.get("tau_points", 16)), 8)
        # ``tau_min/max_s`` remain the bounds for the predictive velocity model
        # for backward compatibility.  The level-process characteristic time
        # has separate optional bounds in v2.1.
        self._tau_min_s = self._optional_float(bayes_cfg.get("tau_min_s"))
        self._tau_max_s = self._optional_float(bayes_cfg.get("tau_max_s"))
        self._char_tau_min_s = self._optional_float(
            bayes_cfg.get("characteristic_tau_min_s", bayes_cfg.get("characteristic_time_min_s"))
        )
        self._char_tau_max_s = self._optional_float(
            bayes_cfg.get("characteristic_tau_max_s", bayes_cfg.get("characteristic_time_max_s"))
        )
        self._characteristic_refit_s = max(float(bayes_cfg.get("characteristic_refit_s", 21600.0)), CHARACTERISTIC_REFIT_MIN_S)
        self._warmup_refit_s = max(float(bayes_cfg.get("warmup_refit_s", 21600.0)), CHARACTERISTIC_REFIT_MIN_S)
        self._level_history_max_points = max(int(bayes_cfg.get("level_history_max_points", 5000)), 200)
        self._warmup_max_points_per_source = max(int(bayes_cfg.get("warmup_max_points_per_source", 5000)), 200)
        self._forget_time_s = self._optional_float(bayes_cfg.get("forget_time_s"))
        self._student_nu = max(float(bayes_cfg.get("student_nu", 4.0)), 1.01)
        # Hard winsorisation caps the *state correction* from one pathological
        # observation while preserving the raw innovation for diagnostics and
        # regime-change detection.  Three sigma is intentionally conservative.
        self._innovation_clip_sigma = REGIME_CHANGE_COMPACT_SIGMA
        # Public diagnostics are intentionally small. This switch only changes
        # presentation; it never changes filtering, training, or persistence.
        # Accepted aliases keep old configs working for one transition cycle.
        diagnostics = str(bayes_cfg.get("diagnostics", "normal")).strip().lower()
        aliases = {
            "compact": "normal",
            "full": "debug",
            "verbose": "debug",
        }
        self._diagnostics_mode = aliases.get(diagnostics, diagnostics)
        if self._diagnostics_mode not in {"minimal", "normal", "debug"}:
            _LOGGER.warning(
                "Bayesian State Filter: unknown diagnostics=%r; using normal",
                diagnostics,
            )
            self._diagnostics_mode = "normal"
        self._diagnostics_full = self._diagnostics_mode == "debug"

        self._attr_native_unit_of_measurement = None
        self._attr_device_class = None
        self._attr_state_class = None
        self._attr_icon = None

        self._gaussian_noise = GaussianNoise(sigma=0.1)
        self._poisson_noise = PoissonLikeNoise(k=0.1)
        self._noise_model_name = "gaussian"
        self._runtime_noise_mode = "gaussian"
        self._runtime_level_fraction = 0.0
        self._runtime_variance_source = "constant"
        self._noise_detection_version = 6
        self._noise_detection = NoiseDetectionResult(reason="not_yet_detected")

        default_tau = self._tau_min_s or 3600.0
        self.filter = CoreFilter(
            state_model=AdaptivePolynomialStateModel(3),
            noise_model=self._gaussian_noise,
            updater=StudentTUpdater(
                nu=self._student_nu, min_weight=0.05,
                clip_sigma=self._innovation_clip_sigma,
            ),
            process_noise=IntegratedWienerProcessNoise(order=3, q=0.0, level_q=0.0),
            prior_timescale_s=default_tau,
        )

        self._calibrations: dict[str, SourceCalibration] = {}
        self._source_cal: OnlineSourceCalibrator | None = None
        # Full [x,v,a,j] dynamics identified from history.  The state order is
        # fixed; history-trained plausibility gates only derivative coupling.
        self._gated_dynamics: GatedDynamicsEstimate | None = None
        self._dynamics_bank = None  # legacy field retained only for migration-safe code paths
        self._dynamics = None       # legacy field retained only for old diagnostics
        # Independent level-process characteristic-time estimate.
        self._characteristic: CharacteristicTimeEstimate | None = None
        self._level_history = deque(maxlen=self._level_history_max_points)
        self._level_grid_step = 60.0

        # Independent causal derivative witness built only from direct
        # raw/fused observations. It has no derivative-transport feedback path.
        # During normal tracking it is diagnostic only; if the historical
        # plausibility gate rejects a derivative tail, the witness may be used
        # once as a conservative recovery pseudo-measurement for v/a.
        self._edge_history = deque(maxlen=4096)
        self._edge_derivative_estimate = None
        self._edge_witness_scales = {}
        self._edge_rate_consensus = None
        self._edge_curvature_consensus = None
        self._edge_witness_tau_s = None
        self._edge_derivative_scales = {}
        self._edge_derivative_consensus = {}
        self._edge_witness_tau_s = None

        # Recovery diagnostics for hierarchical derivative reconditioning.
        self._derivative_recondition_count = 0
        self._last_derivative_recondition = None
        # Independent edge witness guard. A witness may repair the latent
        # derivative state only if it is multiscale-consistent, within the
        # history-trained plausibility envelope, and (at runtime) disagrees
        # persistently with the Bayesian posterior.
        self._edge_witness_plausibility_sigma = 3.0
        self._edge_divergence_sigma = 5.0
        self._edge_divergence_confirm_updates = 5
        self._edge_divergence_counts = {1: 0, 2: 0, 3: 0}
        self._edge_divergence_bad_since = {1: None, 2: None, 3: None}
        self._edge_divergence_cancel_sigma = 3.0
        self._edge_last_divergence_z = {1: 0.0, 2: 0.0, 3: 0.0}
        self._edge_agreement_z = {1: None, 2: None, 3: None}
        self._edge_agreement_local_weight = {1: None, 2: None, 3: None}
        self._edge_jerk_witness_scales = {}
        self._edge_jerk_consensus = None
        self._edge_last_update_ts = None
        self._characteristic_task = None
        self._last_characteristic_fit_ts = 0.0

        # Relative bias is a long-horizon metrology parameter. Refit it only
        # from the configured Recorder history (history_days), never from the
        # short live calibration window. With the default 7 d history this is
        # one refit per day; shorter configured histories never refit faster
        # than every 6 h.
        self._bias_history_refit_s = max(self._history_days * 86400.0 / 7.0, 21600.0)
        self._bias_history_task = None
        self._last_bias_history_refit_ts = 0.0

        self._warmup_history = {
            src: deque(maxlen=self._warmup_max_points_per_source)
            for src in self.sources
        }
        # Source cadence must not depend on warmup history.  Once gated
        # dynamics are restored/trained, warmup history is deliberately
        # stopped and cleared, while cadence estimation continues here.
        self._last_source_event_ts: dict[str, float] = {}
        self._warmup_task = None
        self._last_warmup_fit_ts = 0.0

        # Background-work diagnostics. Kept as scalar counters/timers only:
        # no diagnostic history arrays, so observability cannot recreate the
        # original CPU/memory problem.
        self._diag_bg = {
            "warmup_runs": 0,
            "warmup_failures": 0,
            "warmup_last_ms": 0.0,
            "warmup_max_ms": 0.0,
            "warmup_last_points": 0,
            "warmup_last_finished_ts": 0.0,
            "characteristic_runs": 0,
            "characteristic_failures": 0,
            "characteristic_last_ms": 0.0,
            "characteristic_max_ms": 0.0,
            "characteristic_last_points": 0,
            "characteristic_last_finished_ts": 0.0,
            "save_runs": 0,
            "save_failures": 0,
            "save_last_ms": 0.0,
            "save_max_ms": 0.0,
            "save_last_bytes_est": 0,
        }

        # Estimator still processes every source observation.  Publish the
        # numeric state at ~1 Hz so Recorder sees a smooth trajectory, while
        # rebuilding the expensive diagnostics dictionary only every 5 s.
        # This keeps the chart smooth without returning to ~3 full attr builds/s.
        self._publish_interval_s = 1.0
        self._attrs_publish_interval_s = 5.0
        self._last_state_publish_monotonic = None
        self._last_attrs_publish_monotonic = None

        self._state = None
        self._attrs = {}
        # Diagnostics of the most recent *live* observation.  Keep these
        # separately from _build_attrs() so a characteristic-time refit does
        # not make last-source/update diagnostics disappear.
        self._last_source = None
        self._last_update_diag = None

        # Conservative online change-point detector for genuinely discrete
        # regime changes.  It reacts only to several same-sign, very large
        # innovations.  Normal continuous dynamics remain entirely under the
        # x-v-a-j model.
        self._regime_change_z_threshold = REGIME_CHANGE_Z_THRESHOLD
        self._regime_change_confirmations = REGIME_CHANGE_CONFIRMATIONS
        self._regime_candidate_sign = 0
        self._regime_candidate_count = 0
        self._regime_candidate_first_ts = None
        self._regime_candidate_last_ts = None
        self._regime_candidate_peak_z = 0.0
        self._regime_candidate_values = []
        self._regime_candidate_variances = []
        self._regime_candidate_startup = False
        self._regime_cooldown_until = 0.0
        self._last_regime_change = None
        # A full Recorder bootstrap can leave the latent level extrapolated a
        # long way from the first live sample even though the historical model
        # itself is healthy.  Treat the first few live observations as startup
        # alignment evidence, not as physical regime-change events.
        self._regime_startup_guard_updates = 3
        self._regime_startup_guard_remaining = 0
        self._regime_startup_reanchors = 0
        self._startup_sync_replayed = 0
        self._startup_sync_current = 0
        self._startup_sync_gap_s = 0.0

        # Last-line publication safety.  The Bayesian state continues to run
        # internally, but if its published level becomes demonstrably worse
        # than the direct observation we temporarily expose the untransported
        # raw/fused level instead.  This protects users from runaway dynamics
        # without teaching the filter the actuator/control logic.
        self._fallback_active = False
        self._fallback_value = None
        self._fallback_variance = None
        self._fallback_last_z = 0.0
        self._fallback_reason = None
        self._fallback_safe_count = 0
        self._fallback_entries = 0
        self._fallback_enter_z = 8.0
        self._fallback_exit_z = 3.0
        self._fallback_exit_confirmations = 5
        # Hard derivative recovery is intentionally much slower than the
        # continuous plausibility gate.  The gate may suppress an implausible
        # derivative immediately; destructive tail reset is allowed only when
        # the first bad derivative remains beyond 5 historical sigmas for both
        # a minimum number of observations and several trained process
        # timescales.
        self._derivative_recovery_enter_sigma = 5.0
        self._derivative_recovery_cancel_sigma = 4.0
        self._derivative_recovery_confirm_updates = 5
        self._derivative_recovery_timescale_factor = 1.0
        self._fallback_derivative_weight_threshold = 0.0
        self._fallback_derivative_weight_confirm_updates = self._derivative_recovery_confirm_updates
        self._fallback_derivative_weight_counts = {1: 0, 2: 0, 3: 0}
        self._derivative_recovery_bad_since = {1: None, 2: None, 3: None}
        self._fallback_last_event = None

        # Per-source live diagnostics.  These are presentation-only and are
        # deliberately kept outside SourceCalibration/persistence so adding
        # observability cannot change the estimator or stored calibration.
        self._source_last_diag: dict[str, dict] = {}
        self._ready = False
        self._last_save_ts = 0.0
        self._checkpoint_version = 12  # dev.27: derivative plausibility is retrained at gated dynamics timescale
        self._checkpoint_config_fingerprint = self._make_checkpoint_config_fingerprint()
        self._last_processed_by_source: dict[str, float] = {}
        self._startup_mode = "initializing"
        # New key: old v1 snapshot has incompatible semantics (notably q).
        # Namespace the checkpoint by the physical configuration fingerprint.
        # This prevents an old entity instance from overwriting the new
        # configuration checkpoint during YAML/platform reload or shutdown.
        self._store_key = f"{DOMAIN}_{slug}_passport_{self._checkpoint_config_fingerprint[:16]}"
        self._store = Store(hass, 1, self._store_key)

    def _publication_due(self) -> bool:
        now = time.perf_counter()
        last = self._last_state_publish_monotonic
        if last is None or now - last >= self._publish_interval_s:
            self._last_state_publish_monotonic = now
            return True
        return False

    def _attrs_refresh_due(self) -> bool:
        now = time.perf_counter()
        last = self._last_attrs_publish_monotonic
        if last is None or now - last >= self._attrs_publish_interval_s:
            self._last_attrs_publish_monotonic = now
            return True
        return False

    @staticmethod
    def _optional_float(value):
        if value is None:
            return None
        try:
            v = float(value)
            return v if math.isfinite(v) and v > 0 else None
        except (TypeError, ValueError):
            return None

    async def async_added_to_hass(self):
        self.async_on_remove(
            async_track_state_change_event(
                self.hass, self.sources, self._handle_event
            )
        )

        async def _after_start(_hass):
            try:
                await self._initialize()
            except Exception:
                _LOGGER.exception("Bayesian State Filter 0.5.0-dev.24 initialization failed")
                await self._restore_fallback()
            # Finish hidden startup synchronization before publishing anything.
            # Recorder catch-up plus the latest in-memory source states bridge the
            # restart gap while the entity is still invisible to HA/Recorder.
            await self._synchronize_startup_to_now()
            self._regime_startup_guard_remaining = 0
            self._reset_regime_candidate()
            self._ready = True
            if self._state is not None:
                self.async_write_ha_state()

        self.async_on_remove(async_at_started(self.hass, _after_start))

    async def async_will_remove_from_hass(self):
        if self._warmup_task is not None:
            self._warmup_task.cancel()
        if self._characteristic_task is not None:
            self._characteristic_task.cancel()
        if self._bias_history_task is not None:
            self._bias_history_task.cancel()
        await self._save_state()

    # ------------------------------------------------------------------
    # Startup training
    # ------------------------------------------------------------------

    async def _initialize(self):
        # Fast path: a v2 checkpoint contains the complete learned state.
        # Restore it and replay only Recorder observations that arrived after
        # the checkpoint.  The expensive multi-day training pass is therefore
        # performed only for the first bootstrap (or after an incompatible
        # checkpoint/schema change).
        if await self._restore_checkpoint():
            self._startup_mode = "checkpoint_restore"
            if self._noise_mode_cfg == "auto" and self._noise_detection.reason == "not_yet_detected":
                pseudo_history = {"restored_level": list(self._level_history)}
                self._noise_detection = await self.hass.async_add_executor_job(
                    lambda: detect_noise_model(pseudo_history)
                )
            self._apply_noise_model()

            # Prefer the independently persisted edge history. Older
            # checkpoints do not have it, so first try the restored level
            # history and finally perform a one-time recent Recorder backfill.
            if self._edge_history_observable():
                self._recompute_edge_witness()
                self._edge_last_update_ts = float(self._edge_history[-1][0])
                self._update_edge_agreement_weights()
            else:
                self._seed_edge_history_from_level_history()
                if not self._edge_history_observable():
                    await self._backfill_edge_history_from_recorder()

            await self._catch_up_from_recorder()
            if self.filter.t_last is not None:
                self._state = round(float(self.filter.x[0]), 6)
                self._build_attrs(last_out=None)
            _LOGGER.info(
                "Bayesian State Filter 0.5.0-dev.24 restored checkpoint and caught up incrementally; t=%.3f",
                float(self.filter.t_last or 0.0),
            )
            return

        self._startup_mode = "full_recorder_bootstrap"
        histories = {}
        for entity_id in self.sources:
            seq = await self._fetch_history(entity_id)
            parsed = self._parse_states(seq)
            if parsed:
                histories[entity_id] = parsed

        if not histories:
            await self._restore_fallback()
            return

        if self._noise_mode_cfg == "auto":
            self._noise_detection = await self.hass.async_add_executor_job(
                lambda: detect_noise_model(histories)
            )
            self._apply_noise_model()

        result = await self.hass.async_add_executor_job(
            lambda: calibrate_history(
                histories,
                tau_points=self._tau_points,
                forget_time_s=self._forget_time_s,
                tau_min_s=self._tau_min_s,
                tau_max_s=self._tau_max_s,
                characteristic_tau_min_s=self._char_tau_min_s,
                characteristic_tau_max_s=self._char_tau_max_s,
                bias_anchor=self._bias_anchor,
                source_models=self._source_models,
                model_accuracy=self._model_accuracy,
                huber_delta=self._bias_huber_delta,
            )
        )
        self._calibrations = result.sources
        self._source_cal = OnlineSourceCalibrator(
            self._calibrations,
            startup_pair_rows=result.startup_pair_rows,
            calibration_window_s=result.calibration_window_s,
            bias_anchor=self._bias_anchor,
            source_models=self._source_models,
            model_accuracy=self._model_accuracy,
            huber_delta=self._bias_huber_delta,
        )
        self._last_bias_history_refit_ts = max(
            (float(seq[-1][0]) for seq in histories.values() if seq),
            default=0.0,
        )
        if self._calibrations:
            sigmas = sorted(c.sigma for c in self._calibrations.values() if c.sigma > 0)
            if sigmas:
                self._gaussian_noise.set_sigma(sigmas[len(sigmas) // 2])
        self._dynamics_bank = None
        self._dynamics = None
        self._characteristic = result.characteristic
        self._level_grid_step = max(float(result.grid_step), 1.0)
        self._level_history = deque(result.fused_points[-self._level_history_max_points:], maxlen=self._level_history_max_points)
        if result.fused_points:
            self._last_characteristic_fit_ts = float(result.fused_points[-1][0])

        # Identify q/timescale and robust derivative plausibility for the permanent
        # [x,v,a,j] model.  Effective derivative coupling is trained from the
        # observed history; posterior significance remains diagnostic only.
        self._gated_dynamics = None
        dynamics_points = result.dynamics_points or result.fused_points
        if len(dynamics_points) >= 40:
            try:
                self._gated_dynamics = await self.hass.async_add_executor_job(
                    lambda: train_gated_dynamics(dynamics_points, nu=self._student_nu)
                )
            except Exception:
                _LOGGER.exception("Bayesian State Filter gated dynamics training failed")
        if self._gated_dynamics is not None:
            T = self._gated_dynamics.timescale_s
            if not math.isfinite(T) or T <= 0:
                T = max(self._gated_dynamics.history_span_s, self._level_grid_step, 1.0)
            self.filter.tau = T
            self.filter.q_process = self._gated_dynamics.q_process
            self.filter.level_q_process = self._gated_dynamics.level_q_process
            self._apply_derivative_plausibility()

        self._apply_noise_model()

        # Full bootstrap already has the direct/fused Recorder history in
        # _level_history and has just established the process timescale.
        # Prime the causal witness now so the first published state and any
        # startup derivative reconciliation have historical edge evidence.
        self._seed_edge_history_from_level_history()
        if not self._edge_history_observable():
            await self._backfill_edge_history_from_recorder()

        if result.fused_points:
            await self.hass.async_add_executor_job(self._replay_fused, dynamics_points)
            # The full history bootstrap has consumed all source history through
            # each source's last Recorder sample.  Remember those watermarks so
            # the next restart can request only the unseen tail.
            self._last_processed_by_source = {
                src: float(seq[-1][0]) for src, seq in histories.items() if seq
            }
            # Seed the live source cache with the exact tail already represented
            # by the historical fused replay. This lets subsequent hidden
            # Recorder catch-up fuse the first post-bootstrap source event with
            # contemporaneous values from slower sources instead of waiting for
            # every source to report again.
            self._seed_source_cache_from_histories(histories)
            await self._catch_up_from_recorder()
            self._state = round(float(self.filter.x[0]), 6)
            self._build_attrs(last_out=None)
            await self._save_state()
            _LOGGER.info(
                "Bayesian State Filter 0.4.1-dev.15 trained from %.2f d: gated_tau=%s q=%s level_q=%s "
                "local_rmse=%s characteristic_time=%s (status=%s, conf=%.3f)",
                result.history_span / 86400.0,
                (f"{self.filter.tau:.1f} s" if self._gated_dynamics is not None else "fallback"),
                f"{self.filter.q_process:.6e}",
                f"{self.filter.level_q_process:.6e}",
                (f"{self._gated_dynamics.validation_rmse:.6g}" if self._gated_dynamics is not None else "unknown"),
                (f"{self._characteristic.tau:.1f} s" if self._characteristic and self._characteristic.tau is not None else "unknown"),
                self._characteristic.status if self._characteristic else "unavailable",
                self._characteristic.confidence if self._characteristic else 0.0,
            )
        else:
            await self._restore_fallback()

    def _replay_fused(self, points):
        if not points:
            return
        t0, z0, v0 = points[0]
        self.filter.reset(z0, t=t0, variance=v0)
        for t, z, var in points[1:]:
            self.filter.step(Observation(t=t, z=z, variance=var, source="history_fusion"))

    async def _fetch_history(self, entity_id, start_ts: float | None = None):
        """Fetch raw recorder history for one source.

        Home Assistant removed the old ``history.get_states`` helper.  Use the
        supported ``get_significant_states`` API with keyword arguments: its
        fifth positional argument is ``filters``, not ``include_start_time_state``.
        Passing booleans positionally here therefore silently broke v2.0.0 on
        current HA and made startup fall back to warmup.
        """
        now = datetime.now(timezone.utc)
        start = (datetime.fromtimestamp(float(start_ts), timezone.utc) if start_ts is not None else now - timedelta(days=self._history_days))
        try:
            from homeassistant.components.recorder import get_instance
            from homeassistant.components.recorder.history import get_significant_states

            instance = get_instance(self.hass)
            data = await instance.async_add_executor_job(
                partial(
                    get_significant_states,
                    self.hass,
                    start,
                    now,
                    entity_ids=[entity_id],
                    include_start_time_state=True,
                    significant_changes_only=False,
                    minimal_response=False,
                    no_attributes=True,
                )
            )
            seq = data.get(entity_id, [])
            _LOGGER.info(
                "Bayesian State Filter history: %s -> %d states (%.2f d requested)",
                entity_id,
                len(seq),
                self._history_days,
            )
            return seq
        except Exception:
            _LOGGER.exception(
                "Bayesian State Filter failed to read recorder history for %s",
                entity_id,
            )
            return []

    @staticmethod
    def _parse_states(seq):
        out = []
        for s in seq or []:
            try:
                z = float(s.state)
                if not math.isfinite(z):
                    continue
                t = s.last_updated
                if t.tzinfo is None:
                    t = t.replace(tzinfo=timezone.utc)
                out.append((t.timestamp(), z))
            except Exception:
                continue
        return out

    # ------------------------------------------------------------------
    # Live
    # ------------------------------------------------------------------

    async def _handle_event(self, event):
        if not self._ready:
            return
        st = event.data.get("new_state")
        if not st or st.state in ("unknown", "unavailable", None):
            return
        try:
            raw = float(st.state)
        except (TypeError, ValueError):
            return
        if not math.isfinite(raw):
            return
        self._inherit_metadata(st)
        await self._process_sample(
            st.entity_id, raw, self._state_timestamp(st),
            st=st, write_state=True, allow_save=True, schedule_background=True,
        )

    def _reset_regime_candidate(self):
        self._regime_candidate_sign = 0
        self._regime_candidate_count = 0
        self._regime_candidate_first_ts = None
        self._regime_candidate_last_ts = None
        self._regime_candidate_peak_z = 0.0
        self._regime_candidate_values = []
        self._regime_candidate_variances = []
        self._regime_candidate_startup = False

    def _maybe_recover_regime_change(self, t: float, corrected: float, variance: float, out) -> bool:
        """Detect a persistent *stable* discrete jump and recover level in-place.

        Raw innovations drive detection, while the Student-t updater separately
        clips one-step state corrections at three sigma.  A regime change must
        therefore be both persistent and compact around a new level.  Wild
        startup staircases/glitches can stay many sigma away for several samples
        without being mistaken for a new physical regime.
        """
        if out is None or out.mode != "tracking" or out.dt <= 0:
            self._reset_regime_candidate()
            return False

        now = float(t)
        if now < float(self._regime_cooldown_until):
            self._reset_regime_candidate()
            return False

        innovation = float(out.innovation)
        innovation_var = max(float(out.innovation_var), NUMERIC_VARIANCE_FLOOR)
        z = abs(innovation) / math.sqrt(innovation_var)

        startup_sample = bool(self._ready and self._regime_startup_guard_remaining > 0)
        if startup_sample:
            self._regime_startup_guard_remaining -= 1

        if (not math.isfinite(z)) or z < self._regime_change_z_threshold or innovation == 0.0:
            self._reset_regime_candidate()
            return False

        sign = 1 if innovation > 0.0 else -1
        gap_limit = max(30.0, 4.0 * max(float(out.dt), 1e-6))
        same_event = (
            self._regime_candidate_sign == sign
            and self._regime_candidate_last_ts is not None
            and now - float(self._regime_candidate_last_ts) <= gap_limit
        )

        if not same_event:
            self._regime_candidate_sign = sign
            self._regime_candidate_count = 1
            self._regime_candidate_first_ts = now
            self._regime_candidate_peak_z = z
            self._regime_candidate_values = [float(corrected)]
            self._regime_candidate_variances = [max(float(variance), NUMERIC_VARIANCE_FLOOR)]
            self._regime_candidate_startup = startup_sample
        else:
            self._regime_candidate_count += 1
            self._regime_candidate_peak_z = max(self._regime_candidate_peak_z, z)
            self._regime_candidate_values.append(float(corrected))
            self._regime_candidate_variances.append(max(float(variance), NUMERIC_VARIANCE_FLOOR))
            self._regime_candidate_startup = self._regime_candidate_startup or startup_sample

        # Keep only the confirmation window.  A true step settles around one
        # new level; a startup/glitch staircase does not.
        n = int(self._regime_change_confirmations)
        if len(self._regime_candidate_values) > n:
            self._regime_candidate_values = self._regime_candidate_values[-n:]
            self._regime_candidate_variances = self._regime_candidate_variances[-n:]
        self._regime_candidate_last_ts = now

        if self._regime_candidate_count < n or len(self._regime_candidate_values) < n:
            return False

        vals = list(self._regime_candidate_values[-n:])
        target = float(sorted(vals)[len(vals) // 2])
        target_sigma = math.sqrt(max(
            float(sorted(self._regime_candidate_variances[-n:])[len(vals) // 2]),
            NUMERIC_VARIANCE_FLOOR,
        ))
        compact_limit = self._innovation_clip_sigma * target_sigma
        compact = max(abs(v - target) for v in vals) <= compact_limit
        if not compact:
            # Preserve the latest window and wait for a genuinely stable new
            # level instead of snapping to a multi-point transient.
            self._regime_candidate_count = n
            return False

        old_level = float(self.filter.x[0])
        jump = target - old_level
        evidence = {
            "timestamp": now,
            "jump": jump,
            "innovation": innovation,
            "peak_z": float(self._regime_candidate_peak_z),
            "confirmations": n,
            "direction": "up" if sign > 0 else "down",
        }

        self.filter.recover_level_jump(target, max(float(variance), NUMERIC_VARIANCE_FLOOR), t=now)
        if self._regime_candidate_startup:
            self._regime_startup_reanchors += 1
            _LOGGER.info(
                "Bayesian State Filter startup re-anchor after stable confirmation: "
                "jump=%+.6g z_peak=%.3f",
                jump, evidence["peak_z"],
            )
        else:
            self._last_regime_change = evidence
            _LOGGER.info(
                "Bayesian State Filter regime change recovered: jump=%+.6g z_peak=%.3f",
                jump, evidence["peak_z"],
            )
        self._regime_cooldown_until = now + max(30.0, 4.0 * max(float(out.dt), 1e-6))
        self._reset_regime_candidate()
        return True

    async def _process_sample(self, src: str, raw: float, t: float, *, st=None,
                              write_state: bool, allow_save: bool,
                              schedule_background: bool):
        """Process one raw source observation through the normal live path.

        Recorder catch-up calls this with output/background work disabled, so
        restart replay is mathematically the same as live processing without
        flooding HA state writes or checkpoint saves.
        """
        if self._source_cal is None:
            self._source_cal = OnlineSourceCalibrator(
                self._calibrations, bias_anchor=self._bias_anchor,
                source_models=self._source_models, model_accuracy=self._model_accuracy,
                huber_delta=self._bias_huber_delta,
            )
        self._source_cal.ensure_source(src, raw, t)
        cal = self._calibrations[src]
        self._update_source_dt(cal, src, t)
        self._source_cal.update_snapshot(t, self._window_tau(), updated_source=src)
        self._last_processed_by_source[src] = max(
            float(t), float(self._last_processed_by_source.get(src, float("-inf")))
        )

        if self._fresh_source_count(t) < self.min_sources:
            if self._gated_dynamics is None:
                self._append_warmup_sample(src, t, raw)
            return

        source_corrected = raw - cal.bias
        mode = self._apply_noise_model()

        # Multi-source mode is an estimate of one common physical quantity,
        # not a queue of independent measurements. Build one contemporaneous
        # ensemble observation from the latest estimate of every fresh source.
        # A slow source is not down-weighted merely because it is slow: its
        # value is transported to ``t`` and the uncertainty of that transport
        # is added to its measurement variance.
        fused = self._current_fused_snapshot(t, with_components=True)
        if fused is None:
            return
        corrected, variance, components = fused

        if self.filter.t_last is None:
            # The fused snapshot is already a robust bootstrap measurement.
            # Do not fall back to one arbitrary triggering source.
            pass

        out = self.filter.step(Observation(
            t=t, z=corrected, source="ensemble" if not self._single_source else src,
            variance=variance,
            meta={"trigger_source": src, "raw": raw, "bias": cal.bias},
        ))

        regime_recovered = self._maybe_recover_regime_change(t, corrected, variance, out)

        # Keep the independent causal witness current before derivative
        # recovery decisions.  It is built only from direct/fused observations,
        # never from the latent Bayesian derivative state.
        edge_updated = self._append_edge_snapshot(t)
        # Coupling for the next causal prediction is determined by how well the
        # current Bayesian derivatives agree with the latest independent edge
        # witness. This is refreshed on every source event because Bayes can
        # move even when edge cadence suppresses a new witness point.
        self._update_edge_agreement_weights()

        # Two recovery layers:
        #   1) historical 5-sigma fallback after max(5 effective updates,
        #      1 process timescale), only when no valid edge witness exists;
        #   2) independent Bayes<->edge watchdog: enter beyond 5 witness sigma,
        #      cancel below 3, recover after max(5 fresh edge updates, 1 tau).
        #
        # Only one destructive derivative repair is allowed per live update.
        derivative_recovery = None
        if not regime_recovered:
            derivative_recovery = self._maybe_reset_persistently_untrusted_derivative_tail(t)
            if derivative_recovery is not None:
                self._record_derivative_recondition(
                    t, derivative_recovery,
                    reason=derivative_recovery.get(
                        "reason", "persistent_5sigma_timescale_recovery"
                    ),
                )
            elif edge_updated:
                derivative_recovery = self._runtime_edge_divergence_recovery()
                if derivative_recovery is not None:
                    self._record_derivative_recondition(
                        t, derivative_recovery,
                        reason="persistent_edge_divergence",
                    )

        if derivative_recovery is not None:
            self._update_edge_agreement_weights()

        # Source health is diagnostic evidence about the triggering source
        # relative to the contemporaneous ensemble, not the Kalman innovation
        # of the fused measurement. This avoids attributing a common process
        # move to whichever source happened to report first.
        if self._single_source:
            z = abs(out.innovation) / math.sqrt(
                max(out.innovation_var, NUMERIC_VARIANCE_FLOOR)
            )
            weight = float(out.diag.get("weight", 1.0))
        else:
            comp = components.get(src)
            if comp is not None:
                z = float(comp["z_score"])
                weight = float(comp["robust_weight"])
            else:
                z = 0.0
                weight = 1.0
        cal.updates += 1
        if weight < 0.25 or z > 4.0:
            cal.outliers += 1

        self._source_last_diag[src] = {
            "raw": float(raw),
            "corrected": float(source_corrected),
            "innovation": float(source_corrected - corrected),
            "z_score": float(z),
            "robust_weight": float(weight),
        }
        if not self._single_source:
            self._source_cal.observe_innovation(src, t, z, weight)

        self._update_dynamics(t, corrected, variance, out.dt)
        self._append_level_snapshot(t)
        if schedule_background:
            self._maybe_schedule_characteristic_fit(t)
            self._maybe_schedule_bias_history_refit(t)

        # Publication safety is deliberately evaluated after the Bayesian step
        # and regime-change logic.  The latent filter always keeps learning;
        # only the value exposed to HA can fall back to direct raw/fused data.
        self._update_raw_fallback(t, regime_recovered=regime_recovered)
        bayes_level = float(self.filter.x[0]) if regime_recovered else float(out.y_mean)
        self._state = round(self._published_level(bayes_level), 6)
        self._last_source = src
        self._remember_update_diag(out)
        # Keep estimator hot-path and HA publication cadence independent.
        # For dense sources we publish at most once per configured interval.
        # Sparse sources naturally publish every observation because their
        # inter-arrival time exceeds the throttle.
        if write_state and self._publication_due():
            if not self._attrs or self._attrs_refresh_due():
                self._build_attrs(last_out=out)
            self.async_write_ha_state()

        if self._gated_dynamics is None:
            self._append_warmup_sample(src, t, raw)
            if schedule_background:
                self._maybe_schedule_warmup_training()
        elif any(self._warmup_history.values()):
            # Restored/trained dynamics make warmup evidence dead weight.
            self._clear_warmup_history()

        if allow_save and t - self._last_save_ts >= self._save_every_s:
            self._last_save_ts = t
            await self._save_state()


    async def _catch_up_from_recorder(self) -> int:
        """Replay only source observations newer than persisted watermarks.

        The replay is intentionally silent: callers use it during startup before
        the first HA state publication so restart gaps affect covariance/state
        internally without drawing a synthetic discontinuity in Recorder.
        """
        events = []
        global_floor = float(self.filter.t_last or 0.0)
        for src in self.sources:
            watermark = float(self._last_processed_by_source.get(src, global_floor))
            # Include a tiny overlap because Recorder's include-start semantics
            # and timestamp precision differ between HA versions; duplicates are
            # discarded explicitly below.
            seq = await self._fetch_history(src, start_ts=max(watermark - 1.0, 0.0))
            for t, raw in self._parse_states(seq):
                if t <= watermark + 1e-6:
                    continue
                events.append((float(t), src, float(raw)))

        events.sort(key=lambda x: (x[0], x[1]))
        replayed = 0
        for t, src, raw in events:
            # A late Recorder row older than the already-restored global filter
            # state cannot be replayed into a causal Kalman filter.  It was
            # already represented by the checkpoint and is safely skipped.
            if self.filter.t_last is not None and t < float(self.filter.t_last) - 1e-6:
                self._last_processed_by_source[src] = max(
                    t, self._last_processed_by_source.get(src, t)
                )
                continue
            await self._process_sample(
                src, raw, t, st=None, write_state=False, allow_save=False,
                schedule_background=False,
            )
            replayed += 1

        if replayed:
            self._build_attrs(last_out=None)
            await self._save_state()
        _LOGGER.info("Bayesian State Filter incremental catch-up replayed %d source observations", replayed)
        return replayed

    def _seed_source_cache_from_histories(self, histories):
        """Seed source cache from historical tails without updating the filter.

        Those samples are already represented by ``_replay_fused``. Replaying
        them again would double-count evidence, but retaining their per-source
        values/timestamps is required for a causal ensemble catch-up.
        """
        if self._source_cal is None:
            return
        for src, seq in (histories or {}).items():
            if not seq or src not in self._calibrations:
                continue
            t, raw = seq[-1]
            self._source_cal.ensure_source(src, float(raw), float(t))

    async def _synchronize_startup_to_now(self):
        """Bring restored/trained state to current source time before publication.

        Startup has two hidden bridges:
          1. replay any Recorder rows that appeared while initialization ran;
          2. replay newer in-memory HA source states that Recorder has not flushed
             yet. Older/current-equal states only seed the source cache and are
             never double-counted by the Kalman filter.

        The entity remains ``_ready == False`` throughout, so no intermediate
        state can leak into HA history.
        """
        before = float(self.filter.t_last or 0.0)
        replayed = await self._catch_up_from_recorder()
        self._startup_sync_replayed += int(replayed)

        if self._source_cal is None:
            self._source_cal = OnlineSourceCalibrator(
                self._calibrations, bias_anchor=self._bias_anchor,
                source_models=self._source_models, model_accuracy=self._model_accuracy,
                huber_delta=self._bias_huber_delta,
            )

        baseline = []
        pending = []
        global_t = float(self.filter.t_last or 0.0)
        for src in self.sources:
            st = self.hass.states.get(src)
            if not st or st.state in ("unknown", "unavailable", None):
                continue
            try:
                raw = float(st.state)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(raw):
                continue
            self._inherit_metadata(st)
            t = float(self._state_timestamp(st))
            watermark = float(self._last_processed_by_source.get(src, float("-inf")))
            # If Recorder/filter already consumed this timestamp, we only need
            # its value in the per-source cache. Never step the filter backwards.
            if t <= watermark + 1e-6 or t <= global_t + 1e-6:
                baseline.append((t, src, raw))
            else:
                pending.append((t, src, raw))

        for t, src, raw in sorted(baseline):
            if src in self._calibrations:
                self._source_cal.ensure_source(src, raw, t)

        current_replayed = 0
        for t, src, raw in sorted(pending, key=lambda x: (x[0], x[1])):
            await self._process_sample(
                src, raw, t, st=None, write_state=False, allow_save=False,
                schedule_background=False,
            )
            current_replayed += 1

        self._startup_sync_current += current_replayed
        after = float(self.filter.t_last or before)
        if before > 0.0 and after >= before:
            self._startup_sync_gap_s = max(self._startup_sync_gap_s, after - before)

        # Startup replay goes through the same persistence gate as live data.
        # The counters are updated for every replayed measurement, so a stale
        # checkpoint tail is reset only if its low plausibility weight persists
        # for the configured number of consecutive observations.
        startup_recovery = self._maybe_reset_persistently_untrusted_derivative_tail(
            float(self.filter.t_last or after or before or 0.0),
            count_update=False,
        )
        if startup_recovery is not None:
            self._record_derivative_recondition(
                float(self.filter.t_last or after or before or 0.0),
                startup_recovery,
                reason="startup_persistent_5sigma_timescale_recovery",
            )

        if self.filter.t_last is not None:
            self._state = round(float(self.filter.x[0]), 6)
            self._build_attrs(last_out=None)
        if replayed or current_replayed or startup_recovery is not None:
            await self._save_state()

        _LOGGER.info(
            "Bayesian State Filter startup sync complete: recorder=%d current=%d gap=%.3f s t=%.3f",
            replayed, current_replayed, self._startup_sync_gap_s,
            float(self.filter.t_last or 0.0),
        )

    def _apply_derivative_plausibility(self):
        """Install history-trained v/a/j plausibility scales in the state model."""
        model = self.filter.state_model
        if not hasattr(model, "set_derivative_plausibility"):
            return
        gd = self._gated_dynamics
        if gd is None:
            return
        model.set_derivative_plausibility(
            scales=gd.derivative_scales,
            centers=gd.derivative_centers,
            samples=gd.derivative_samples,
        )

    def _update_dynamics(self, t, corrected, variance, dt):
        # Structural q/timescale are identified from history and persisted.
        # Live observations update the [x,v,a,j] posterior only.
        return

    def _window_tau(self):
        """Time scale used only for robust/source-maintenance windows.

        The level-process characteristic time is preferred when it is actually
        identified.  Otherwise fall back to the predictive velocity memory;
        the two quantities are never conflated in diagnostics.
        """
        if (self._characteristic is not None
                and self._characteristic.identifiable
                and self._characteristic.tau is not None):
            return max(float(self._characteristic.tau), 1.0)
        return max(float(self.filter.tau), 1.0)

    def _transport_source_to_now(self, corrected: float, age_s: float):
        """Transport a cached source estimate to the current ensemble time.

        The cached value remains an estimate of the same physical process.
        We use the current common derivative posterior only to transport its
        mean; uncertainty in that transport is added to the source variance.
        Thus source age affects *variance*, not an arbitrary age weight.
        """
        age = max(float(age_s), 0.0)
        if age <= 1e-9 or self.filter.t_last is None:
            return float(corrected), 0.0

        try:
            x = self.filter.x
            P = self.filter.P
            weights = self.filter.state_model.effective_weights(x, P, age)

            h = np.zeros(len(x), dtype=float)
            delta = 0.0
            for order in range(1, len(x)):
                coeff = float(weights[order]) * (age ** order) / math.factorial(order)
                delta += coeff * float(x[order])
                h[order] = coeff

            # Uncertainty of the inferred process change since the source last
            # reported, plus fresh process noise accumulated over that age.
            derivative_var = float(h @ P @ h.T)
            q = self.filter.process_noise.Q(age)
            process_var = float(q[0, 0]) if q.size else 0.0
            age_var = max(derivative_var, 0.0) + max(process_var, 0.0)
            if not math.isfinite(age_var):
                age_var = float("inf")
            return float(corrected + delta), age_var
        except Exception:
            # Conservative fallback: keep the last estimate and at least grow
            # uncertainty with the independent level random walk.
            q_level = max(float(self.filter.level_q_process), 0.0)
            return float(corrected), float(q_level * age)

    @staticmethod
    def _robust_fuse(values, variances, labels=None):
        """Variance-aware robust fusion of contemporaneous source estimates.

        Median/MAD provide the robust centre/scale used only for outlier
        resistance. The final location is a precision-weighted mean, so an
        older source contributes less *only because its propagated variance is
        larger*. A conservative variance floor prevents correlated sensors from
        making the ensemble spuriously overconfident.
        """
        if not values:
            return None
        vals = np.asarray(values, dtype=float)
        vars_ = np.maximum(np.asarray(variances, dtype=float), 1e-12)
        n = len(vals)

        center = float(np.median(vals))
        mad = 1.4826 * float(np.median(np.abs(vals - center))) if n > 1 else 0.0
        sensor_floor = math.sqrt(float(np.median(vars_)))
        scale = max(mad, sensor_floor, 1e-12)

        # Huber-style robust factor around a median seed. The source precision
        # still comes exclusively from its propagated variance.
        denom = np.sqrt(vars_ + scale * scale)
        u = np.abs(vals - center) / np.maximum(denom, 1e-12)
        delta = 2.5
        robust = np.ones(n, dtype=float)
        mask = u > delta
        robust[mask] = delta / np.maximum(u[mask], 1e-12)

        precision = robust / vars_
        psum = float(np.sum(precision))
        if not math.isfinite(psum) or psum <= 0.0:
            return None
        fused = float(np.sum(precision * vals) / psum)

        # Independent precision result, plus conservative floors for common
        # process correlation and actual inter-source disagreement.
        independent_var = 1.0 / psum
        median_var = float(np.median(vars_))
        p2 = float(np.sum(precision * precision))
        n_eff = (psum * psum / p2) if p2 > 0.0 else 1.0
        disagreement = float(np.sum(precision * (vals - fused) ** 2) / psum)
        disagreement_mean_var = disagreement / max(n_eff, 1.0)
        fused_var = max(
            independent_var,
            0.25 * median_var,
            disagreement_mean_var,
            1e-12,
        )

        components = {}
        if labels is not None:
            for i, label in enumerate(labels):
                # Diagnostic source residual relative to the fused ensemble.
                resid_var = max(float(vars_[i]) + fused_var, 1e-12)
                z = abs(float(vals[i]) - fused) / math.sqrt(resid_var)
                components[label] = {
                    "value": float(vals[i]),
                    "variance": float(vars_[i]),
                    "z_score": float(z),
                    "robust_weight": float(robust[i]),
                }
        return fused, fused_var, components

    def _current_observed_snapshot(self, now, *, with_components=False):
        """Return a direct raw/fused observation with *no* state transport.

        This is intentionally independent of x/v/a/j.  It is used only as a
        safety reference/output fallback so a bad latent trajectory cannot
        manufacture the evidence that is supposed to reject that trajectory.
        Source bias/noise calibration and the normal freshness rules are still
        respected.
        """
        if self._source_cal is None:
            return None
        values, variances, labels = [], [], []
        tau = self._window_tau()
        mode = self._runtime_noise_mode
        for src, (t_src, raw) in self._source_cal.cache.items():
            cal = self._calibrations.get(src)
            if cal is None:
                continue
            age = max(float(now) - float(t_src), 0.0)
            max_age = max(
                FRESHNESS_MEDIAN_DT_MULTIPLIER * cal.median_dt,
                FRESHNESS_TAU_FRACTION * tau,
                FRESHNESS_MIN_S,
            )
            if age > max_age:
                continue
            corrected = float(raw) - float(cal.bias)
            variance = cal.variance(
                corrected,
                noise_mode=mode,
                level_fraction=self._runtime_level_fraction,
            )
            if not (math.isfinite(corrected) and math.isfinite(variance)):
                continue
            values.append(corrected)
            variances.append(max(float(variance), NUMERIC_VARIANCE_FLOOR))
            labels.append(src)
        if len(values) < self.min_sources:
            return None
        fused = self._robust_fuse(values, variances, labels)
        if fused is None:
            return None
        z, var, components = fused
        if with_components:
            return z, var, components
        return z, var

    def _update_raw_fallback(self, now: float, *, regime_recovered: bool = False):
        """Update publication-only raw/fused fallback with hysteresis.

        Entry requires a gross level disagreement plus corroborating evidence:
        either a persistent regime-change candidate or collapsed derivative
        plausibility.  This avoids turning one isolated raw outlier into a
        published spike.  Exit requires sustained agreement, except after a
        confirmed level re-anchor where the latent state is already reset to
        the observation and fallback can end immediately.
        """
        obs = self._current_observed_snapshot(now)
        if obs is None:
            return
        observed, observed_var = obs
        self._fallback_value = float(observed)
        self._fallback_variance = max(float(observed_var), NUMERIC_VARIANCE_FLOOR)

        finite_state = (
            np.all(np.isfinite(self.filter.x))
            and np.all(np.isfinite(self.filter.P))
            and math.isfinite(float(self.filter.x[0]))
        )
        if finite_state:
            level_var = max(float(self.filter.P[0, 0]), NUMERIC_VARIANCE_FLOOR)
            denom = math.sqrt(max(level_var + self._fallback_variance, NUMERIC_VARIANCE_FLOOR))
            level_z = abs(float(self.filter.x[0]) - self._fallback_value) / denom
        else:
            level_z = float('inf')
        self._fallback_last_z = float(level_z)

        weights = np.ones(len(self.filter.x), dtype=float)
        try:
            if hasattr(self.filter.state_model, 'effective_weights'):
                weights = self.filter.state_model.effective_weights(
                    self.filter.x, self.filter.P, max(float(getattr(self, '_level_grid_step', 1.0)), 1e-6)
                )
        except Exception:
            pass
        derivative_collapsed = (
            len(weights) > 1
            and float(np.min(weights[1:])) <= self._fallback_derivative_weight_threshold
        )
        persistent_jump = self._regime_candidate_count >= 2

        if regime_recovered and finite_state:
            # recover_level_jump() snapped level and cleared old derivatives.
            # If it is now close to observation, returning to Bayesian output is
            # safe immediately and avoids an unnecessary raw plateau.
            if level_z <= self._fallback_exit_z:
                self._fallback_active = False
                self._fallback_safe_count = 0
                self._fallback_reason = None
                return

        unsafe = (
            not finite_state
            or (
                level_z >= self._fallback_enter_z
                and (persistent_jump or derivative_collapsed)
            )
        )

        if not self._fallback_active:
            if unsafe:
                self._fallback_active = True
                self._fallback_safe_count = 0
                self._fallback_entries += 1
                if not finite_state:
                    reason = 'nonfinite_state'
                elif persistent_jump and derivative_collapsed:
                    reason = 'level_divergence+regime_candidate+derivative_collapse'
                elif persistent_jump:
                    reason = 'level_divergence+regime_candidate'
                else:
                    reason = 'level_divergence+derivative_collapse'
                self._fallback_reason = reason
                self._fallback_last_event = {
                    'timestamp': float(now),
                    'z': float(level_z),
                    'reason': reason,
                }
            return

        # Fallback is already active. Stay conservative until latent level and
        # direct observation agree for several consecutive source updates.
        if finite_state and level_z <= self._fallback_exit_z:
            self._fallback_safe_count += 1
            if self._fallback_safe_count >= self._fallback_exit_confirmations:
                self._fallback_active = False
                self._fallback_safe_count = 0
                self._fallback_reason = None
        else:
            self._fallback_safe_count = 0

    def _published_level(self, bayes_level: float) -> float:
        if self._fallback_active and self._fallback_value is not None:
            return float(self._fallback_value)
        return float(bayes_level)

    def _current_fused_snapshot(self, now, *, with_components=False):
        if self._source_cal is None:
            return None
        values, variances, labels = [], [], []
        tau = self._window_tau()
        mode = self._runtime_noise_mode
        for src, (t_src, raw) in self._source_cal.cache.items():
            cal = self._calibrations.get(src)
            if cal is None:
                continue
            age = max(float(now) - float(t_src), 0.0)
            max_age = max(
                FRESHNESS_MEDIAN_DT_MULTIPLIER * cal.median_dt,
                FRESHNESS_TAU_FRACTION * tau,
                FRESHNESS_MIN_S,
            )
            if age > max_age:
                continue
            corrected = float(raw) - float(cal.bias)
            base_var = cal.variance(
                corrected,
                noise_mode=mode,
                level_fraction=self._runtime_level_fraction,
            )

            # A source is not stale merely because it has not published again
            # before its normal reporting cadence.  This matters especially for
            # event-driven/battery sensors (for example Aqara) and also removes
            # sub-second precision modulation between nominally 1 Hz sources.
            # Only the age *beyond* the source's typical reporting interval is
            # propagated into extra prediction uncertainty.
            cadence_s = max(float(cal.median_dt), 0.0)
            excess_age = max(0.0, age - cadence_s)
            value_now, age_var = self._transport_source_to_now(corrected, excess_age)
            eff_var = max(float(base_var) + float(age_var), 1e-12)
            if not (math.isfinite(value_now) and math.isfinite(eff_var)):
                continue
            values.append(value_now)
            variances.append(eff_var)
            labels.append(src)
        if len(values) < self.min_sources:
            return None
        fused = self._robust_fuse(values, variances, labels)
        if fused is None:
            return None
        z, var, components = fused
        if with_components:
            return z, var, components
        return z, var

    def _edge_recovery_hints(self):
        """Return independent v/a/j witness estimates suitable for recovery.

        A witness is admitted by observability and internal multiscale
        consistency, not by historical derivative magnitude.  A real novel
        process may legitimately lie far outside the training distribution;
        historical sigma measures novelty, not correctness.
        """
        consensus = self._edge_derivative_consensus
        if not consensus:
            return {}

        hints = {}
        candidates = {}
        rate_consensus = consensus.get("rate")
        curvature_consensus = consensus.get("curvature")
        jerk_consensus = consensus.get("jerk")

        if (
            rate_consensus is not None
            and int(rate_consensus.estimates) >= 2
            and float(rate_consensus.worst_disagreement_z) <= 2.0
        ):
            candidates[1] = (rate_consensus.value, rate_consensus.sigma)

        if (
            curvature_consensus is not None
            and int(curvature_consensus.estimates) >= 2
            and float(curvature_consensus.worst_disagreement_z) <= 2.0
        ):
            candidates[2] = (curvature_consensus.value, curvature_consensus.sigma)

        if (
            jerk_consensus is not None
            and int(jerk_consensus.estimates) >= 2
            and float(jerk_consensus.worst_disagreement_z) <= 2.0
        ):
            candidates[3] = (jerk_consensus.value, jerk_consensus.sigma)

        for order, pair in candidates.items():
            try:
                value = float(pair[0])
                sigma = float(pair[1])
            except (TypeError, ValueError, IndexError):
                continue
            if not (
                math.isfinite(value)
                and math.isfinite(sigma)
                and sigma > 0.0
            ):
                continue
            hints[order] = (value, sigma)

        return hints


    def _maybe_reset_persistently_untrusted_derivative_tail(
        self, t: float | None = None, *, count_update: bool = True
    ):
        """Hard-reset a derivative tail only after a sustained 5-sigma failure.

        The normal derivative plausibility gate remains the fast protection
        mechanism.  Recovery is destructive and therefore requires the first
        implausible derivative in the hierarchy to remain outside the
        history-trained 5-sigma envelope for BOTH:

        * ``_derivative_recovery_confirm_updates`` observations; and
        * one ``gated_dynamics.timescale_s``.

        A candidate is cancelled once it returns below 4 sigma.  Values between
        4 and 5 sigma keep an existing candidate alive but do not start a new
        one, which provides hysteresis around the entry boundary.
        """
        model = self.filter.state_model
        dim = int(model.dim_x())
        now = float(
            self.filter.t_last if t is None and self.filter.t_last is not None
            else (0.0 if t is None else t)
        )

        params = model.derivative_plausibility_parameters()
        scales = list(params.get("scales", []) or [])

        enter_z = float(self._derivative_recovery_enter_sigma)
        cancel_z = float(self._derivative_recovery_cancel_sigma)
        required_updates = max(1, int(self._derivative_recovery_confirm_updates))

        # This is the trained process timescale requested for recovery timing.
        process_timescale = None
        if self._gated_dynamics is not None:
            try:
                candidate_T = float(self._gated_dynamics.timescale_s)
                if math.isfinite(candidate_T) and candidate_T > 0.0:
                    process_timescale = candidate_T
            except Exception:
                pass
        if process_timescale is None:
            try:
                candidate_T = float(self.filter.tau)
                if math.isfinite(candidate_T) and candidate_T > 0.0:
                    process_timescale = candidate_T
            except Exception:
                pass
        if process_timescale is None:
            return None

        required_duration = (
            float(self._derivative_recovery_timescale_factor) * process_timescale
        )

        first_bad = None
        first_bad_z = None

        # Hierarchical semantics: only the first currently bad derivative can
        # initiate recovery; all higher orders belong to the same discarded
        # tail and are not independent recovery candidates.
        for order in range(1, dim):
            scale = float(scales[order]) if order < len(scales) else float("nan")
            value = float(self.filter.x[order])

            if not math.isfinite(scale) or scale <= 0.0 or not math.isfinite(value):
                self._fallback_derivative_weight_counts[order] = 0
                self._derivative_recovery_bad_since[order] = None
                continue

            z = abs(value) / scale
            since = self._derivative_recovery_bad_since.get(order)

            if since is None:
                if z >= enter_z:
                    first_bad = order
                    first_bad_z = z
                    self._derivative_recovery_bad_since[order] = now
                    self._fallback_derivative_weight_counts[order] = 1 if count_update else 0
                else:
                    self._fallback_derivative_weight_counts[order] = 0
                # If this derivative is healthy, continue looking higher.
                if z < enter_z:
                    continue
                break

            # Existing candidate: cancel only after returning below 4 sigma.
            if z < cancel_z:
                self._derivative_recovery_bad_since[order] = None
                self._fallback_derivative_weight_counts[order] = 0
                continue

            first_bad = order
            first_bad_z = z
            if count_update:
                self._fallback_derivative_weight_counts[order] = (
                    int(self._fallback_derivative_weight_counts.get(order, 0)) + 1
                )
            break

        # Any higher-order candidates are invalid whenever a lower-order
        # derivative is the active first bad member of the cascade.
        if first_bad is not None:
            for order in range(first_bad + 1, dim):
                self._derivative_recovery_bad_since[order] = None
                self._fallback_derivative_weight_counts[order] = 0

        if first_bad is None:
            return None

        since = self._derivative_recovery_bad_since.get(first_bad)
        elapsed = max(0.0, now - float(since if since is not None else now))
        updates = int(self._fallback_derivative_weight_counts.get(first_bad, 0))

        if updates < required_updates or elapsed < required_duration:
            return None

        # A valid edge witness makes historical 5-sigma a novelty flag only.
        # For any derivative with a valid witness, correctness is decided by Bayes<->edge
        # disagreement, so defer to the edge watchdog instead of resetting a
        # potentially real new process.
        edge_hints = self._edge_recovery_hints()
        if first_bad in edge_hints:
            self._fallback_derivative_weight_counts[first_bad] = 0
            self._derivative_recovery_bad_since[first_bad] = None
            return None

        # Prefer an independent causal edge witness when it has passed its
        # own observability, multiscale-consistency and historical-plausibility
        # gates.  This gives the repaired tail a data-derived local state
        # instead of repeatedly dropping it onto the same zero prior.  If no
        # witness is trustworthy, retain the conservative zero-prior fallback.
        hints = self._edge_recovery_hints()
        if hints:
            recovery = self.filter.recondition_derivatives_from_hints(
                start_order=first_bad,
                derivative_hints=hints,
                reason="persistent_5sigma_timescale_edge_recondition",
            )
        else:
            # z>=5 maps to zero plausibility weight in the production model, so
            # a zero threshold resets exactly the confirmed bad tail.
            recovery = self.filter.reset_untrusted_derivative_tail(
                weight_threshold=float(self._fallback_derivative_weight_threshold)
            )
        if recovery is None:
            return None

        for order in range(first_bad, dim):
            self._fallback_derivative_weight_counts[order] = 0
            self._derivative_recovery_bad_since[order] = None

        recovery["confirm_updates"] = required_updates
        recovery["confirmed_from_order"] = int(first_bad)
        recovery["trigger_z"] = float(first_bad_z)
        recovery["confirm_duration_s"] = float(required_duration)
        recovery["candidate_elapsed_s"] = float(elapsed)
        recovery["process_timescale_s"] = float(process_timescale)
        recovery["timescale_factor"] = float(self._derivative_recovery_timescale_factor)
        return recovery


    def _record_derivative_recondition(self, t, recovery, *, reason=None):
        """Persist/log one derivative-tail repair in a common format."""
        if recovery is None:
            return
        self._derivative_recondition_count += 1
        names = {1: "rate", 2: "curvature", 3: "jerk"}
        start_order = int(recovery.get("from_order", 1))
        applied_hints = dict(recovery.get("applied_hints", {}) or {})
        why = str(reason or recovery.get("reason") or "plausibility_rejection")
        self._last_derivative_recondition = {
            "timestamp": float(t),
            "from_order": start_order,
            "from_name": names.get(start_order, str(start_order)),
            "trigger_z": float(recovery.get("trigger_z", float("nan"))),
            "old_derivatives": list(recovery.get("old_derivatives", []) or []),
            "restored_sigmas": dict(recovery.get("restored_sigmas", {}) or {}),
            "applied_hints": applied_hints,
            "reason": why,
            "candidate_elapsed_s": recovery.get("candidate_elapsed_s"),
            "confirm_duration_s": recovery.get("confirm_duration_s"),
            "process_timescale_s": recovery.get("process_timescale_s"),
            "timescale_factor": recovery.get("timescale_factor"),
        }
        _LOGGER.warning(
            "Bayesian State Filter derivative tail reconditioned from %s: "
            "reason=%s edge_hints=%s",
            names.get(start_order, start_order), why, sorted(applied_hints),
        )

    def _startup_reconcile_derivatives_from_edge(self):
        """One-shot startup repair only when Bayes actually disagrees with the witness.

        A valid causal witness is *evidence*, not an instruction to overwrite a
        healthy checkpoint.  This matters for slow/noisy processes (radiation is
        the canonical example): the endpoint fit may be perfectly valid yet
        merely represent a harmless local slope different from an already
        converged Bayesian derivative state.

        Startup therefore uses the same independent-witness divergence concept
        as the live watchdog, but without the three-update persistence delay:

        * witness must first pass N+span observability, multiscale consistency
          and internal quality gates in ``_edge_recovery_hints``;
        * Bayes must then differ by more than ``_edge_divergence_sigma`` witness
          sigmas for rate, curvature or jerk;
        * only the first divergent order and the derivative tail above it are
          hard reconditioned from the corresponding multiscale witness.

        Divergence is measured against the witness' own multiscale
        uncertainty.  Historical robust sigma describes novelty only. A
        healthy slow process therefore remains
        untouched when Bayes and witness agree, while a runaway derivative is
        not allowed to hide behind a deliberately inflated reference sigma.
        """
        hints = self._edge_recovery_hints()
        if not hints:
            return None

        trigger_order = None
        trigger_z = None
        for order in (1, 2, 3):
            hint = hints.get(order)
            if hint is None or order >= len(self.filter.x):
                self._edge_last_divergence_z[order] = 0.0
                continue
            witness_value, witness_sigma = float(hint[0]), float(hint[1])
            reference_sigma = max(
                witness_sigma, math.sqrt(NUMERIC_VARIANCE_FLOOR)
            )
            z = abs(float(self.filter.x[order]) - witness_value) / reference_sigma
            self._edge_last_divergence_z[order] = float(z)
            if math.isfinite(z) and z > float(self._edge_divergence_sigma):
                trigger_order = order
                trigger_z = float(z)
                break

        if trigger_order is None:
            return None

        recovery = self.filter.recondition_derivatives_from_hints(
            start_order=trigger_order,
            derivative_hints=hints,
            reason="startup_edge_witness",
        )
        if recovery is not None and trigger_z is not None:
            recovery["trigger_z"] = float(trigger_z)
        return recovery

    @staticmethod
    def _edge_agreement_weight(z: float) -> float:
        """Map Bayes<->edge disagreement to a local coupling weight.

        Full trust inside one witness sigma, then a shifted Gaussian tail,
        with a hard five-sigma rejection:

            w = 1                              z <= 1
            w = exp(-0.5 * (z - 1)^2)         1 < z < 5
            w = 0                              z >= 5
        """
        z = float(z)
        if not math.isfinite(z):
            return 0.0
        if z <= 1.0:
            return 1.0
        if z >= 5.0:
            return 0.0
        return float(math.exp(-0.5 * (z - 1.0) ** 2))

    def _update_edge_agreement_weights(self):
        """Update witness-based local v/a/j coupling weights.

        A valid multiscale edge witness overrides historical magnitude gating
        for the corresponding derivative. If no valid witness exists, the
        state model receives no override and falls back to history.
        """
        model = self.filter.state_model
        if not hasattr(model, "set_external_derivative_weights"):
            return

        hints = self._edge_recovery_hints()
        external = {}
        for order in (1, 2, 3):
            hint = hints.get(order)
            if hint is None or order >= len(self.filter.x):
                self._edge_agreement_z[order] = None
                self._edge_agreement_local_weight[order] = None
                continue

            witness_value, witness_sigma = float(hint[0]), float(hint[1])
            reference_sigma = max(
                witness_sigma, math.sqrt(NUMERIC_VARIANCE_FLOOR)
            )
            z = abs(float(self.filter.x[order]) - witness_value) / reference_sigma
            weight = self._edge_agreement_weight(z)
            self._edge_agreement_z[order] = float(z)
            self._edge_agreement_local_weight[order] = float(weight)
            external[order] = float(weight)

        model.set_external_derivative_weights(external)

    def _runtime_edge_divergence_recovery(self):
        """Recover only from sustained Bayes<->edge disagreement.

        ENTER grace at >5 witness sigma.
        CANCEL grace at <3 witness sigma.
        The 3..5 sigma band preserves the current grace state.

        Recovery requires BOTH at least five fresh independent edge updates
        above the 5-sigma entry boundary and at least one trained process
        timescale since grace began.  The effective grace is therefore
        max(time for five edge updates, process_timescale_s).
        """
        hints = self._edge_recovery_hints()
        if not hints:
            for order in self._edge_divergence_counts:
                self._edge_divergence_counts[order] = 0
                self._edge_divergence_bad_since[order] = None
                self._edge_last_divergence_z[order] = 0.0
            return None

        if self._gated_dynamics is not None:
            process_timescale = float(self._gated_dynamics.timescale_s)
        else:
            process_timescale = float(self.filter.tau)
        if not math.isfinite(process_timescale) or process_timescale <= 0.0:
            return None

        now = float(
            self._edge_last_update_ts
            if self._edge_last_update_ts is not None
            else (self.filter.t_last or 0.0)
        )
        enter_sigma = float(self._edge_divergence_sigma)
        cancel_sigma = float(self._edge_divergence_cancel_sigma)
        required_updates = max(1, int(self._edge_divergence_confirm_updates))
        required_duration = float(process_timescale)

        trigger_order = None
        trigger_z = None

        for order in (1, 2, 3):
            hint = hints.get(order)
            if hint is None or order >= len(self.filter.x):
                self._edge_divergence_counts[order] = 0
                self._edge_divergence_bad_since[order] = None
                self._edge_last_divergence_z[order] = 0.0
                continue

            witness_value, witness_sigma = float(hint[0]), float(hint[1])
            bayes_value = float(self.filter.x[order])
            reference_sigma = max(
                witness_sigma, math.sqrt(NUMERIC_VARIANCE_FLOOR)
            )
            z = abs(bayes_value - witness_value) / reference_sigma
            self._edge_last_divergence_z[order] = float(z)

            since = self._edge_divergence_bad_since.get(order)

            if z < cancel_sigma:
                # Bayes has returned to agreement during grace.
                self._edge_divergence_counts[order] = 0
                self._edge_divergence_bad_since[order] = None
                continue

            if since is None:
                if z > enter_sigma:
                    self._edge_divergence_bad_since[order] = now
                    self._edge_divergence_counts[order] = 1
                else:
                    # 3..5 sigma cannot start grace.
                    continue
            else:
                if z > enter_sigma:
                    self._edge_divergence_counts[order] += 1
                # 3..5 sigma: keep grace alive, but do not count another
                # strong independent confirmation.

            since = self._edge_divergence_bad_since.get(order)
            elapsed = max(
                0.0,
                now - float(since if since is not None else now),
            )

            if (
                self._edge_divergence_counts[order] >= required_updates
                and elapsed >= required_duration
            ):
                trigger_order = order
                trigger_z = float(z)
                break

        if trigger_order is None:
            return None

        recovery = self.filter.recondition_derivatives_from_hints(
            start_order=trigger_order,
            derivative_hints=hints,
            reason="persistent_edge_divergence",
        )
        if recovery is not None:
            since = self._edge_divergence_bad_since.get(trigger_order)
            recovery["trigger_z"] = trigger_z
            recovery["confirm_updates"] = required_updates
            recovery["confirm_duration_s"] = required_duration
            recovery["candidate_elapsed_s"] = max(
                0.0,
                now - float(since if since is not None else now),
            )
            recovery["process_timescale_s"] = float(process_timescale)
            recovery["timescale_factor"] = 1.0

        for order in self._edge_divergence_counts:
            self._edge_divergence_counts[order] = 0
            self._edge_divergence_bad_since[order] = None

        return recovery


    def _seed_edge_history_from_level_history(self):
        """Prime the edge witness from fused history after a restart.

        Startup observability must satisfy BOTH constraints of the longest
        witness: at least 55 points and at least 2 process timescales of span.
        This matters for sparse sources: selecting only by elapsed span can
        leave too few points even when Recorder already contains enough history.
        """
        if not self._level_history:
            return

        if self._gated_dynamics is not None:
            tau_s = float(self._gated_dynamics.timescale_s)
        else:
            tau_s = float(self.filter.tau)
        if not math.isfinite(tau_s) or tau_s <= 0.0:
            tau_s = 1.0
        tau_s = max(tau_s, 1.0)

        required_span_s = 2.0 * tau_s
        required_points = 55

        rows = [
            (float(t), float(z), max(float(var), NUMERIC_VARIANCE_FLOOR))
            for t, z, var in self._level_history
            if math.isfinite(float(t))
            and math.isfinite(float(z))
            and math.isfinite(float(var))
            and float(var) > 0.0
        ]
        if not rows:
            return

        rows.sort(key=lambda x: x[0])
        newest = rows[-1][0]

        # Satisfy both constraints.  Start far enough back for 2*tau and, for
        # sparse data, at least as far back as the 55th point from the end.
        span_cutoff = newest - max(required_span_s, 1.0)
        if len(rows) >= required_points:
            count_cutoff = float(rows[-required_points][0])
            cutoff = min(span_cutoff, count_cutoff)
        else:
            cutoff = span_cutoff

        first = 0
        for i, row in enumerate(rows):
            if float(row[0]) >= cutoff:
                first = i
                break
        tail = rows[first:]

        # Discrete timestamps can make the first selected sample a fraction of
        # a cadence too new for the exact 2*tau requirement. Extend backwards
        # until both observability conditions are truly satisfied.
        while first > 0 and (
            len(tail) < required_points
            or newest - float(tail[0][0]) < required_span_s
        ):
            first -= 1
            tail.insert(0, rows[first])

        # Keep the hot-path buffer bounded. If Recorder history is dense,
        # uniform decimation preserves both endpoints and therefore the span.
        max_keep = self._edge_history.maxlen or 4096
        if len(tail) > max_keep:
            keep = max(required_points, int(max_keep))
            idx = np.linspace(0, len(tail) - 1, keep, dtype=int)
            tail = [tail[int(i)] for i in idx]

        self._edge_history.clear()
        self._edge_history.extend(tail)
        self._recompute_edge_witness()
        if self.filter.t_last is not None:
            self._update_edge_agreement_weights()


    def _edge_history_observable(self) -> bool:
        """Return True when the latest causal edge segment satisfies the long gate."""
        self._trim_edge_history_to_latest_segment()
        if len(self._edge_history) < 55:
            return False
        tau_s = self._edge_process_timescale_s()
        span_s = float(self._edge_history[-1][0]) - float(self._edge_history[0][0])
        return math.isfinite(span_s) and span_s >= 2.0 * tau_s

    async def _backfill_edge_history_from_recorder(self) -> bool:
        """Build missing edge history directly from recent Recorder source rows.

        This is a one-time compatibility/backfill path for checkpoints created
        before edge_history became persistent, or for an otherwise insufficient
        saved witness buffer. It reproduces the live direct-observation witness:
        calibrated source values, normal freshness gates, robust fusion, and
        witness cadence throttling, but never transports values with latent
        Bayesian derivatives.
        """
        if self._edge_history_observable():
            self._recompute_edge_witness()
            if self.filter.t_last is not None:
                self._update_edge_agreement_weights()
            return True

        if not self._calibrations:
            return False

        if self._gated_dynamics is not None:
            tau_s = float(self._gated_dynamics.timescale_s)
        else:
            tau_s = float(self.filter.tau)
        if not math.isfinite(tau_s) or tau_s <= 0.0:
            return False

        cadences = [
            float(cal.median_dt)
            for cal in self._calibrations.values()
            if math.isfinite(float(cal.median_dt)) and float(cal.median_dt) > 0.0
        ]
        median_dt = float(statistics.median(cadences)) if cadences else 1.0

        # Longest witness needs 55 points and 2*tau. Ask for a modest margin on
        # both constraints so Recorder timestamp jitter/freshness does not leave
        # us one sample short.
        required_horizon_s = max(
            2.2 * tau_s,
            1.2 * 55.0 * median_dt,
            180.0,
        )
        end_ts = float(self.filter.t_last or datetime.now(timezone.utc).timestamp())
        start_ts = max(0.0, end_ts - required_horizon_s)

        histories = {}
        for src in self.sources:
            seq = await self._fetch_history(src, start_ts=start_ts)
            parsed = self._parse_states(seq)
            if parsed:
                histories[src] = parsed
        if not histories:
            return False

        events = []
        for src, seq in histories.items():
            for t, raw in seq:
                if t <= end_ts + 1e-6:
                    events.append((float(t), src, float(raw)))
        events.sort(key=lambda x: (x[0], x[1]))
        if not events:
            return False

        cache = {}
        rebuilt = []
        min_dt = max(0.80 * median_dt, 0.05)
        last_edge_t = None
        tau_fresh = self._window_tau()
        mode = self._runtime_noise_mode

        for t, src, raw in events:
            cache[src] = (t, raw)
            if last_edge_t is not None and t - last_edge_t < min_dt:
                continue

            values, variances, labels = [], [], []
            for entity_id, (t_src, raw_src) in cache.items():
                cal = self._calibrations.get(entity_id)
                if cal is None:
                    continue
                age = max(t - float(t_src), 0.0)
                max_age = max(
                    FRESHNESS_MEDIAN_DT_MULTIPLIER * cal.median_dt,
                    FRESHNESS_TAU_FRACTION * tau_fresh,
                    FRESHNESS_MIN_S,
                )
                if age > max_age:
                    continue

                corrected = float(raw_src) - float(cal.bias)
                variance = cal.variance(
                    corrected,
                    noise_mode=mode,
                    level_fraction=self._runtime_level_fraction,
                )
                if not (math.isfinite(corrected) and math.isfinite(variance)):
                    continue
                values.append(corrected)
                variances.append(max(float(variance), NUMERIC_VARIANCE_FLOOR))
                labels.append(entity_id)

            if len(values) < self.min_sources:
                continue
            fused = self._robust_fuse(values, variances, labels)
            if fused is None:
                continue
            z, var, _components = fused
            rebuilt.append(
                (float(t), float(z), max(float(var), NUMERIC_VARIANCE_FLOOR))
            )
            last_edge_t = float(t)

        if not rebuilt:
            return False

        # Merge with any persisted rows, deduplicate by timestamp, and retain a
        # bounded causal tail.
        merged = {}
        for row in list(self._edge_history) + rebuilt:
            merged[float(row[0])] = (
                float(row[0]),
                float(row[1]),
                max(float(row[2]), NUMERIC_VARIANCE_FLOOR),
            )
        rows = [merged[k] for k in sorted(merged)]
        max_keep = self._edge_history.maxlen or 4096
        rows = rows[-max_keep:]

        self._edge_history.clear()
        self._edge_history.extend(rows)
        if self._edge_history:
            self._edge_last_update_ts = float(self._edge_history[-1][0])

        self._recompute_edge_witness()
        if self.filter.t_last is not None:
            self._update_edge_agreement_weights()

        return self._edge_history_observable()

    @staticmethod
    def _edge_fit_points(history, min_span_s, min_points, max_fit_points=96):
        """Return a bounded causal tail satisfying BOTH N and elapsed span.

        The selected window must include at least ``min_points`` observations
        and reach back at least ``min_span_s`` from the newest observation.
        Sparse sources are therefore count-limited while dense sources are
        span-limited.
        """
        rows = list(history)
        if not rows:
            return []

        newest = float(rows[-1][0])
        cutoff = newest - max(float(min_span_s), 0.0)

        # Earliest index required by temporal span.
        first_by_span = len(rows) - 1
        for i in range(len(rows) - 1, -1, -1):
            first_by_span = i
            if float(rows[i][0]) <= cutoff:
                break

        # Earliest index required by sample count.
        first_by_count = max(0, len(rows) - max(int(min_points), 1))

        # To satisfy BOTH constraints, start at the earlier of the two.
        first = min(first_by_span, first_by_count)
        tail = rows[first:]

        if len(tail) <= max_fit_points:
            return tail

        # Downsample only after the required causal tail has been selected.
        # Preserve both endpoints so the temporal span cannot collapse.
        keep = max(int(max_fit_points), int(min_points))
        idx = np.linspace(0, len(tail) - 1, keep, dtype=int)
        return [tail[int(i)] for i in idx]


    def _edge_process_timescale_s(self) -> float:
        """Return the physical process timescale used to segment edge history."""
        if self._gated_dynamics is not None:
            tau_s = float(self._gated_dynamics.timescale_s)
        else:
            tau_s = float(self.filter.tau)
        if not math.isfinite(tau_s) or tau_s <= 0.0:
            return 1.0
        return max(tau_s, 1.0)

    def _reset_edge_segment_state(self):
        """Drop edge-derived trust/recovery state at a causal observation gap."""
        self._edge_witness_scales = {}
        self._edge_derivative_consensus = {}
        self._edge_rate_consensus = None
        self._edge_curvature_consensus = None
        self._edge_jerk_witness_scales = {}
        self._edge_jerk_consensus = None
        self._edge_derivative_estimate = None

        for order in self._edge_divergence_counts:
            self._edge_divergence_counts[order] = 0
            self._edge_divergence_bad_since[order] = None
            self._edge_last_divergence_z[order] = 0.0
            self._edge_agreement_z[order] = None
            self._edge_agreement_local_weight[order] = None

        # A historical hard-recovery candidate must not accumulate elapsed
        # time through an interval in which the process was unobserved.
        for order in self._fallback_derivative_weight_counts:
            self._fallback_derivative_weight_counts[order] = 0
            self._derivative_recovery_bad_since[order] = None

        model = self.filter.state_model
        if hasattr(model, "set_external_derivative_weights"):
            model.set_external_derivative_weights({})

    def _trim_edge_history_to_latest_segment(self):
        """Keep only the latest contiguous edge segment.

        A gap longer than one process timescale means the derivative trajectory
        across the missing interval is unobserved. No local polynomial may span
        such a gap.
        """
        if len(self._edge_history) < 2:
            return False

        gap_limit = self._edge_process_timescale_s()
        rows = list(self._edge_history)
        split_at = None
        for i in range(len(rows) - 1, 0, -1):
            gap = float(rows[i][0]) - float(rows[i - 1][0])
            if math.isfinite(gap) and gap > gap_limit:
                split_at = i
                break

        if split_at is None:
            return False

        tail = rows[split_at:]
        self._edge_history.clear()
        self._edge_history.extend(tail)
        self._reset_edge_segment_state()
        return True

    def _append_edge_snapshot(self, now):
        """Update independent causal derivative witnesses.

        Returns True only when a fresh direct/fused observation was actually
        appended.  Runtime divergence persistence is counted on these witness
        updates rather than on every source event, so a dense multi-source
        ensemble cannot manufacture three confirmations in a fraction of a
        second.

        A derivative is observable only when two independent conditions hold:
        enough points *and* enough elapsed physical time.  The three nested
        witnesses require respectively 25/40/55 points and at least
        0.5/1/2 process timescales.  At startup the buffer is primed from
        fused Recorder history, so a restart itself does not erase derivative
        context.
        """
        snap = self._current_observed_snapshot(now)
        if snap is None:
            return False
        z, var = snap

        cadences = [
            float(cal.median_dt)
            for cal in self._calibrations.values()
            if math.isfinite(float(cal.median_dt)) and float(cal.median_dt) > 0.0
        ]
        natural_dt = float(statistics.median(cadences)) if cadences else 1.0
        min_dt = max(0.80 * natural_dt, 0.05)

        if self._edge_history:
            last_t = float(self._edge_history[-1][0])
            if float(now) <= last_t:
                return False

            gap_s = float(now) - last_t
            if gap_s > self._edge_process_timescale_s():
                # Missing observations for longer than the physical process
                # timescale terminate the causal derivative segment.
                self._edge_history.clear()
                self._reset_edge_segment_state()
            elif gap_s < min_dt:
                return False

        self._edge_history.append(
            (float(now), float(z), max(float(var), NUMERIC_VARIANCE_FLOOR))
        )
        self._edge_last_update_ts = float(now)

        self._recompute_edge_witness()
        return True

    def _recompute_edge_witness(self):
        """Recompute causal multiscale witnesses from the latest causal segment."""
        self._trim_edge_history_to_latest_segment()
        if not self._edge_history:
            self._edge_witness_scales = {}
            self._edge_derivative_consensus = {}
            self._edge_rate_consensus = None
            self._edge_curvature_consensus = None
            self._edge_jerk_witness_scales = {}
            self._edge_jerk_consensus = None
            return

        tau_s = self._edge_process_timescale_s()
        self._edge_witness_tau_s = tau_s

        specs = {
            "short":  {"min_points": 25, "min_span_s": 0.5 * tau_s, "max_order": 1},
            "medium": {"min_points": 40, "min_span_s": 1.0 * tau_s, "max_order": 2},
            "long":   {"min_points": 55, "min_span_s": 2.0 * tau_s, "max_order": 3},
        }

        estimates = {}
        for name, spec in specs.items():
            fit_points = self._edge_fit_points(
                self._edge_history,
                spec["min_span_s"],
                spec["min_points"],
            )
            estimates[name] = robust_causal_local_polynomial(
                fit_points,
                max_points=96,
                max_order=spec["max_order"],
                min_points=spec["min_points"],
                min_span_s=spec["min_span_s"],
                tukey_c=2.5,
                max_iter=6,
            )

        valid = {k: v for k, v in estimates.items() if v is not None}
        self._edge_witness_scales = valid
        self._edge_derivative_estimate = (
            valid.get("medium") or valid.get("long") or valid.get("short")
        )

        # Jerk needs at least two independent scale estimates just like v/a.
        # Preserve the existing linear/quadratic production fits for rate and
        # curvature; compute a separate cubic fit on the medium window and pair
        # it with the existing cubic long-window fit.
        medium_jerk_points = self._edge_fit_points(
            self._edge_history, 1.0 * tau_s, 40
        )
        medium_jerk = robust_causal_local_polynomial(
            medium_jerk_points,
            max_points=96,
            max_order=3,
            min_points=40,
            min_span_s=1.0 * tau_s,
            tukey_c=2.5,
            max_iter=6,
        )
        jerk_valid = {}
        if medium_jerk is not None:
            jerk_valid["medium"] = medium_jerk
        if valid.get("long") is not None:
            jerk_valid["long"] = valid["long"]
        self._edge_jerk_witness_scales = jerk_valid

        # Dataclass consensuses feed recovery hints. Rate/curvature use the
        # existing production windows; jerk uses the two cubic windows above.
        self._edge_derivative_consensus = {}
        for attr, sigma_attr in (
            ("rate", "rate_sigma"),
            ("curvature", "curvature_sigma"),
        ):
            consensus = combine_multiscale_derivative(valid, attr, sigma_attr)
            if consensus is not None:
                self._edge_derivative_consensus[attr] = consensus

        jerk_consensus = combine_multiscale_derivative(
            jerk_valid, "jerk", "jerk_sigma"
        )
        if jerk_consensus is not None:
            self._edge_derivative_consensus["jerk"] = jerk_consensus

        # Keep the legacy dict diagnostics used by _build_attrs().
        def _consensus(attr, sigma_attr, source=None):
            vals = []
            source = valid if source is None else source
            for name in ("short", "medium", "long"):
                est = source.get(name)
                if est is None:
                    continue
                value = getattr(est, attr)
                sigma = getattr(est, sigma_attr)
                if value is None or sigma is None:
                    continue
                value = float(value)
                sigma = float(sigma)
                if not (math.isfinite(value) and math.isfinite(sigma) and sigma > 0.0):
                    continue
                vals.append((name, value, sigma))

            if not vals:
                return None

            precisions = np.asarray([1.0 / (sig * sig) for _, _, sig in vals], dtype=float)
            values = np.asarray([value for _, value, _ in vals], dtype=float)
            psum = float(np.sum(precisions))
            center = float(np.sum(precisions * values) / psum)
            fit_sigma = max(math.sqrt(1.0 / psum), min(sig for _, _, sig in vals))

            if len(vals) >= 2:
                spread_var = float(np.sum(precisions * (values - center) ** 2) / psum)
                scale_sigma = math.sqrt(max(spread_var, 0.0))
                disagreement_z = max(
                    abs(value - center) / math.sqrt(sig * sig + fit_sigma * fit_sigma)
                    for _, value, sig in vals
                )
            else:
                scale_sigma = 0.0
                disagreement_z = 0.0

            total_sigma = math.sqrt(fit_sigma * fit_sigma + scale_sigma * scale_sigma)
            return {
                "value": center,
                "sigma": total_sigma,
                "fit_sigma": fit_sigma,
                "scale_sigma": scale_sigma,
                "disagreement_z": float(disagreement_z),
                "consistent": bool(len(vals) >= 2 and disagreement_z <= 2.0),
                "available": len(vals),
                "used": [name for name, _, _ in vals],
            }

        self._edge_rate_consensus = _consensus("rate", "rate_sigma")
        self._edge_curvature_consensus = _consensus("curvature", "curvature_sigma")
        self._edge_jerk_consensus = _consensus(
            "jerk", "jerk_sigma", source=jerk_valid
        )

    def _append_level_snapshot(self, now):
        snap = self._current_fused_snapshot(now)
        if snap is None:
            return
        if self._level_history:
            last_t = self._level_history[-1][0]
            if now <= last_t:
                return
            if now - last_t < 0.80 * max(self._level_grid_step, 1.0):
                return
        z, var = snap
        self._level_history.append((float(now), float(z), float(var)))
        cutoff = now - self._history_days * 86400.0
        while self._level_history and self._level_history[0][0] < cutoff:
            self._level_history.popleft()

    def _maybe_schedule_bias_history_refit(self, now):
        if len(self.sources) < 2:
            return
        if self._bias_history_task is not None and not self._bias_history_task.done():
            return
        if (self._last_bias_history_refit_ts
                and float(now) - self._last_bias_history_refit_ts < self._bias_history_refit_s):
            return
        self._last_bias_history_refit_ts = float(now)
        self._bias_history_task = self.hass.async_create_task(
            self._refit_bias_from_recorder()
        )

    async def _refit_bias_from_recorder(self):
        """Re-estimate source bias from quiet sections of full Recorder history."""
        try:
            histories = {}
            for entity_id in self.sources:
                seq = await self._fetch_history(entity_id)
                parsed = self._parse_states(seq)
                if parsed:
                    histories[entity_id] = parsed
            if len(histories) < 2:
                return

            biases, stable_points, total_points = await self.hass.async_add_executor_job(
                lambda: estimate_biases_from_history(
                    histories,
                    bias_anchor=self._bias_anchor,
                    source_models=self._source_models,
                    model_accuracy=self._model_accuracy,
                    huber_delta=self._bias_huber_delta,
                )
            )
            if not biases:
                return

            # A long-history refit estimates relative source offsets. Applying
            # those offsets with a new common gauge can create an artificial
            # level step even though the physical process did not move. Keep
            # the current fused corrected level continuous by using the free
            # common bias offset as a gauge adjustment.
            now = datetime.now(timezone.utc).timestamp()
            old_snapshot = self._current_fused_snapshot(now)
            old_biases = {
                src: float(cal.bias)
                for src, cal in self._calibrations.items()
            }

            staged = {}
            for src, bias in biases.items():
                cal = self._calibrations.get(src)
                if cal is not None and math.isfinite(float(bias)):
                    staged[src] = float(bias)
                    cal.bias = float(bias)

            gauge_shift = 0.0
            if staged:
                new_snapshot = self._current_fused_snapshot(now)
                if old_snapshot is not None and new_snapshot is not None:
                    # corrected = raw - bias, so adding this common shift to
                    # every bias subtracts the same amount from the new fused
                    # level and restores the pre-refit level exactly.
                    gauge_shift = float(new_snapshot[0] - old_snapshot[0])
                else:
                    # No contemporaneous fused snapshot (e.g. all sources are
                    # stale). Preserve the previous gauge robustly from the
                    # common component of the bias change.
                    deltas = [
                        old_biases[src] - staged[src]
                        for src in staged
                        if src in old_biases
                    ]
                    if deltas:
                        gauge_shift = float(statistics.median(deltas))

                if math.isfinite(gauge_shift) and gauge_shift != 0.0:
                    for src in staged:
                        self._calibrations[src].bias += gauge_shift

            if self._source_cal is not None:
                # Calibrations are shared objects.  Record the common gauge
                # correction used to keep the live level continuous; relative
                # inter-source bias estimates remain unchanged.
                self._source_cal.last_anchor = BiasAnchorResult(
                    mode=self._bias_anchor, shift=float(gauge_shift)
                )
            _LOGGER.info(
                "Bayesian State Filter bias refit from %.2f d Recorder history: "
                "stable_grid=%d/%d, gauge_shift=%+.8g",
                self._history_days, stable_points, total_points, gauge_shift,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            _LOGGER.exception("Bayesian State Filter long-history bias refit failed")

    def _maybe_schedule_characteristic_fit(self, now):
        if self._characteristic_task is not None and not self._characteristic_task.done():
            return
        if len(self._level_history) < 40:
            return
        if self._level_history[-1][0] - self._level_history[0][0] < 10.0 * max(self._level_grid_step, 1.0):
            return
        if now - self._last_characteristic_fit_ts < self._characteristic_refit_s:
            return
        self._last_characteristic_fit_ts = float(now)
        self._characteristic_task = self.hass.async_create_task(self._refit_characteristic())

    async def _refit_characteristic(self):
        points = list(self._level_history)
        point_count = len(points)
        started = time.perf_counter()
        self._diag_bg["characteristic_runs"] += 1
        self._diag_bg["characteristic_last_points"] = point_count
        failed = False
        try:
            estimate = await self.hass.async_add_executor_job(
                lambda: estimate_characteristic_time(
                    points,
                    tau_points=max(self._tau_points * 3, 32),
                    tau_min_s=self._char_tau_min_s,
                    tau_max_s=self._char_tau_max_s,
                )
            )
            if estimate is not None:
                self._characteristic = estimate
                self._build_attrs(last_out=None)
                if self._state is not None:
                    self.async_write_ha_state()
        except asyncio.CancelledError:
            raise
        except Exception:
            failed = True
            self._diag_bg["characteristic_failures"] += 1
            _LOGGER.exception("Bayesian State Filter characteristic-time refit failed")
        finally:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            self._diag_bg["characteristic_last_ms"] = elapsed_ms
            self._diag_bg["characteristic_max_ms"] = max(
                self._diag_bg["characteristic_max_ms"], elapsed_ms
            )
            self._diag_bg["characteristic_last_finished_ts"] = datetime.now(
                timezone.utc
            ).timestamp()
            _LOGGER.info(
                "Bayesian State Filter characteristic refit finished: "
                "points=%d, elapsed=%.1f ms, success=%s",
                point_count,
                elapsed_ms,
                not failed,
            )

    def _bootstrap_median(self, now):
        if self._source_cal is None:
            return None
        vals = []
        for src, (t, raw) in self._source_cal.cache.items():
            cal = self._calibrations[src]
            max_age = max(FRESHNESS_MEDIAN_DT_MULTIPLIER * cal.median_dt, FRESHNESS_MIN_S)
            if now - t <= max_age:
                vals.append(raw - cal.bias)
        if len(vals) < self.min_sources:
            return None
        vals.sort()
        n = len(vals)
        return vals[n // 2] if n % 2 else 0.5 * (vals[n // 2 - 1] + vals[n // 2])

    def _fresh_source_count(self, now):
        if self._source_cal is None:
            return 0
        count = 0
        for src, (t, _raw) in self._source_cal.cache.items():
            cal = self._calibrations[src]
            max_age = max(FRESHNESS_MEDIAN_DT_MULTIPLIER * cal.median_dt, FRESHNESS_TAU_FRACTION * self._window_tau(), FRESHNESS_MIN_S)
            if now - t <= max_age:
                count += 1
        return count

    def _update_source_dt(self, cal, src, t):
        # Cadence estimation is independent of warmup history.  This avoids
        # keeping/rebuilding thousands of obsolete warmup points forever.
        prev = self._last_source_event_ts.get(src)
        self._last_source_event_ts[src] = float(t)
        if prev is not None:
            dt = float(t) - float(prev)
            if dt > 0 and math.isfinite(dt):
                # Robust-ish slow EW cadence estimate; large gaps are capped.
                dt = min(dt, 10.0 * max(cal.median_dt, 1.0))
                cal.median_dt = 0.95 * cal.median_dt + 0.05 * dt

    def _append_warmup_sample(self, src, t, raw):
        """Append one warmup point with O(1) bounded maintenance."""
        dq = self._warmup_history.setdefault(
            src, deque(maxlen=self._warmup_max_points_per_source)
        )
        dq.append((float(t), float(raw)))
        cutoff = float(t) - max(self._history_days * 86400.0, 3600.0)
        while dq and dq[0][0] < cutoff:
            dq.popleft()

    def _clear_warmup_history(self):
        for dq in self._warmup_history.values():
            dq.clear()

    def _maybe_schedule_warmup_training(self):
        # Once gated dynamics exist, warmup training has completed its job.
        if self._gated_dynamics is not None:
            return
        if self._warmup_task is not None and not self._warmup_task.done():
            return

        points = sum(len(v) for v in self._warmup_history.values())
        times = [p[0] for v in self._warmup_history.values() for p in v]
        if points < 30 or not times or max(times) - min(times) < 600.0:
            return

        now = max(times)
        # Failed / inconclusive training used to be retried on the very next
        # observation, repeatedly copying and refitting growing arrays. Limit
        # retries to the configured cadence (default: 6 hours).
        if now - self._last_warmup_fit_ts < self._warmup_refit_s:
            return

        self._last_warmup_fit_ts = float(now)
        self._warmup_task = self.hass.async_create_task(self._learn_from_warmup())

    async def _learn_from_warmup(self):
        histories = {k: list(v) for k, v in self._warmup_history.items() if v}
        point_count = sum(len(v) for v in histories.values())
        started = time.perf_counter()
        self._diag_bg["warmup_runs"] += 1
        self._diag_bg["warmup_last_points"] = point_count
        failed = False
        try:
            result = await self.hass.async_add_executor_job(
                lambda: calibrate_history(
                    histories,
                    tau_points=self._tau_points,
                    forget_time_s=self._forget_time_s,
                    tau_min_s=self._tau_min_s,
                    tau_max_s=self._tau_max_s,
                    characteristic_tau_min_s=self._char_tau_min_s,
                    characteristic_tau_max_s=self._char_tau_max_s,
                    bias_anchor=self._bias_anchor,
                    source_models=self._source_models,
                    model_accuracy=self._model_accuracy,
                    huber_delta=self._bias_huber_delta,
                )
            )
            self._dynamics_bank = None
            self._dynamics = None
            if result.characteristic is not None:
                self._characteristic = result.characteristic
            if result.fused_points:
                self._level_grid_step = max(float(result.grid_step), 1.0)
                self._level_history = deque(result.fused_points[-self._level_history_max_points:], maxlen=self._level_history_max_points)
                if len(result.fused_points) >= 40:
                    try:
                        self._gated_dynamics = await self.hass.async_add_executor_job(
                            lambda: train_gated_dynamics((result.dynamics_points or result.fused_points), nu=self._student_nu)
                        )
                        T = self._gated_dynamics.timescale_s
                        if not math.isfinite(T) or T <= 0:
                            T = max(self._gated_dynamics.history_span_s, self._level_grid_step, 1.0)
                        self.filter.tau = T
                        self.filter.q_process = self._gated_dynamics.q_process
                        self.filter.level_q_process = self._gated_dynamics.level_q_process
                        self._apply_derivative_plausibility()
                        self._clear_warmup_history()
                    except Exception:
                        _LOGGER.exception("Bayesian State Filter warmup gated dynamics training failed")
            # Preserve live calibration if it has more evidence; only fill
            # missing sources from warmup training.  If this is the first real
            # pairwise calibration, seed its compact historical evidence so
            # online sigma adaptation starts continuously rather than from an
            # empty pair buffer.
            for src, c in result.sources.items():
                self._calibrations.setdefault(src, c)
            if self._source_cal is None:
                self._source_cal = OnlineSourceCalibrator(
                    self._calibrations,
                    startup_pair_rows=result.startup_pair_rows,
                    calibration_window_s=result.calibration_window_s,
                    bias_anchor=self._bias_anchor,
                    source_models=self._source_models,
                    model_accuracy=self._model_accuracy,
                    huber_delta=self._bias_huber_delta,
                )
            elif result.startup_pair_rows and not self._source_cal.startup_pair_rows:
                self._source_cal.seed_startup_evidence(
                    result.startup_pair_rows, result.calibration_window_s
                )
        except asyncio.CancelledError:
            raise
        except Exception:
            failed = True
            self._diag_bg["warmup_failures"] += 1
            _LOGGER.exception("Bayesian State Filter warmup training failed")
        finally:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            self._diag_bg["warmup_last_ms"] = elapsed_ms
            self._diag_bg["warmup_max_ms"] = max(
                self._diag_bg["warmup_max_ms"], elapsed_ms
            )
            self._diag_bg["warmup_last_finished_ts"] = datetime.now(
                timezone.utc
            ).timestamp()
            _LOGGER.info(
                "Bayesian State Filter warmup training finished: points=%d, "
                "elapsed=%.1f ms, success=%s, gated=%s",
                point_count,
                elapsed_ms,
                not failed,
                self._gated_dynamics is not None,
            )

    def _make_checkpoint_config_fingerprint(self):
        """Hash configuration that changes the physical meaning of saved state.

        A checkpoint is only reusable when source membership/model mapping and
        bias-anchor semantics are unchanged.  Otherwise the stored latent level
        and source biases may live in a different gauge.
        """
        payload = {
            "sources": sorted(
                (src, self._source_models.get(src)) for src in self.sources
            ),
            "min_sources": int(self.min_sources),
            "models": sorted(
                (str(model), float(acc))
                for model, acc in self._model_accuracy.items()
            ),
            "bias_anchor": self._bias_anchor,
            "bias_huber_delta": float(self._bias_huber_delta),
            "noise_model": self._noise_mode_cfg,
            "noise_detection_version": int(self._noise_detection_version),
            "innovation_clip_sigma": float(self._innovation_clip_sigma),
            "regime_stability_version": 1,
            "bias_history_version": 1,
        }
        # Single-source observation-noise calibration has its own schema.
        # Adding this key only for one-source configurations invalidates their
        # old checkpoints once, without forcing established multi-source
        # ensembles through a full Recorder bootstrap.
        if len(self.sources) == 1:
            payload["single_source_sigma_version"] = 1
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _checkpoint_is_compatible(self, saved):
        if not saved or int(saved.get("checkpoint_version", 0)) != self._checkpoint_version:
            return False
        return saved.get("config_fingerprint") == self._checkpoint_config_fingerprint

    # ------------------------------------------------------------------
    # Persistence / diagnostics
    # ------------------------------------------------------------------

    async def _save_state(self):
        started = time.perf_counter()
        self._diag_bg["save_runs"] += 1
        now_ts = float(self.filter.t_last or datetime.now(timezone.utc).timestamp())
        source_cal_state = (
            self._source_cal.dump_compact(now_ts, self._window_tau())
            if self._source_cal is not None else None
        )
        payload = {
            "checkpoint_version": self._checkpoint_version,
            "config_fingerprint": self._checkpoint_config_fingerprint,
            "saved_at": datetime.now(timezone.utc).timestamp(),
            "filter": self.filter.dump_state(),
            "noise_detection_version": self._noise_detection_version,
            "noise_detection": self._noise_detection.dump(),
            "sources": {k: v.dump() for k, v in self._calibrations.items()},
            "source_calibrator": source_cal_state,
            "gated_dynamics": (self._gated_dynamics.dump() if self._gated_dynamics is not None else None),
            "characteristic": self._characteristic.dump() if self._characteristic is not None else None,
            "level_grid_step": self._level_grid_step,
            "level_history": [list(p) for p in self._level_history],
            "edge_history": [list(p) for p in self._edge_history],
            "last_characteristic_fit_ts": self._last_characteristic_fit_ts,
            "last_bias_history_refit_ts": self._last_bias_history_refit_ts,
            "last_warmup_fit_ts": self._last_warmup_fit_ts,
            "last_processed_by_source": self._last_processed_by_source,
        }
        try:
            await self._store.async_save(payload)
        except Exception:
            self._diag_bg["save_failures"] += 1
            raise
        finally:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            self._diag_bg["save_last_ms"] = elapsed_ms
            self._diag_bg["save_max_ms"] = max(
                self._diag_bg["save_max_ms"], elapsed_ms
            )
        self._last_save_ts = datetime.now(timezone.utc).timestamp()

    async def _restore_checkpoint(self) -> bool:
        saved = await self._store.async_load()
        if not self._checkpoint_is_compatible(saved):
            return False
        try:
            self.filter.load_state(saved.get("filter", {}))
            restored_noise = None
            if int(saved.get("noise_detection_version", 0) or 0) == self._noise_detection_version:
                restored_noise = NoiseDetectionResult.load(saved.get("noise_detection"))
            if restored_noise is not None:
                self._noise_detection = restored_noise
            if self.filter.t_last is None:
                return False
            self._calibrations = {
                k: SourceCalibration.load(v) for k, v in (saved.get("sources", {}) or {}).items()
            }
            if not self._calibrations:
                return False
            self._source_cal = OnlineSourceCalibrator.load_compact(
                saved.get("source_calibrator"), self._calibrations,
                bias_anchor=self._bias_anchor, source_models=self._source_models,
                model_accuracy=self._model_accuracy, huber_delta=self._bias_huber_delta,
            )
            self._dynamics_bank = None
            self._dynamics = None
            self._gated_dynamics = GatedDynamicsEstimate.load(saved.get("gated_dynamics"))
            if self._gated_dynamics is not None:
                self._apply_derivative_plausibility()
                self._clear_warmup_history()
            c = saved.get("characteristic") or {}
            self._characteristic = CharacteristicTimeEstimate.load(c) if c else None
            self._level_grid_step = max(float(saved.get("level_grid_step", 60.0)), 1.0)
            level_history = []
            for row in saved.get("level_history", []) or []:
                if len(row) >= 3:
                    level_history.append((float(row[0]), float(row[1]), float(row[2])))
            self._level_history = deque(level_history[-self._level_history_max_points:], maxlen=self._level_history_max_points)

            edge_history = []
            for row in saved.get("edge_history", []) or []:
                try:
                    if len(row) >= 3:
                        t_edge = float(row[0])
                        z_edge = float(row[1])
                        v_edge = float(row[2])
                        if (
                            math.isfinite(t_edge)
                            and math.isfinite(z_edge)
                            and math.isfinite(v_edge)
                            and v_edge > 0.0
                        ):
                            edge_history.append(
                                (t_edge, z_edge, max(v_edge, NUMERIC_VARIANCE_FLOOR))
                            )
                except Exception:
                    continue
            self._edge_history.clear()
            self._edge_history.extend(edge_history[-(self._edge_history.maxlen or 4096):])

            self._last_characteristic_fit_ts = float(saved.get("last_characteristic_fit_ts", 0.0) or 0.0)
            self._last_bias_history_refit_ts = float(saved.get("last_bias_history_refit_ts", 0.0) or 0.0)
            self._last_warmup_fit_ts = float(saved.get("last_warmup_fit_ts", 0.0) or 0.0)
            self._last_processed_by_source = {
                str(k): float(v) for k, v in (saved.get("last_processed_by_source", {}) or {}).items()
            }
            if self._calibrations:
                sigmas = sorted(c.sigma for c in self._calibrations.values() if c.sigma > 0)
                if sigmas:
                    self._gaussian_noise.set_sigma(sigmas[len(sigmas) // 2])
            self._state = round(float(self.filter.x[0]), 6)
            self._build_attrs(last_out=None)
            return True
        except Exception:
            _LOGGER.exception("Bayesian State Filter checkpoint restore failed; falling back to full training")
            return False

    async def _restore_fallback(self):
        """Best-effort restore for legacy snapshots when Recorder is unavailable."""
        saved = await self._store.async_load()
        if not saved:
            return
        if not self._checkpoint_is_compatible(saved):
            _LOGGER.warning(
                "Bayesian State Filter checkpoint ignored: configuration/gauge changed and Recorder history is unavailable"
            )
            return
        self.filter.load_state(saved.get("filter", {}))
        self._calibrations = {
            k: SourceCalibration.load(v) for k, v in (saved.get("sources", {}) or {}).items()
        }
        self._source_cal = OnlineSourceCalibrator.load_compact(
            saved.get("source_calibrator"), self._calibrations,
            bias_anchor=self._bias_anchor, source_models=self._source_models,
            model_accuracy=self._model_accuracy, huber_delta=self._bias_huber_delta,
        )
        self._dynamics = None
        self._dynamics_bank = None
        self._gated_dynamics = GatedDynamicsEstimate.load(saved.get("gated_dynamics"))
        if self._gated_dynamics is not None:
            self._clear_warmup_history()
        c = saved.get("characteristic") or {}
        if c:
            try:
                self._characteristic = CharacteristicTimeEstimate.load(c)
            except Exception:
                self._characteristic = None
        try:
            self._level_grid_step = max(float(saved.get("level_grid_step", 60.0)), 1.0)
        except Exception:
            self._level_grid_step = 60.0
        rows = []
        for row in saved.get("level_history", []) or []:
            try:
                rows.append((float(row[0]), float(row[1]), float(row[2])))
            except Exception:
                continue
        if rows:
            self._level_history = deque(rows[-self._level_history_max_points:], maxlen=self._level_history_max_points)

        edge_rows = []
        for row in saved.get("edge_history", []) or []:
            try:
                if len(row) >= 3:
                    t_edge = float(row[0])
                    z_edge = float(row[1])
                    v_edge = float(row[2])
                    if (
                        math.isfinite(t_edge)
                        and math.isfinite(z_edge)
                        and math.isfinite(v_edge)
                        and v_edge > 0.0
                    ):
                        edge_rows.append(
                            (t_edge, z_edge, max(v_edge, NUMERIC_VARIANCE_FLOOR))
                        )
            except Exception:
                continue
        if edge_rows:
            self._edge_history.clear()
            self._edge_history.extend(edge_rows[-(self._edge_history.maxlen or 4096):])
            self._recompute_edge_witness()
            if self.filter.t_last is not None:
                self._update_edge_agreement_weights()

        self._last_processed_by_source = {
            str(k): float(v) for k, v in (saved.get("last_processed_by_source", {}) or {}).items()
        }
        if self.filter.t_last is not None:
            self._state = round(float(self.filter.x[0]), 6)
            self._build_attrs(last_out=None)

    @staticmethod
    def _round_optional(value, digits=3):
        if value is None:
            return None
        try:
            value = float(value)
            return round(value, digits) if math.isfinite(value) else None
        except (TypeError, ValueError):
            return None

    def _noise_model_params(self):
        """Return stochastic-family selection and independent quantization metadata."""
        if self._noise_mode_cfg == "auto":
            params = {
                "confidence": round(float(self._noise_detection.confidence), 4),
                "reason": self._noise_detection.reason,
            }
        else:
            params = {"selection": "explicit"}

        if self._noise_mode_cfg == "auto" and self._noise_detection.quantization_step is not None:
            q = float(self._noise_detection.quantization_step)
            params["quantization_step"] = round(q, 10)
            params["quantization_sigma"] = round(q / math.sqrt(12.0), 10)
            params["quantization_confidence"] = round(
                float(self._noise_detection.quantization_confidence), 4
            )

        if self._noise_mode_cfg == "auto" and self._noise_detection.level_variance_reason is not None:
            local_active = (
                self._runtime_variance_source == "local_variance_law"
                and self._runtime_level_fraction > 0.0
            )
            params["local_variance_law"] = {
                "model": "constant_plus_level",
                "level_fraction": (
                    None if self._noise_detection.level_variance_fraction is None
                    else round(float(self._noise_detection.level_variance_fraction), 4)
                ),
                "confidence": round(float(self._noise_detection.level_variance_confidence), 4),
                "reference_level": (
                    None if self._noise_detection.level_variance_reference is None
                    else round(float(self._noise_detection.level_variance_reference), 10)
                ),
                "reason": self._noise_detection.level_variance_reason,
                "span_z": (
                    None if self._noise_detection.level_variance_span_z is None
                    else round(float(self._noise_detection.level_variance_span_z), 3)
                ),
                "boundary_limited": bool(
                    self._noise_detection.level_variance_boundary_limited
                ),
                "fraction_identifiable": bool(
                    self._noise_detection.level_variance_fraction is not None
                    and not self._noise_detection.level_variance_boundary_limited
                ),
                "active": bool(local_active),
            }

        runtime_boundary = bool(
            self._runtime_variance_source == "local_variance_law"
            and self._noise_detection.level_variance_boundary_limited
        )
        params["runtime_variance_model"] = {
            "model": "constant_plus_level",
            "level_fraction": round(float(self._runtime_level_fraction), 4),
            "additive_fraction": round(1.0 - float(self._runtime_level_fraction), 4),
            "source": self._runtime_variance_source,
            "boundary_limited": runtime_boundary,
            "fraction_identifiable": not runtime_boundary,
            "active": True,
        }

        if self._noise_model_name == "poisson":
            k = self._noise_detection.poisson_scale
            if k is None and self._last_source is not None:
                cal = self._calibrations.get(self._last_source)
                if cal is not None:
                    k = (cal.sigma * cal.sigma) / max(cal.typical_abs_level, 1e-9)
            if k is not None and math.isfinite(float(k)) and float(k) > 0:
                params["variance_scale"] = round(float(k), 10)
        return params

    def _remember_update_diag(self, out):
        """Cache diagnostics for the latest observation without touching math."""
        if out is None:
            return
        measurement_var = float(
            out.diag.get("measurement_var", out.innovation_var)
        )
        if not math.isfinite(measurement_var) or measurement_var <= 0:
            measurement_var = max(float(out.innovation_var), 1e-12)
        z = abs(float(out.innovation)) / math.sqrt(max(float(out.innovation_var), NUMERIC_VARIANCE_FLOOR))
        self._last_update_diag = {
            "innovation": float(out.innovation),
            "z_score": float(z),
            "robust_weight": float(out.diag.get("weight", 1.0)),
            "innovation_used": float(out.diag.get("innovation_used", out.innovation)),
            "innovation_clipped": bool(out.diag.get("innovation_clipped", False)),
            "clip_sigma": out.diag.get("clip_sigma"),
            "measurement_variance": float(measurement_var),
            "measurement_sigma": math.sqrt(max(float(measurement_var), 0.0)),
            "update_dt_s": float(out.dt),
            "innovation_var": float(out.innovation_var),
            "p_value": float(out.probability_of_event),
            "effective_innovation_variance": float(
                out.diag.get("effective_innovation_var", out.innovation_var)
            ),
            "noise_velocity": float(out.noise_velocity),
        }

    def _build_attrs(self, last_out):
        """Build the public entity surface and optional laboratory diagnostics.

        ``minimal`` keeps only the operational essentials, ``normal`` is the
        default human-readable surface, and ``debug`` adds the full laboratory
        detail under one nested ``debug`` key.  No flat compatibility aliases
        are emitted: each diagnostic has exactly one public location.
        """
        x_diag = np.array(self.filter.x, dtype=float, copy=True)
        P_diag = np.array(self.filter.P, dtype=float, copy=True)

        var = max(float(P_diag[0, 0]), 0.0)
        if self._fallback_active and self._fallback_variance is not None:
            var = max(float(self._fallback_variance), NUMERIC_VARIANCE_FLOOR)

        velocity = float(x_diag[1]) if len(x_diag) > 1 else 0.0
        acceleration = float(x_diag[2]) if len(x_diag) > 2 else 0.0
        jerk = float(x_diag[3]) if len(x_diag) > 3 else 0.0
        dt_weight = self._level_grid_step
        if self._last_update_diag is not None:
            dt_weight = max(float(self._last_update_diag.get("update_dt_s", dt_weight)), 1e-6)

        model = self.filter.state_model
        if hasattr(model, "effective_weights"):
            eff_w = model.effective_weights(x_diag, P_diag, dt_weight)
            conf_w = model.confidence_weights(x_diag, P_diag)
        else:
            eff_w = [1.0] * len(x_diag)
            conf_w = eff_w

        def _sigma(order):
            if order >= len(x_diag):
                return 0.0
            return math.sqrt(max(float(P_diag[order, order]), NUMERIC_VARIANCE_FLOOR))

        def _posterior_z(order):
            sigma = _sigma(order)
            if sigma <= 0.0 or order >= len(x_diag):
                return 0.0
            return abs(float(x_diag[order])) / sigma

        def _round_ratio(mean_value, scale_value):
            try:
                mean_value = float(mean_value)
                scale_value = float(scale_value)
            except (TypeError, ValueError):
                return None
            if not math.isfinite(mean_value) or not math.isfinite(scale_value) or scale_value <= 0.0:
                return None
            return round(mean_value / scale_value, 6)

        # Remember one internally consistent observation/update record before
        # building either the normal or debug surface.
        if last_out is not None:
            self._remember_update_diag(last_out)

        pp = (
            model.derivative_plausibility_parameters()
            if hasattr(model, "derivative_plausibility_parameters") else {}
        )
        scales = list(pp.get("scales", []) or [])
        centers = list(pp.get("centers", []) or [])
        samples = list(pp.get("samples", []) or [])
        gd = self._gated_dynamics
        means = list(getattr(gd, "derivative_means", []) or []) if gd is not None else []
        training_diag = dict(
            getattr(gd, "derivative_training_diagnostics", {}) or {}
        ) if gd is not None else {}

        def _scale(order):
            if len(scales) <= order:
                return None
            value = float(scales[order])
            return value if math.isfinite(value) and value > 0.0 else None

        def _mean(order):
            if len(means) <= order:
                return None
            value = float(means[order])
            return value if math.isfinite(value) else None

        def _center(order):
            if len(centers) <= order:
                return None
            value = float(centers[order])
            return value if math.isfinite(value) else None

        rate_scale = _scale(1)
        curvature_scale = _scale(2)
        jerk_scale = _scale(3)

        dynamics = {
            "rate": {
                "value_per_hour": round(velocity * 3600.0, 10),
                "weight": round(float(eff_w[1]), 6) if len(eff_w) > 1 else 0.0,
            },
            "curvature": {
                "value_per_hour2": round(acceleration * (3600.0 ** 2), 10),
                "weight": round(float(eff_w[2]), 6) if len(eff_w) > 2 else 0.0,
            },
            "jerk": {
                "value_per_hour3": round(jerk * (3600.0 ** 3), 10),
                "weight": round(float(eff_w[3]), 6) if len(eff_w) > 3 else 0.0,
            },
            "plausibility": {
                "rate_sigma_per_hour": (
                    round(rate_scale * 3600.0, 10) if rate_scale is not None else None
                ),
                "curvature_sigma_per_hour2": (
                    round(curvature_scale * (3600.0 ** 2), 10)
                    if curvature_scale is not None else None
                ),
                "jerk_sigma_per_hour3": (
                    round(jerk_scale * (3600.0 ** 3), 10)
                    if jerk_scale is not None else None
                ),
                "rate_mean_over_sigma": _round_ratio(_mean(1), rate_scale),
                "curvature_mean_over_sigma": _round_ratio(_mean(2), curvature_scale),
                "jerk_mean_over_sigma": _round_ratio(_mean(3), jerk_scale),
                "rate_edge_z": (
                    round(float(self._edge_agreement_z[1]), 4)
                    if self._edge_agreement_z.get(1) is not None else None
                ),
                "curvature_edge_z": (
                    round(float(self._edge_agreement_z[2]), 4)
                    if self._edge_agreement_z.get(2) is not None else None
                ),
                "rate_edge_local_weight": (
                    round(float(self._edge_agreement_local_weight[1]), 6)
                    if self._edge_agreement_local_weight.get(1) is not None else None
                ),
                "curvature_edge_local_weight": (
                    round(float(self._edge_agreement_local_weight[2]), 6)
                    if self._edge_agreement_local_weight.get(2) is not None else None
                ),
                "jerk_edge_z": (
                    round(float(self._edge_agreement_z[3]), 4)
                    if self._edge_agreement_z.get(3) is not None else None
                ),
                "jerk_edge_local_weight": (
                    round(float(self._edge_agreement_local_weight[3]), 6)
                    if self._edge_agreement_local_weight.get(3) is not None else None
                ),
            },
        }

        if self._last_derivative_recondition is not None:
            rr = self._last_derivative_recondition
            recovery_diag = {
                "count": int(self._derivative_recondition_count),
                "last_from": rr.get("from_name"),
                "last_reason": rr.get("reason"),
                "last_trigger_z": (
                    round(float(rr.get("trigger_z")), 4)
                    if math.isfinite(float(rr.get("trigger_z", float("nan")))) else None
                ),
                "last_candidate_elapsed_s": (
                    round(float(rr.get("candidate_elapsed_s")), 3)
                    if rr.get("candidate_elapsed_s") is not None else None
                ),
                "last_confirm_duration_s": (
                    round(float(rr.get("confirm_duration_s")), 3)
                    if rr.get("confirm_duration_s") is not None else None
                ),
                "enter_sigma": float(self._derivative_recovery_enter_sigma),
                "cancel_sigma": float(self._derivative_recovery_cancel_sigma),
                "confirm_updates": int(self._derivative_recovery_confirm_updates),
                "timescale_factor": float(self._derivative_recovery_timescale_factor),
                "process_timescale_s": (
                    round(float(self._gated_dynamics.timescale_s), 3)
                    if self._gated_dynamics is not None
                    and math.isfinite(float(self._gated_dynamics.timescale_s))
                    else None
                ),
                "bad_update_counts": {
                    {1: "rate", 2: "curvature", 3: "jerk"}.get(int(k), str(k)): int(v)
                    for k, v in sorted(self._fallback_derivative_weight_counts.items())
                    if int(k) < self.filter.state_model.dim_x()
                },
                "bad_duration_s": {
                    {1: "rate", 2: "curvature", 3: "jerk"}.get(int(k), str(k)): (
                        round(max(
                            0.0,
                            float(self.filter.t_last or 0.0) - float(v)
                        ), 3)
                        if v is not None else 0.0
                    )
                    for k, v in sorted(self._derivative_recovery_bad_since.items())
                    if int(k) < self.filter.state_model.dim_x()
                },
            }
            applied_hints = dict(rr.get("applied_hints", {}) or {})
            recovery_diag["hinted_orders"] = [
                {1: "rate", 2: "curvature", 3: "jerk"}.get(int(k), str(k))
                for k in sorted(applied_hints, key=lambda x: int(x))
            ]
            recovery_diag["edge_divergence"] = {
                "enter_sigma": float(self._edge_divergence_sigma),
                "cancel_sigma": float(self._edge_divergence_cancel_sigma),
                "confirm_updates": int(self._edge_divergence_confirm_updates),
                "counts": {
                    {1: "rate", 2: "curvature", 3: "jerk"}.get(int(k), str(k)): int(v)
                    for k, v in sorted(self._edge_divergence_counts.items())
                },
                "z": {
                    {1: "rate", 2: "curvature", 3: "jerk"}.get(int(k), str(k)): round(float(v), 4)
                    for k, v in sorted(self._edge_last_divergence_z.items())
                },
                "last_edge_update_age_s": (
                    round(
                        max(
                            0.0,
                            float(self.filter.t_last or 0.0) - float(self._edge_last_update_ts)
                        ),
                        3,
                    )
                    if self._edge_last_update_ts is not None
                    else None
                ),
            }
            if self._diagnostics_mode == "debug":
                recovery_diag.update({
                    "last_timestamp": rr.get("timestamp"),
                    "old_derivatives": rr.get("old_derivatives"),
                    "restored_sigmas": rr.get("restored_sigmas"),
                    "applied_hints": applied_hints,
                })
            dynamics["recovery"] = recovery_diag

        else:
            # Expose watchdog state even before the first recovery.  This is
            # diagnostic-only and lets us distinguish "no valid edge hints"
            # from "waiting for 3 witness confirmations".
            dynamics["recovery"] = {
                "count": int(self._derivative_recondition_count),
                "enter_sigma": float(self._derivative_recovery_enter_sigma),
                "cancel_sigma": float(self._derivative_recovery_cancel_sigma),
                "confirm_updates": int(self._derivative_recovery_confirm_updates),
                "timescale_factor": float(self._derivative_recovery_timescale_factor),
                "process_timescale_s": (
                    round(float(self._gated_dynamics.timescale_s), 3)
                    if self._gated_dynamics is not None
                    and math.isfinite(float(self._gated_dynamics.timescale_s))
                    else None
                ),
                "edge_divergence": {
                    "enter_sigma": float(self._edge_divergence_sigma),
                "cancel_sigma": float(self._edge_divergence_cancel_sigma),
                    "confirm_updates": int(self._edge_divergence_confirm_updates),
                    "counts": {
                        {1: "rate", 2: "curvature", 3: "jerk"}.get(int(k), str(k)): int(v)
                        for k, v in sorted(self._edge_divergence_counts.items())
                    },
                    "z": {
                        {1: "rate", 2: "curvature", 3: "jerk"}.get(int(k), str(k)): round(float(v), 4)
                        for k, v in sorted(self._edge_last_divergence_z.items())
                    },
                    "last_edge_update_age_s": (
                        round(
                            max(
                                0.0,
                                float(self.filter.t_last or 0.0) - float(self._edge_last_update_ts)
                            ),
                            3,
                        )
                        if self._edge_last_update_ts is not None
                        else None
                    ),
                },
            }

        edge = self._edge_derivative_estimate
        rate_consensus = self._edge_derivative_consensus.get("rate")
        curvature_consensus = self._edge_derivative_consensus.get("curvature")
        jerk_consensus = self._edge_derivative_consensus.get("jerk")
        if edge is not None and rate_consensus is not None:
            rate_consensus = self._edge_rate_consensus
            curvature_consensus = self._edge_curvature_consensus
            jerk_consensus = self._edge_jerk_consensus
            scale_spans = {}
            for name, est in (self._edge_witness_scales or {}).items():
                scale_spans[name] = {
                    "points": int(est.points),
                    "span_s": round(float(est.span_s), 3),
                }

            edge_diag = {
                "method": "robust_multiscale_causal_local_polynomial",
                "bandwidth_source": "point_count_and_process_timescale",
                "windows_points": [25, 40, 55],
                "windows_tau_factors": [0.5, 1.0, 2.0],
                "process_timescale_s": (
                    round(float(self._edge_witness_tau_s), 3)
                    if self._edge_witness_tau_s is not None else None
                ),
                "gap_limit_s": round(float(self._edge_process_timescale_s()), 3),
                "segment_points": int(len(self._edge_history)),
                "segment_span_s": (
                    round(
                        float(self._edge_history[-1][0])
                        - float(self._edge_history[0][0]),
                        3,
                    )
                    if len(self._edge_history) >= 2 else 0.0
                ),
                "windows": scale_spans,
            }

            if rate_consensus is not None:
                edge_diag.update({
                    "rate_per_hour": round(float(rate_consensus["value"]) * 3600.0, 10),
                    "rate_sigma_per_hour": round(float(rate_consensus["sigma"]) * 3600.0, 10),
                    "rate_fit_sigma_per_hour": round(float(rate_consensus["fit_sigma"]) * 3600.0, 10),
                    "rate_scale_sigma_per_hour": round(float(rate_consensus["scale_sigma"]) * 3600.0, 10),
                    "scale_disagreement_z": round(float(rate_consensus["disagreement_z"]), 4),
                    "scale_consistent": bool(rate_consensus["consistent"]),
                    "available_windows": int(rate_consensus["available"]),
                    "windows_used": list(rate_consensus["used"]),
                })

            if curvature_consensus is not None:
                edge_diag.update({
                    "curvature_per_hour2": round(
                        float(curvature_consensus["value"]) * (3600.0 ** 2), 10
                    ),
                    "curvature_sigma_per_hour2": round(
                        float(curvature_consensus["sigma"]) * (3600.0 ** 2), 10
                    ),
                    "curvature_fit_sigma_per_hour2": round(
                        float(curvature_consensus["fit_sigma"]) * (3600.0 ** 2), 10
                    ),
                    "curvature_scale_sigma_per_hour2": round(
                        float(curvature_consensus["scale_sigma"]) * (3600.0 ** 2), 10
                    ),
                    "curvature_scale_disagreement_z": round(
                        float(curvature_consensus["disagreement_z"]), 4
                    ),
                })

            if jerk_consensus is not None:
                edge_diag.update({
                    "jerk_per_hour3": round(
                        float(jerk_consensus["value"]) * (3600.0 ** 3), 10
                    ),
                    "jerk_sigma_per_hour3": round(
                        float(jerk_consensus["sigma"]) * (3600.0 ** 3), 10
                    ),
                    "jerk_fit_sigma_per_hour3": round(
                        float(jerk_consensus["fit_sigma"]) * (3600.0 ** 3), 10
                    ),
                    "jerk_scale_sigma_per_hour3": round(
                        float(jerk_consensus["scale_sigma"]) * (3600.0 ** 3), 10
                    ),
                    "jerk_scale_disagreement_z": round(
                        float(jerk_consensus["disagreement_z"]), 4
                    ),
                    "jerk_available_windows": int(jerk_consensus["available"]),
                    "jerk_windows_used": list(jerk_consensus["used"]),
                })

            dynamics["edge_witness"] = edge_diag

        # Compact operational source summary.  Detailed per-source calibration
        # lives only in debug mode.
        health = self._source_health()
        unhealthy = []
        for src, item in health.items():
            last_z = item.get("last_z_score")
            weight = item.get("robust_weight")
            if (
                (last_z is not None and float(last_z) >= self._regime_change_z_threshold)
                or (weight is not None and float(weight) <= STUDENT_T_MIN_WEIGHT + 1e-12)
            ):
                unhealthy.append(src)
        active_sources = len(getattr(self._source_cal, "cache", {}) or {}) if self._source_cal else 0
        sources_summary = {
            "configured": len(self.sources),
            "active": active_sources,
            "unhealthy": len(unhealthy),
        }
        if unhealthy:
            sources_summary["unhealthy_entities"] = unhealthy

        filter_summary = {
            "mode": self._mode_name(),
            "noise_model": self._noise_model_name,
        }
        if self._noise_mode_cfg == "auto":
            filter_summary["noise_confidence"] = round(float(self._noise_detection.confidence), 4)

        fallback = {
            "active": bool(self._fallback_active),
        }
        if self._fallback_active:
            fallback.update({
                "reason": self._fallback_reason,
                "level_z": (
                    round(float(self._fallback_last_z), 4)
                    if math.isfinite(float(self._fallback_last_z)) else None
                ),
                "observed_level": (
                    round(float(self._fallback_value), 10)
                    if self._fallback_value is not None else None
                ),
            })

        regime = {
            "candidate": bool(self._regime_candidate_count > 0),
        }
        if self._regime_candidate_count > 0:
            regime.update({
                "confirmations": int(self._regime_candidate_count),
                "direction": (
                    "up" if self._regime_candidate_sign > 0
                    else "down" if self._regime_candidate_sign < 0
                    else None
                ),
                "peak_z": round(float(self._regime_candidate_peak_z), 4),
            })
        if self._last_regime_change is not None:
            regime["last_jump"] = round(float(self._last_regime_change["jump"]), 10)
            regime["last_direction"] = self._last_regime_change["direction"]

        attrs = {
            ATTR_STDDEV: round(math.sqrt(var), 10),
            "filter": filter_summary,
            "dynamics": dynamics,
            "fallback": fallback,
        }

        if self._diagnostics_mode != "minimal":
            attrs["regime"] = regime
            attrs["sources"] = sources_summary
            if self._last_update_diag is not None:
                d = self._last_update_diag
                attrs["last_update"] = {
                    "source": self._last_source,
                    "innovation_z": round(d["z_score"], 4),
                    "clipped": bool(d["innovation_clipped"]),
                    "dt_s": round(d["update_dt_s"], 3),
                }
                if d["innovation_clipped"]:
                    attrs["last_update"].update({
                        "innovation": round(d["innovation"], 10),
                        "innovation_used": round(d["innovation_used"], 10),
                        "clip_sigma": d["clip_sigma"],
                    })

        if self._diagnostics_mode == "debug":
            anchor_diag = self._source_cal.last_anchor if self._source_cal is not None else None
            calibration = {
                "bias_anchor_mode": self._bias_anchor,
                "bias_anchor_last_shift": (
                    round(float(anchor_diag.shift), 10) if anchor_diag else 0.0
                ),
                "noise_variance_source": (
                    "per_source_calibration" if self._calibrations else "model_default"
                ),
            }
            if anchor_diag and anchor_diag.model_centers:
                calibration["bias_anchor_models"] = {
                    model_name: {
                        "center": round(float(center), 8),
                        "absolute_accuracy": round(
                            float(self._model_accuracy.get(model_name, 0.0)), 8
                        ),
                        "effective_weight": round(
                            float((anchor_diag.model_weights or {}).get(model_name, 0.0)), 8
                        ),
                    }
                    for model_name, center in anchor_diag.model_centers.items()
                }

            timescales = {
                "gated": {
                    "timescale_s": round(float(self.filter.tau), 3),
                    "local_rmse": None,
                    "local_rmse_step1": None,
                    "local_rmse_step2": None,
                },
                "characteristic": {
                    "time_s": None,
                    "p10_s": None,
                    "p90_s": None,
                    "confidence": 0.0,
                    "status": "unavailable",
                    "identifiable": False,
                    "boundary_limited": False,
                },
            }
            if self._gated_dynamics is not None:
                timescales["gated"].update({
                    "local_rmse": round(float(self._gated_dynamics.validation_rmse), 10),
                    "local_rmse_step1": round(float(self._gated_dynamics.validation_rmse_step1), 10),
                    "local_rmse_step2": round(float(self._gated_dynamics.validation_rmse_step2), 10),
                })
            if self._characteristic is not None:
                c = self._characteristic
                timescales["characteristic"].update({
                    "time_s": self._round_optional(c.tau),
                    "p10_s": self._round_optional(c.p10),
                    "p90_s": self._round_optional(c.p90),
                    "confidence": round(c.confidence, 4),
                    "status": c.status,
                    "identifiable": bool(c.identifiable),
                    "boundary_limited": bool(c.boundary_limited),
                })

            derivative_debug = {
                "law": pp.get("law"),
                "hierarchy": "cumulative_lower_order_penalties",
                "cutoff_sigma": pp.get("cutoff_sigma"),
                "regime_segmentation": "same_6sigma_3_confirmation_compact_plateau_semantics",
                "posterior": {
                    "rate_sigma_per_hour": round(_sigma(1) * 3600.0, 10),
                    "curvature_sigma_per_hour2": round(_sigma(2) * (3600.0 ** 2), 10),
                    "jerk_sigma_per_hour3": round(_sigma(3) * (3600.0 ** 3), 10),
                    "rate_z": round(_posterior_z(1), 4),
                    "curvature_z": round(_posterior_z(2), 4),
                    "jerk_z": round(_posterior_z(3), 4),
                    "confidence_weights": [round(float(v), 6) for v in conf_w[1:4]],
                },
                "training": {
                    "rate_median_per_hour": (
                        round(_center(1) * 3600.0, 10) if _center(1) is not None else None
                    ),
                    "curvature_median_per_hour2": (
                        round(_center(2) * (3600.0 ** 2), 10) if _center(2) is not None else None
                    ),
                    "jerk_median_per_hour3": (
                        round(_center(3) * (3600.0 ** 3), 10) if _center(3) is not None else None
                    ),
                    "rate_mean_per_hour": (
                        round(_mean(1) * 3600.0, 10) if _mean(1) is not None else None
                    ),
                    "curvature_mean_per_hour2": (
                        round(_mean(2) * (3600.0 ** 2), 10) if _mean(2) is not None else None
                    ),
                    "jerk_mean_per_hour3": (
                        round(_mean(3) * (3600.0 ** 3), 10) if _mean(3) is not None else None
                    ),
                    "samples": list(samples[1:4]) if len(samples) >= 4 else list(samples),
                    **training_diag,
                },
            }

            full_last_update = None
            if self._last_update_diag is not None:
                d = self._last_update_diag
                full_last_update = {
                    "source": self._last_source,
                    "measurement_sigma": round(d["measurement_sigma"], 10),
                    "measurement_variance": round(d["measurement_variance"], 10),
                    "innovation": round(d["innovation"], 10),
                    "innovation_z": round(d["z_score"], 4),
                    "robust_weight": round(d["robust_weight"], 6),
                    "innovation_used": round(d["innovation_used"], 10),
                    "innovation_clipped": bool(d["innovation_clipped"]),
                    "clip_sigma": d["clip_sigma"],
                    "update_dt_s": round(d["update_dt_s"], 3),
                    "innovation_variance": round(d["innovation_var"], 10),
                    "effective_innovation_variance": round(
                        d["effective_innovation_variance"], 10
                    ),
                    "p_value": round(d["p_value"], 8),
                    "noise_velocity": round(d["noise_velocity"], 10),
                }

            debug = {
                "variance": round(var, 10),
                "noise": {
                    "mode_config": self._noise_mode_cfg,
                    "model": self._noise_model_name,
                    "params": self._noise_model_params(),
                },
                "calibration": calibration,
                "dynamics": derivative_debug,
                "timescales": timescales,
                "startup": {
                    "mode": self._startup_mode,
                    "checkpoint_store_key": self._store_key,
                    "hidden_recorder_replayed": int(self._startup_sync_replayed),
                    "hidden_current_replayed": int(self._startup_sync_current),
                    "hidden_sync_gap_s": round(float(self._startup_sync_gap_s), 3),
                },
                "fallback": {
                    "active": bool(self._fallback_active),
                    "reason": self._fallback_reason,
                    "level_z": (
                        round(float(self._fallback_last_z), 4)
                        if math.isfinite(float(self._fallback_last_z)) else None
                    ),
                    "enter_z": self._fallback_enter_z,
                    "exit_z": self._fallback_exit_z,
                    "exit_confirmations": self._fallback_exit_confirmations,
                    "safe_count": int(self._fallback_safe_count),
                    "entries": int(self._fallback_entries),
                    "observed_level": (
                        round(float(self._fallback_value), 10)
                        if self._fallback_value is not None else None
                    ),
                    "last_event": self._fallback_last_event,
                },
                "source_health": health,
                "regime_change": {
                    "z_threshold": self._regime_change_z_threshold,
                    "confirmations_required": self._regime_change_confirmations,
                    "startup_guard_remaining": int(self._regime_startup_guard_remaining),
                    "startup_reanchors": int(self._regime_startup_reanchors),
                    "candidate_count": int(self._regime_candidate_count),
                    "candidate_direction": (
                        "up" if self._regime_candidate_sign > 0
                        else "down" if self._regime_candidate_sign < 0
                        else None
                    ),
                    "candidate_peak_z": round(float(self._regime_candidate_peak_z), 4),
                    "last_event": self._last_regime_change,
                },
                "last_update": full_last_update,
            }

            if self._characteristic is not None:
                c = self._characteristic
                debug["characteristic_fit"] = {
                    "edge_mass": round(c.edge_mass, 4),
                    "nugget_variance": round(c.nugget_variance, 10),
                    "process_variance": round(c.process_variance, 10),
                    "signal_fraction": round(c.signal_fraction, 4),
                    "fit_error": round(c.fit_error, 4),
                    "lag_count": int(c.lag_count),
                    "pair_count": int(c.pair_count),
                    "min_lag_s": round(c.min_lag, 3),
                    "max_lag_s": round(c.max_lag, 3),
                }
            attrs["debug"] = debug

        self._attrs = attrs

    def _mode_name(self):
        # Publication mode is intentionally distinct from latent filter mode.
        # In raw_fallback the Bayesian state keeps updating internally while HA
        # receives the direct untransported observation until agreement returns.
        if self.filter.t_last is None:
            return "warmup"
        if self._fallback_active:
            return "raw_fallback"
        try:
            if hasattr(self.filter.state_model, "effective_weights"):
                weights = self.filter.state_model.effective_weights(
                    self.filter.x, self.filter.P, max(float(self._level_grid_step), 1e-6)
                )
                if len(weights) > 1 and float(np.min(weights[1:])) <= self._fallback_derivative_weight_threshold:
                    return "degraded"
        except Exception:
            pass
        return "tracking"

    def _source_health(self):
        """Return compact per-source runtime diagnostics.

        This attribute is published on every state update, so keep it small.
        Long-lived calibration accounting is available through the adaptive
        calibration CPU diagnostics and does not belong in the hot state path.
        """
        out = {}
        for src, c in self._calibrations.items():
            model = self._source_models.get(src)
            item = {
                "model": model,
                "bias": round(c.bias, 8),
                "sigma": round(c.sigma, 8),
                "median_dt_s": round(c.median_dt, 3),
                "outlier_rate": round(c.outlier_rate, 5),
            }

            d = self._source_last_diag.get(src)
            if d is not None:
                item.update({
                    "last_z_score": round(d["z_score"], 4),
                    "robust_weight": round(d["robust_weight"], 6),
                })

            out[src] = item
        return out

    def _seed_current_sources(self):
        if self._source_cal is None:
            self._source_cal = OnlineSourceCalibrator(
                self._calibrations, bias_anchor=self._bias_anchor,
                source_models=self._source_models, model_accuracy=self._model_accuracy,
                huber_delta=self._bias_huber_delta,
            )
        for src in self.sources:
            st = self.hass.states.get(src)
            if not st or st.state in ("unknown", "unavailable", None):
                continue
            try:
                z = float(st.state)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(z):
                continue
            self._inherit_metadata(st)
            t = self._state_timestamp(st)
            self._source_cal.ensure_source(src, z, t)

        if self.filter.t_last is None and self._source_cal.cache:
            now = max(t for t, _ in self._source_cal.cache.values())
            bootstrap = self._bootstrap_median(now)
            if bootstrap is not None:
                vars_ = [c.sigma * c.sigma for c in self._calibrations.values()]
                var = sorted(vars_)[len(vars_) // 2] if vars_ else 1.0
                out = self.filter.step(Observation(
                    t=now, z=bootstrap, variance=max(var, 1e-12), source="bootstrap"
                ))
                self._state = round(out.y_mean, 6)
                self._last_source = None
                self._remember_update_diag(out)
                self._build_attrs(last_out=out)

    @staticmethod
    def _state_timestamp(st):
        t = getattr(st, "last_reported", None) or st.last_updated
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
        return t.timestamp()

    def _inherit_metadata(self, st):
        if self._attr_native_unit_of_measurement is None:
            self._attr_native_unit_of_measurement = (
                st.attributes.get("native_unit_of_measurement")
                or st.attributes.get("unit_of_measurement")
            )
            self._attr_device_class = st.attributes.get("device_class")
            self._attr_state_class = st.attributes.get("state_class")
            self._attr_icon = st.attributes.get("icon")

    def _apply_noise_model(self):
        """Apply stochastic-family diagnostics and runtime variance shape.

        Runtime source variance uses one continuous family:

            R(x) = R_ref * ((1-p) + p * |x| / x_ref)

        where R_ref is the existing per-source calibrated sigma^2.  Gaussian
        and scaled-Poisson are therefore just p=0 and p=1.  A statistically
        supported local variance law may choose an intermediate p without
        changing the learned source-noise scale.
        """
        mode = self._noise_mode_cfg
        if mode == "auto":
            family = self._noise_detection.family
        else:
            family = mode

        runtime_p = 0.0
        runtime_source = "constant"

        if mode == "poisson":
            runtime_p = 1.0
            runtime_source = "explicit_poisson"
        elif mode == "gaussian":
            runtime_p = 0.0
            runtime_source = "explicit_gaussian"
        elif family == "poisson":
            # Stationary scaled-Poisson evidence is informative even when the
            # signal has too little independent level span for the local law.
            runtime_p = 1.0
            runtime_source = self._noise_detection.reason or "poisson_detection"
        else:
            lv_reason = self._noise_detection.level_variance_reason
            lv_conf = float(self._noise_detection.level_variance_confidence or 0.0)
            lv_fraction = self._noise_detection.level_variance_fraction
            if (
                lv_reason == "level_dependent_local_residuals"
                and lv_fraction is not None
                and lv_conf >= 0.55
            ):
                runtime_p = max(0.0, min(1.0, float(lv_fraction)))
                runtime_source = "local_variance_law"

        self._runtime_level_fraction = float(runtime_p)
        self._runtime_variance_source = runtime_source
        self._runtime_noise_mode = (
            "poisson" if runtime_p >= 1.0 - 1e-12 else
            "gaussian" if runtime_p <= 1e-12 else
            "mixed"
        )

        # CoreFilter normally receives explicit fused observation variance, so
        # this object is principally the fallback/p-value model.  Both current
        # implementations use the same Gaussian-tail p-value calculation.
        if family == "poisson":
            k = self._noise_detection.poisson_scale
            if k is not None and math.isfinite(float(k)) and float(k) > 0:
                self._poisson_noise.set_scale(float(k))
            self.filter.noise_model = self._poisson_noise
        else:
            self.filter.noise_model = self._gaussian_noise

        # Public noise_model describes the observation-noise law actually used
        # by runtime, not merely the legacy binary detector family.  A local
        # level-dependent fit is reported as gaussian+poisson even when its
        # constrained best fit lands on p=1: boundary-limited p=1 is not proof
        # of a pure Poisson mechanism.
        if mode == "poisson" or (mode == "auto" and family == "poisson"):
            self._noise_model_name = "poisson"
        elif mode == "gaussian":
            self._noise_model_name = "gaussian"
        elif runtime_source == "local_variance_law":
            self._noise_model_name = "gaussian+poisson"
        else:
            self._noise_model_name = "gaussian"

        return self._runtime_noise_mode

    @property
    def native_value(self):
        return self._state

    @property
    def extra_state_attributes(self):
        return self._attrs
