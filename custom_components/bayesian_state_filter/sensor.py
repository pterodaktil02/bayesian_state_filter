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
)
from .core.filter import CoreFilter
from .core.noise_models import GaussianNoise, PoissonLikeNoise
from .core.noise_detection import NoiseDetectionResult, detect_noise_model
from .core.process_noise import IntegratedWienerProcessNoise
from .core.state_models import AdaptivePolynomialStateModel
from .core.gated_training import GatedDynamicsEstimate, train_gated_dynamics
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
    """Bayesian State Filter 0.4.1.

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
        # Public diagnostics are intentionally compact by default.  This is a
        # presentation-only switch: it does not change filtering, training,
        # persistence, or any estimator hyperparameters.
        diagnostics = str(bayes_cfg.get("diagnostics", "compact")).strip().lower()
        self._diagnostics_full = diagnostics in {"full", "debug", "verbose"}

        self._attr_native_unit_of_measurement = None
        self._attr_device_class = None
        self._attr_state_class = None
        self._attr_icon = None

        self._gaussian_noise = GaussianNoise(sigma=0.1)
        self._poisson_noise = PoissonLikeNoise(k=0.1)
        self._noise_model_name = "gaussian"
        self._noise_detection_version = 3
        self._noise_detection = NoiseDetectionResult(reason="not_yet_detected")

        default_tau = self._tau_min_s or 3600.0
        self.filter = CoreFilter(
            state_model=AdaptivePolynomialStateModel(3),
            noise_model=self._gaussian_noise,
            updater=StudentTUpdater(nu=self._student_nu, min_weight=0.05),
            process_noise=IntegratedWienerProcessNoise(order=3, q=0.0, level_q=0.0),
            prior_timescale_s=default_tau,
        )

        self._calibrations: dict[str, SourceCalibration] = {}
        self._source_cal: OnlineSourceCalibrator | None = None
        # Full [x,v,a,j] dynamics identified from history.  The state order is
        # fixed; posterior confidence gates only derivative coupling.
        self._gated_dynamics: GatedDynamicsEstimate | None = None
        self._dynamics_bank = None  # legacy field retained only for migration-safe code paths
        self._dynamics = None       # legacy field retained only for old diagnostics
        # Independent level-process characteristic-time estimate.
        self._characteristic: CharacteristicTimeEstimate | None = None
        self._level_history = deque(maxlen=self._level_history_max_points)
        self._level_grid_step = 60.0
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
        # Per-source live diagnostics.  These are presentation-only and are
        # deliberately kept outside SourceCalibration/persistence so adding
        # observability cannot change the estimator or stored calibration.
        self._source_last_diag: dict[str, dict] = {}
        self._ready = False
        self._last_save_ts = 0.0
        self._checkpoint_version = 8
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
                _LOGGER.exception("Bayesian State Filter 0.4.1 initialization failed")
                await self._restore_fallback()
            self._seed_current_sources()
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
            await self._catch_up_from_recorder()
            if self.filter.t_last is not None:
                self._state = round(float(self.filter.x[0]), 6)
                self._build_attrs(last_out=None)
            _LOGGER.info(
                "Bayesian State Filter 0.4.1 restored checkpoint and caught up incrementally; t=%.3f",
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

        # Identify q/timescale for the permanent confidence-gated [x,v,a,j]
        # model.  Confidence itself is not fitted: c(z)=erf(|z|/sqrt(2)).
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

        self._apply_noise_model()

        if result.fused_points:
            await self.hass.async_add_executor_job(self._replay_fused, dynamics_points)
            # The full history bootstrap has consumed all source history through
            # each source's last Recorder sample.  Remember those watermarks so
            # the next restart can request only the unseen tail.
            self._last_processed_by_source = {
                src: float(seq[-1][0]) for src, seq in histories.items() if seq
            }
            self._state = round(float(self.filter.x[0]), 6)
            self._build_attrs(last_out=None)
            await self._save_state()
            _LOGGER.info(
                "Bayesian State Filter 0.4.1 trained from %.2f d: gated_tau=%s q=%s level_q=%s "
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
        self._state = round(out.y_mean, 6)
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


    async def _catch_up_from_recorder(self):
        """Replay only source observations newer than the persisted watermarks."""
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

    def _current_fused_snapshot(self, now, *, with_components=False):
        if self._source_cal is None:
            return None
        values, variances, labels = [], [], []
        tau = self._window_tau()
        mode = self._noise_model_name
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
            base_var = cal.variance(corrected, noise_mode=mode)

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
                self._clear_warmup_history()
            c = saved.get("characteristic") or {}
            self._characteristic = CharacteristicTimeEstimate.load(c) if c else None
            self._level_grid_step = max(float(saved.get("level_grid_step", 60.0)), 1.0)
            level_history = []
            for row in saved.get("level_history", []) or []:
                if len(row) >= 3:
                    level_history.append((float(row[0]), float(row[1]), float(row[2])))
            self._level_history = deque(level_history[-self._level_history_max_points:], maxlen=self._level_history_max_points)
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
        """Build user-facing diagnostics without changing estimator behavior.

        ``compact`` is the public/default surface.  ``full`` restores the
        laboratory diagnostics from 2.1.0 for troubleshooting and research.
        """
        # Take one atomic numerical snapshot for every state-derived public
        # diagnostic.  The estimator can be updated independently of the
        # slower attribute publication cadence, so mixing separate reads of x
        # and P can otherwise expose an internally inconsistent diagnostic
        # tuple (most visibly jerk / jerk_stddev / jerk_z).
        x_diag = np.array(self.filter.x, dtype=float, copy=True)
        P_diag = np.array(self.filter.P, dtype=float, copy=True)

        var = max(float(P_diag[0, 0]), 0.0)
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
            eff_w = [1.0] * len(self.filter.x)
            conf_w = eff_w

        def _sigma(order):
            if order >= len(x_diag):
                return 0.0
            return math.sqrt(max(float(P_diag[order, order]), NUMERIC_VARIANCE_FLOOR))

        def _z(order):
            sigma = _sigma(order)
            if sigma <= 0.0 or order >= len(x_diag):
                return 0.0
            return abs(float(x_diag[order])) / sigma

        # Compact public surface.  Group related diagnostics so the entity
        # remains readable in Home Assistant.  The old flat attributes are
        # retained for one compatibility cycle below.
        anchor_diag = self._source_cal.last_anchor if self._source_cal is not None else None

        dynamics = {
            "rate": {
                "value_per_hour": round(velocity * 3600.0, 10),
                "stddev_per_hour": round(_sigma(1) * 3600.0, 10),
                "z": round(_z(1), 4),
                "weight": round(float(eff_w[1]), 6) if len(eff_w) > 1 else 0.0,
            },
            "curvature": {
                "value_per_hour2": round(acceleration * (3600.0 ** 2), 10),
                "stddev_per_hour2": round(_sigma(2) * (3600.0 ** 2), 10),
                "z": round(_z(2), 4),
                "weight": round(float(eff_w[2]), 6) if len(eff_w) > 2 else 0.0,
            },
            "jerk": {
                "value_per_hour3": round(jerk * (3600.0 ** 3), 10),
                "stddev_per_hour3": round(_sigma(3) * (3600.0 ** 3), 10),
                "z": round(_z(3), 4),
                "weight": round(float(eff_w[3]), 6) if len(eff_w) > 3 else 0.0,
            },
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

        calibration = {
            "bias_anchor_mode": self._bias_anchor,
            "bias_anchor_last_shift": (round(float(anchor_diag.shift), 10) if anchor_diag else 0.0),
            "noise_variance_source": ("per_source_calibration" if self._calibrations else "model_default"),
        }
        if anchor_diag and anchor_diag.model_centers:
            calibration["bias_anchor_models"] = {
                model_name: {
                    "center": round(float(center), 8),
                    "absolute_accuracy": round(float(self._model_accuracy.get(model_name, 0.0)), 8),
                    "effective_weight": round(float((anchor_diag.model_weights or {}).get(model_name, 0.0)), 8),
                }
                for model_name, center in anchor_diag.model_centers.items()
            }

        attrs = {
            ATTR_STDDEV: round(math.sqrt(var), 10),
            "model": {
                "filter_mode": self._mode_name(),
                "noise_model_mode": self._noise_mode_cfg,
                "noise_model": self._noise_model_name,
                "noise_model_params": self._noise_model_params(),
            },
            "calibration": calibration,
            "dynamics": dynamics,
            "timescales": timescales,
            "startup": {
                "mode": self._startup_mode,
                "checkpoint_store_key": self._store_key,
            },
            ATTR_SOURCE_HEALTH: self._source_health(),
        }

        # Show one internally consistent observation/update record. These
        # values all refer to the same source and the same Bayesian update.
        if last_out is not None:
            self._remember_update_diag(last_out)
        if self._last_update_diag is not None:
            d = self._last_update_diag
            attrs["last_update"] = {
                "source": self._last_source,
                "measurement_sigma": round(d["measurement_sigma"], 10),
                "measurement_variance": round(d["measurement_variance"], 10),
                "innovation": round(d["innovation"], 10),
                "z_score": round(d["z_score"], 4),
                "robust_weight": round(d["robust_weight"], 6),
                "update_dt_s": round(d["update_dt_s"], 3),
            }

        # Compatibility layer for 0.4.1-era templates/automations.  Keep these
        # flat aliases for one release cycle; new consumers should use the
        # grouped dictionaries above.
        attrs.update({
            "bias_anchor_mode": calibration["bias_anchor_mode"],
            "bias_anchor_last_shift": calibration["bias_anchor_last_shift"],
            "startup_mode": self._startup_mode,
            "checkpoint_store_key": self._store_key,
            ATTR_FILTER_MODE: self._mode_name(),
            ATTR_NOISE_MODEL_MODE: self._noise_mode_cfg,
            ATTR_NOISE_MODEL: self._noise_model_name,
            ATTR_NOISE_MODEL_PARAMS: self._noise_model_params(),
            ATTR_NOISE_VARIANCE_SOURCE: calibration["noise_variance_source"],
            ATTR_RATE_PER_HOUR: dynamics["rate"]["value_per_hour"],
            ATTR_RATE_STDDEV_PER_HOUR: dynamics["rate"]["stddev_per_hour"],
            ATTR_CURVATURE_PER_HOUR2: dynamics["curvature"]["value_per_hour2"],
            ATTR_CURVATURE_STDDEV_PER_HOUR2: dynamics["curvature"]["stddev_per_hour2"],
            ATTR_JERK_PER_HOUR3: dynamics["jerk"]["value_per_hour3"],
            ATTR_JERK_STDDEV_PER_HOUR3: dynamics["jerk"]["stddev_per_hour3"],
            ATTR_RATE_WEIGHT: dynamics["rate"]["weight"],
            ATTR_CURVATURE_WEIGHT: dynamics["curvature"]["weight"],
            ATTR_JERK_WEIGHT: dynamics["jerk"]["weight"],
            ATTR_RATE_Z: dynamics["rate"]["z"],
            ATTR_CURVATURE_Z: dynamics["curvature"]["z"],
            ATTR_JERK_Z: dynamics["jerk"]["z"],
            ATTR_GATED_TIMESCALE: timescales["gated"]["timescale_s"],
            ATTR_GATED_LOCAL_RMSE: timescales["gated"]["local_rmse"],
            ATTR_GATED_LOCAL_RMSE_STEP1: timescales["gated"]["local_rmse_step1"],
            ATTR_GATED_LOCAL_RMSE_STEP2: timescales["gated"]["local_rmse_step2"],
            ATTR_CHARACTERISTIC_TIME: timescales["characteristic"]["time_s"],
            ATTR_CHARACTERISTIC_TIME_P10: timescales["characteristic"]["p10_s"],
            ATTR_CHARACTERISTIC_TIME_P90: timescales["characteristic"]["p90_s"],
            ATTR_CHARACTERISTIC_TIME_CONFIDENCE: timescales["characteristic"]["confidence"],
            ATTR_CHARACTERISTIC_TIME_STATUS: timescales["characteristic"]["status"],
            ATTR_CHARACTERISTIC_TIME_IDENTIFIABLE: timescales["characteristic"]["identifiable"],
            ATTR_DYNAMICS_BOUNDARY_LIMITED: timescales["characteristic"]["boundary_limited"],
        })
        if self._last_update_diag is not None:
            d = self._last_update_diag
            if self._last_source is not None:
                attrs[ATTR_LAST_SOURCE] = self._last_source
            attrs.update({
                ATTR_MEASUREMENT_SIGMA: round(d["measurement_sigma"], 10),
                ATTR_MEASUREMENT_VARIANCE: round(d["measurement_variance"], 10),
                ATTR_INNOVATION: round(d["innovation"], 10),
                ATTR_Z_SCORE: round(d["z_score"], 4),
                ATTR_ROBUST_WEIGHT: round(d["robust_weight"], 6),
                ATTR_UPDATE_DT: round(d["update_dt_s"], 3),
            })
        if "bias_anchor_models" in calibration:
            attrs["bias_anchor_models"] = calibration["bias_anchor_models"]

        if self._diagnostics_full:
            # Exact 2.1-era laboratory diagnostics.  Kept behind an explicit
            # switch so existing investigations remain possible without
            # overwhelming normal entity attributes.
            attrs.update({
                ATTR_VARIANCE: round(var, 10),
                ATTR_VELOCITY: round(velocity, 10),
                ATTR_VELOCITY_TIME: round(self.filter.tau, 3),
                ATTR_PROCESS_NOISE: f"{self.filter.q_process:.6e}",
            })

            if self._characteristic is not None:
                c = self._characteristic
                attrs.update({
                    ATTR_DYNAMICS_EDGE_MASS: round(c.edge_mass, 4),
                    ATTR_CHARACTERISTIC_NUGGET_VARIANCE: round(c.nugget_variance, 10),
                    ATTR_CHARACTERISTIC_PROCESS_VARIANCE: round(c.process_variance, 10),
                    ATTR_CHARACTERISTIC_SIGNAL_FRACTION: round(c.signal_fraction, 4),
                    ATTR_CHARACTERISTIC_FIT_ERROR: round(c.fit_error, 4),
                    ATTR_CHARACTERISTIC_LAG_COUNT: int(c.lag_count),
                    ATTR_CHARACTERISTIC_PAIR_COUNT: int(c.pair_count),
                    ATTR_CHARACTERISTIC_MIN_LAG: round(c.min_lag, 3),
                    ATTR_CHARACTERISTIC_MAX_LAG: round(c.max_lag, 3),
                })

            if self._last_update_diag is not None:
                d = self._last_update_diag
                attrs.update({
                    ATTR_NOISE_VELOCITY: round(d["noise_velocity"], 10),
                    ATTR_INNOVATION_VAR: round(d["innovation_var"], 10),
                    ATTR_P_VALUE: round(d["p_value"], 8),
                    ATTR_EFFECTIVE_INNOVATION_VARIANCE: round(
                        d["effective_innovation_variance"], 10
                    ),
                })

        self._attrs = attrs

    def _mode_name(self):
        # The level/state estimate can be healthy even when the experimental
        # predictive velocity-memory hyperparameter is not identifiable.  Do
        # not label the whole filter as "learning" merely because velocity_tau
        # is boundary-limited.
        if self.filter.t_last is None:
            return "warmup"
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
        """Apply configured or history-detected observation-noise family."""
        mode = self._noise_mode_cfg
        if mode == "auto":
            family = self._noise_detection.family
        else:
            family = mode

        if family == "poisson":
            k = self._noise_detection.poisson_scale
            if k is not None and math.isfinite(float(k)) and float(k) > 0:
                self._poisson_noise.set_scale(float(k))
            self.filter.noise_model = self._poisson_noise
            self._noise_model_name = "poisson"
            return "poisson"

        # Quantization is an independent observation property.  It is already
        # present in the empirically learned source sigma, so q^2/12 is not
        # added again here.
        self.filter.noise_model = self._gaussian_noise
        self._noise_model_name = "gaussian"
        return "gaussian"

    @property
    def native_value(self):
        return self._state

    @property
    def extra_state_attributes(self):
        return self._attrs
