from __future__ import annotations

import asyncio
import logging
import math
import statistics
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
    ATTR_RATE_PER_HOUR, ATTR_CURVATURE_PER_HOUR2, ATTR_JERK_PER_HOUR3,
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
from .core.process_noise import IntegratedWienerProcessNoise
from .core.state_models import AdaptivePolynomialStateModel
from .core.gated_training import GatedDynamicsEstimate, train_gated_dynamics
from .core.training import (
    OnlineSourceCalibrator,
    SourceCalibration,
    calibrate_history,
)
from .core.types import Observation
from .core.updaters import StudentTUpdater
from .core.variogram import CharacteristicTimeEstimate, estimate_characteristic_time

_LOGGER = logging.getLogger(__name__)


async def async_setup_platform(hass, config, async_add_entities, discovery_info=None):
    async_add_entities([BayesianEnsembleSensor(hass, config)])


class BayesianEnsembleSensor(SensorEntity):
    """Bayesian State Filter 0.3.0.

    Backward-compatible YAML platform.  The implementation is intentionally
    source-aware: raw observations are never collapsed into one irreversible
    weighted average before the Bayesian update.
    """

    _attr_has_entity_name = True

    def __init__(self, hass, config):
        self.hass = hass
        name = config.get("name", "Bayesian Sensor")
        self._attr_name = name
        slug = name.lower().replace(" ", "_")
        self._attr_unique_id = f"{DOMAIN}_{slug}"

        ens_cfg = config.get("ensemble", {}) or {}
        self.sources = list(ens_cfg.get("sources", []))
        self.min_sources = max(int(ens_cfg.get("min_sources", 1)), 1)

        bayes_cfg = config.get("bayes", {}) or {}
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
        self._characteristic_refit_s = max(float(bayes_cfg.get("characteristic_refit_s", 1800.0)), CHARACTERISTIC_REFIT_MIN_S)
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

        default_tau = self._tau_min_s or 3600.0
        self.filter = CoreFilter(
            state_model=AdaptivePolynomialStateModel(3),
            noise_model=self._gaussian_noise,
            updater=StudentTUpdater(nu=self._student_nu, min_weight=0.05),
            process_noise=IntegratedWienerProcessNoise(order=3, q=0.0),
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
        self._level_history = deque(maxlen=20000)
        self._level_grid_step = 60.0
        self._characteristic_task = None
        self._last_characteristic_fit_ts = 0.0

        self._warmup_history = {src: [] for src in self.sources}
        self._warmup_task = None

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
        self._checkpoint_version = 4
        self._last_processed_by_source: dict[str, float] = {}
        # New key: old v1 snapshot has incompatible semantics (notably q).
        self._store = Store(hass, 1, f"{DOMAIN}_{slug}_v31")

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
                _LOGGER.exception("Bayesian State Filter 0.3.0 initialization failed")
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
            await self._catch_up_from_recorder()
            self._select_noise_model()
            if self.filter.t_last is not None:
                self._state = round(float(self.filter.x[0]), 6)
                self._build_attrs(last_out=None)
            _LOGGER.info(
                "Bayesian State Filter 0.3.0 restored checkpoint and caught up incrementally; t=%.3f",
                float(self.filter.t_last or 0.0),
            )
            return

        histories = {}
        for entity_id in self.sources:
            seq = await self._fetch_history(entity_id)
            parsed = self._parse_states(seq)
            if parsed:
                histories[entity_id] = parsed

        if not histories:
            await self._restore_fallback()
            return

        result = await self.hass.async_add_executor_job(
            lambda: calibrate_history(
                histories,
                tau_points=self._tau_points,
                forget_time_s=self._forget_time_s,
                tau_min_s=self._tau_min_s,
                tau_max_s=self._tau_max_s,
                characteristic_tau_min_s=self._char_tau_min_s,
                characteristic_tau_max_s=self._char_tau_max_s,
            )
        )
        self._calibrations = result.sources
        self._source_cal = OnlineSourceCalibrator(
            self._calibrations,
            startup_pair_rows=result.startup_pair_rows,
            calibration_window_s=result.calibration_window_s,
        )
        if self._calibrations:
            sigmas = sorted(c.sigma for c in self._calibrations.values() if c.sigma > 0)
            if sigmas:
                self._gaussian_noise.set_sigma(sigmas[len(sigmas) // 2])
        self._dynamics_bank = None
        self._dynamics = None
        self._characteristic = result.characteristic
        self._level_grid_step = max(float(result.grid_step), 1.0)
        self._level_history = deque(result.fused_points[-20000:], maxlen=20000)
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

        self._select_noise_model()

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
                "Bayesian State Filter 0.3.1 trained from %.2f d: gated_tau=%s q=%s "
                "local_rmse=%s characteristic_time=%s (status=%s, conf=%.3f)",
                result.history_span / 86400.0,
                (f"{self.filter.tau:.1f} s" if self._gated_dynamics is not None else "fallback"),
                f"{self.filter.q_process:.6e}",
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
            self._source_cal = OnlineSourceCalibrator(self._calibrations)
        self._source_cal.ensure_source(src, raw, t)
        cal = self._calibrations[src]
        self._update_source_dt(cal, src, t)
        self._source_cal.update_snapshot(t, self._window_tau(), updated_source=src)
        self._last_processed_by_source[src] = max(
            float(t), float(self._last_processed_by_source.get(src, float("-inf")))
        )

        if self._fresh_source_count(t) < self.min_sources:
            self._warmup_history.setdefault(src, []).append((t, raw))
            self._trim_warmup_history(t)
            return

        source_corrected = raw - cal.bias
        corrected = source_corrected
        mode = self._select_noise_model(st)
        variance = cal.variance(corrected, noise_mode=mode)

        if self.filter.t_last is None:
            bootstrap = self._bootstrap_median(t)
            corrected = bootstrap if bootstrap is not None else corrected

        out = self.filter.step(Observation(
            t=t, z=corrected, source=src, variance=variance,
            meta={"raw": raw, "bias": cal.bias},
        ))

        z = abs(out.innovation) / math.sqrt(max(out.innovation_var, NUMERIC_VARIANCE_FLOOR))
        weight = float(out.diag.get("weight", 1.0))
        cal.updates += 1
        if weight < 0.25 or z > 4.0:
            cal.outliers += 1

        self._source_last_diag[src] = {
            "raw": float(raw),
            "corrected": float(source_corrected),
            "innovation": float(out.innovation),
            "z_score": float(z),
            "robust_weight": float(weight),
        }

        self._update_dynamics(t, corrected, variance, out.dt)
        self._append_level_snapshot(t)
        if schedule_background:
            self._maybe_schedule_characteristic_fit(t)
        self._state = round(out.y_mean, 6)
        self._last_source = src
        self._remember_update_diag(out)
        self._build_attrs(last_out=out)
        if write_state:
            self.async_write_ha_state()

        self._warmup_history.setdefault(src, []).append((t, raw))
        self._trim_warmup_history(t)
        if schedule_background and self._gated_dynamics is None:
            self._maybe_schedule_warmup_training()

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

    @staticmethod
    def _robust_median_variance(values, variances):
        if not values:
            return None
        vals = list(map(float, values))
        vars_ = [max(float(v), 1e-12) for v in variances]
        zmed = float(statistics.median(vals))
        if len(vals) >= 3:
            absdev = [abs(v - zmed) for v in vals]
            scale = 1.4826 * float(statistics.median(absdev))
            floor = math.sqrt(float(statistics.median(vars_)))
            scale = max(scale, floor, 1e-12)
            kept = [(z, v) for z, v in zip(vals, vars_) if abs(z - zmed) <= 4.685 * scale]
            if kept:
                vals = [z for z, _ in kept]
                vars_ = [v for _, v in kept]
                zmed = float(statistics.median(vals))
        if len(vars_) == 1:
            vf = vars_[0]
        else:
            mvar = float(statistics.median(vars_))
            vf = max(1.57 * mvar / len(vars_), 0.25 * mvar, 1e-12)
        return zmed, vf

    def _current_fused_snapshot(self, now):
        if self._source_cal is None:
            return None
        values, variances = [], []
        tau = self._window_tau()
        for src, (t, raw) in self._source_cal.cache.items():
            cal = self._calibrations.get(src)
            if cal is None:
                continue
            max_age = max(FRESHNESS_MEDIAN_DT_MULTIPLIER * cal.median_dt, FRESHNESS_TAU_FRACTION * tau, FRESHNESS_MIN_S)
            if now - t > max_age:
                continue
            corrected = raw - cal.bias
            values.append(corrected)
            variances.append(cal.variance(corrected, noise_mode=self._noise_model_name))
        if len(values) < self.min_sources:
            return None
        return self._robust_median_variance(values, variances)

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
            _LOGGER.exception("Bayesian State Filter characteristic-time refit failed")

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
        # ensure_source has already stored current time, so keep an independent
        # last-event timestamp in warmup data for cadence adaptation.
        hist = self._warmup_history.get(src, [])
        if hist:
            dt = t - hist[-1][0]
            if dt > 0 and math.isfinite(dt):
                # Robust-ish slow EW cadence estimate; large gaps are capped.
                dt = min(dt, 10.0 * max(cal.median_dt, 1.0))
                cal.median_dt = 0.95 * cal.median_dt + 0.05 * dt

    def _trim_warmup_history(self, now):
        cutoff = now - max(self._history_days * 86400.0, 3600.0)
        for src, seq in self._warmup_history.items():
            if len(seq) > 5000:
                self._warmup_history[src] = [p for p in seq if p[0] >= cutoff][-5000:]

    def _maybe_schedule_warmup_training(self):
        if self._warmup_task is not None and not self._warmup_task.done():
            return
        points = sum(len(v) for v in self._warmup_history.values())
        times = [p[0] for v in self._warmup_history.values() for p in v]
        if points < 30 or not times or max(times) - min(times) < 600.0:
            return
        self._warmup_task = self.hass.async_create_task(self._learn_from_warmup())

    async def _learn_from_warmup(self):
        histories = {k: list(v) for k, v in self._warmup_history.items() if v}
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
                )
            )
            self._dynamics_bank = None
            self._dynamics = None
            if result.characteristic is not None:
                self._characteristic = result.characteristic
            if result.fused_points:
                self._level_grid_step = max(float(result.grid_step), 1.0)
                self._level_history = deque(result.fused_points[-20000:], maxlen=20000)
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
                )
            elif result.startup_pair_rows and not self._source_cal.startup_pair_rows:
                self._source_cal.seed_startup_evidence(
                    result.startup_pair_rows, result.calibration_window_s
                )
        except asyncio.CancelledError:
            raise
        except Exception:
            _LOGGER.exception("Bayesian State Filter warmup training failed")

    # ------------------------------------------------------------------
    # Persistence / diagnostics
    # ------------------------------------------------------------------

    async def _save_state(self):
        now_ts = float(self.filter.t_last or datetime.now(timezone.utc).timestamp())
        source_cal_state = (
            self._source_cal.dump_compact(now_ts, self._window_tau())
            if self._source_cal is not None else None
        )
        await self._store.async_save({
            "checkpoint_version": self._checkpoint_version,
            "saved_at": datetime.now(timezone.utc).timestamp(),
            "filter": self.filter.dump_state(),
            "sources": {k: v.dump() for k, v in self._calibrations.items()},
            "source_calibrator": source_cal_state,
            "gated_dynamics": (self._gated_dynamics.dump() if self._gated_dynamics is not None else None),
            "characteristic": self._characteristic.dump() if self._characteristic is not None else None,
            "level_grid_step": self._level_grid_step,
            "level_history": [list(p) for p in self._level_history],
            "last_characteristic_fit_ts": self._last_characteristic_fit_ts,
            "last_processed_by_source": self._last_processed_by_source,
        })
        self._last_save_ts = datetime.now(timezone.utc).timestamp()

    async def _restore_checkpoint(self) -> bool:
        saved = await self._store.async_load()
        if not saved or int(saved.get("checkpoint_version", 0)) != self._checkpoint_version:
            return False
        try:
            self.filter.load_state(saved.get("filter", {}))
            if self.filter.t_last is None:
                return False
            self._calibrations = {
                k: SourceCalibration.load(v) for k, v in (saved.get("sources", {}) or {}).items()
            }
            if not self._calibrations:
                return False
            self._source_cal = OnlineSourceCalibrator.load_compact(
                saved.get("source_calibrator"), self._calibrations
            )
            self._dynamics_bank = None
            self._dynamics = None
            self._gated_dynamics = GatedDynamicsEstimate.load(saved.get("gated_dynamics"))
            c = saved.get("characteristic") or {}
            self._characteristic = CharacteristicTimeEstimate.load(c) if c else None
            self._level_grid_step = max(float(saved.get("level_grid_step", 60.0)), 1.0)
            level_history = []
            for row in saved.get("level_history", []) or []:
                if len(row) >= 3:
                    level_history.append((float(row[0]), float(row[1]), float(row[2])))
            self._level_history = deque(level_history[-20000:], maxlen=20000)
            self._last_characteristic_fit_ts = float(saved.get("last_characteristic_fit_ts", 0.0) or 0.0)
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
        self.filter.load_state(saved.get("filter", {}))
        self._calibrations = {
            k: SourceCalibration.load(v) for k, v in (saved.get("sources", {}) or {}).items()
        }
        self._source_cal = OnlineSourceCalibrator.load_compact(
            saved.get("source_calibrator"), self._calibrations
        )
        self._dynamics = None
        self._dynamics_bank = None
        self._gated_dynamics = GatedDynamicsEstimate.load(saved.get("gated_dynamics"))
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
            self._level_history = deque(rows[-20000:], maxlen=20000)
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
        """Return only parameters specific to the active noise family.

        The generic per-observation uncertainty lives in measurement_sigma /
        measurement_variance.  Keeping it out of this mapping makes the public
        interface future-proof for auto-detected Poisson noise: ``noise_model``
        names the family, while this mapping contains only family-specific
        learned parameters.
        """
        if self._noise_model_name == "poisson" and self._last_source is not None:
            cal = self._calibrations.get(self._last_source)
            if cal is not None:
                k = (cal.sigma * cal.sigma) / max(cal.typical_abs_level, 1e-9)
                if math.isfinite(k) and k > 0:
                    return {
                        "variance_scale": round(float(k), 10),
                        "estimated": True,
                    }
        return {}

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
        var = max(float(self.filter.P[0, 0]), 0.0)
        velocity = float(self.filter.x[1]) if len(self.filter.x) > 1 else 0.0
        acceleration = float(self.filter.x[2]) if len(self.filter.x) > 2 else 0.0
        jerk = float(self.filter.x[3]) if len(self.filter.x) > 3 else 0.0
        dt_weight = self._level_grid_step
        if self._last_update_diag is not None:
            dt_weight = max(float(self._last_update_diag.get("update_dt_s", dt_weight)), 1e-6)
        model = self.filter.state_model
        if hasattr(model, "effective_weights"):
            eff_w = model.effective_weights(self.filter.x, self.filter.P, dt_weight)
            conf_w = model.confidence_weights(self.filter.x, self.filter.P)
        else:
            eff_w = [1.0] * len(self.filter.x)
            conf_w = eff_w

        def _z(order):
            if order >= len(self.filter.x):
                return 0.0
            sigma = math.sqrt(max(float(self.filter.P[order, order]), NUMERIC_VARIANCE_FLOOR))
            return abs(float(self.filter.x[order])) / sigma

        # Compact public surface: state uncertainty, robust update health,
        # source calibration, and the independently estimated level-process
        # characteristic time.
        attrs = {
            ATTR_STDDEV: round(math.sqrt(var), 10),
            ATTR_FILTER_MODE: self._mode_name(),
            # ``noise_model_mode`` is the user's selection policy (auto /
            # gaussian / poisson); ``noise_model`` is the family actually in
            # use right now.  Auto is deliberately Gaussian-only for now, but
            # this split lets a future detector select Poisson without changing
            # YAML or the public attribute schema.
            ATTR_NOISE_MODEL_MODE: self._noise_mode_cfg,
            ATTR_NOISE_MODEL: self._noise_model_name,
            ATTR_NOISE_MODEL_PARAMS: self._noise_model_params(),
            ATTR_NOISE_VARIANCE_SOURCE: ("per_source_calibration" if self._calibrations else "model_default"),
            ATTR_SOURCE_HEALTH: self._source_health(),
            # Full state is always [x,v,a,j]. Internal derivative units are per
            # second; publish human-scale per-hour diagnostics.
            ATTR_RATE_PER_HOUR: round(velocity * 3600.0, 10),
            ATTR_CURVATURE_PER_HOUR2: round(acceleration * (3600.0 ** 2), 10),
            ATTR_JERK_PER_HOUR3: round(jerk * (3600.0 ** 3), 10),
            ATTR_RATE_WEIGHT: round(float(eff_w[1]), 6) if len(eff_w) > 1 else 0.0,
            ATTR_CURVATURE_WEIGHT: round(float(eff_w[2]), 6) if len(eff_w) > 2 else 0.0,
            ATTR_JERK_WEIGHT: round(float(eff_w[3]), 6) if len(eff_w) > 3 else 0.0,
            ATTR_RATE_Z: round(_z(1), 4),
            ATTR_CURVATURE_Z: round(_z(2), 4),
            ATTR_JERK_Z: round(_z(3), 4),
            ATTR_GATED_TIMESCALE: round(float(self.filter.tau), 3),
        }
        if self._gated_dynamics is not None:
            attrs.update({
                ATTR_GATED_LOCAL_RMSE: round(float(self._gated_dynamics.validation_rmse), 10),
                ATTR_GATED_LOCAL_RMSE_STEP1: round(float(self._gated_dynamics.validation_rmse_step1), 10),
                ATTR_GATED_LOCAL_RMSE_STEP2: round(float(self._gated_dynamics.validation_rmse_step2), 10),
            })

        # Level-process characteristic time: variogram estimate, entirely
        # separate from the damped-velocity tau used internally by CoreFilter.
        if self._characteristic is not None:
            c = self._characteristic
            attrs.update({
                ATTR_CHARACTERISTIC_TIME: self._round_optional(c.tau),
                ATTR_CHARACTERISTIC_TIME_P10: self._round_optional(c.p10),
                ATTR_CHARACTERISTIC_TIME_P90: self._round_optional(c.p90),
                ATTR_CHARACTERISTIC_TIME_CONFIDENCE: round(c.confidence, 4),
                ATTR_CHARACTERISTIC_TIME_STATUS: c.status,
                ATTR_CHARACTERISTIC_TIME_IDENTIFIABLE: bool(c.identifiable),
                ATTR_DYNAMICS_BOUNDARY_LIMITED: bool(c.boundary_limited),
            })
        else:
            attrs.update({
                ATTR_CHARACTERISTIC_TIME: None,
                ATTR_CHARACTERISTIC_TIME_CONFIDENCE: 0.0,
                ATTR_CHARACTERISTIC_TIME_STATUS: "unavailable",
                ATTR_CHARACTERISTIC_TIME_IDENTIFIABLE: False,
                ATTR_DYNAMICS_BOUNDARY_LIMITED: False,
            })

        # Show one internally consistent observation/update record.  These
        # values all refer to the same source and the same Bayesian update.
        # They remain visible if _build_attrs() is later called by a background
        # characteristic-time refit.
        if last_out is not None:
            self._remember_update_diag(last_out)
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
        out = {}
        for src, c in self._calibrations.items():
            item = {
                "bias": round(c.bias, 8),
                "sigma": round(c.sigma, 8),
                "median_dt_s": round(c.median_dt, 3),
                # Keep both numerator and denominator next to the ratio.
                # A rate of 1.0 after 2/2 updates means something very different
                # from 100/100 and should be obvious from one HA attribute dump.
                "outliers": int(c.outliers),
                "outlier_rate": round(c.outlier_rate, 5),
                "history_samples": int(c.samples),
                # Current working pairwise evidence.  In 0.2.1.7 this is a
                # continuous startup+live rolling-window estimate; the startup
                # evidence ages out over calibration_window_s instead of being
                # replaced by the first short live burst after reload.
                "calibration_samples": int(c.calibration_samples),
                "calibration_span_s": round(c.calibration_span, 1),
                "calibration_pairs": int(c.calibration_pairs),
                # Explicit history/startup evidence, kept immutable so it does
                # not disappear from diagnostics after live calibration starts.
                "startup_calibration_samples": int(c.startup_calibration_samples),
                "startup_calibration_span_s": round(c.startup_calibration_span, 1),
                "startup_calibration_pairs": int(c.startup_calibration_pairs),
                "startup_sigma": (
                    round(c.startup_sigma, 8) if c.startup_sigma > 0 else None
                ),
                "calibration_window_s": (
                    round(c.calibration_window_s, 1)
                    if c.calibration_window_s > 0 else None
                ),
                "live_updates": int(c.updates),
            }
            if self._source_cal is not None:
                item.update({
                    "live_calibration_samples": int(
                        self._source_cal.live_calibration_samples.get(src, 0)
                    ),
                    "live_calibration_span_s": round(
                        self._source_cal.live_calibration_span.get(src, 0.0), 1
                    ),
                    "live_calibration_pairs": int(
                        self._source_cal.live_calibration_pairs.get(src, 0)
                    ),
                })
            d = self._source_last_diag.get(src)
            if d is not None:
                item.update({
                    "last_raw_value": round(d["raw"], 8),
                    "last_corrected_value": round(d["corrected"], 8),
                    "last_innovation": round(d["innovation"], 8),
                    "last_z_score": round(d["z_score"], 4),
                    "last_robust_weight": round(d["robust_weight"], 6),
                })
            out[src] = item
        return out

    def _seed_current_sources(self):
        if self._source_cal is None:
            self._source_cal = OnlineSourceCalibrator(self._calibrations)
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

    def _select_noise_model(self, st=None):
        mode = self._noise_mode_cfg
        # ``auto`` is intentionally conservative in 0.3.0 and therefore
        # selects Gaussian.  The public mode/family split is already in place
        # so a future history-based detector can select Poisson without a YAML
        # or attribute-schema migration.  Explicit ``poisson`` remains
        # available for count/rate-like sources.
        if mode == "poisson":
            self.filter.noise_model = self._poisson_noise
            self._noise_model_name = "poisson"
            return "poisson"
        self.filter.noise_model = self._gaussian_noise
        self._noise_model_name = "gaussian"
        return "gaussian"

    @property
    def native_value(self):
        return self._state

    @property
    def extra_state_attributes(self):
        return self._attrs
