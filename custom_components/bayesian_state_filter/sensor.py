from __future__ import annotations

import asyncio
import logging
import math
import statistics
from collections import deque
from functools import partial
from datetime import datetime, timedelta, timezone

from homeassistant.components.sensor import SensorEntity
from homeassistant.const import EVENT_HOMEASSISTANT_STARTED
from homeassistant.helpers.event import async_track_state_change_event
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
    NUMERIC_VARIANCE_FLOOR, STUDENT_T_MIN_WEIGHT,
    FRESHNESS_MEDIAN_DT_MULTIPLIER, FRESHNESS_TAU_FRACTION,
    FRESHNESS_MIN_S, CHARACTERISTIC_REFIT_MIN_S,
)
from .core.filter import CoreFilter
from .core.noise_models import GaussianNoise, PoissonLikeNoise
from .core.process_noise import DampedAccelerationProcessNoise
from .core.state_models import LevelVelocityModel
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
    """Bayesian State Filter 0.2.1.x.

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
        self._save_every_s = max(float(bayes_cfg.get("save_every_s", 60.0)), 5.0)
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
            state_model=LevelVelocityModel(tau=default_tau),
            noise_model=self._gaussian_noise,
            updater=StudentTUpdater(nu=self._student_nu, min_weight=STUDENT_T_MIN_WEIGHT),
            process_noise=DampedAccelerationProcessNoise(q_acc=1e-9, tau=default_tau),
        )

        self._calibrations: dict[str, SourceCalibration] = {}
        self._source_cal: OnlineSourceCalibrator | None = None
        # Predictive velocity-model hyperparameters.
        self._dynamics_bank = None
        self._dynamics = None
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
        # New key: old v1 snapshot has incompatible semantics (notably q).
        self._store = Store(hass, 1, f"{DOMAIN}_{slug}_v21")

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
        async_track_state_change_event(self.hass, self.sources, self._handle_event)

        async def _after_start(event):
            try:
                await self._initialize()
            except Exception:
                _LOGGER.exception("Bayesian State Filter 0.2.1.6 initialization failed")
                await self._restore_fallback()
            self._seed_current_sources()
            self._ready = True
            if self._state is not None:
                self.async_write_ha_state()

        self.hass.bus.async_listen_once(EVENT_HOMEASSISTANT_STARTED, _after_start)

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
        self._source_cal = OnlineSourceCalibrator(self._calibrations)
        if self._calibrations:
            sigmas = sorted(c.sigma for c in self._calibrations.values() if c.sigma > 0)
            if sigmas:
                self._gaussian_noise.set_sigma(sigmas[len(sigmas) // 2])
        self._dynamics_bank = result.bank
        self._dynamics = result.dynamics
        self._characteristic = result.characteristic
        self._level_grid_step = max(float(result.grid_step), 1.0)
        self._level_history = deque(result.fused_points[-20000:], maxlen=20000)
        if result.fused_points:
            self._last_characteristic_fit_ts = float(result.fused_points[-1][0])

        if self._dynamics is not None:
            # This is velocity_tau: a predictive model hyperparameter.  It is
            # intentionally independent of characteristic_time.
            self.filter.tau = self._dynamics.tau
            self.filter.q_acc = self._dynamics.q_acc

        self._select_noise_model()

        if result.fused_points:
            await self.hass.async_add_executor_job(self._replay_fused, result.fused_points)
            self._state = round(float(self.filter.x[0]), 6)
            self._build_attrs(last_out=None)
            await self._save_state()
            _LOGGER.info(
                "Bayesian State Filter 0.2.1.6 trained from %.2f d: velocity_tau=%.1f s "
                "(conf=%.3f), characteristic_time=%s (status=%s, conf=%.3f)",
                result.history_span / 86400.0,
                self.filter.tau,
                self._dynamics.confidence if self._dynamics else 0.0,
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

    async def _fetch_history(self, entity_id):
        """Fetch raw recorder history for one source.

        Home Assistant removed the old ``history.get_states`` helper.  Use the
        supported ``get_significant_states`` API with keyword arguments: its
        fifth positional argument is ``filters``, not ``include_start_time_state``.
        Passing booleans positionally here therefore silently broke v2.0.0 on
        current HA and made startup fall back to warmup.
        """
        now = datetime.now(timezone.utc)
        start = now - timedelta(days=self._history_days)
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
        t = self._state_timestamp(st)
        src = st.entity_id

        if self._source_cal is None:
            self._source_cal = OnlineSourceCalibrator(self._calibrations)
        self._source_cal.ensure_source(src, raw, t)
        cal = self._calibrations[src]
        self._update_source_dt(cal, src, t)
        self._source_cal.update_snapshot(t, self._window_tau(), updated_source=src)

        if self._fresh_source_count(t) < self.min_sources:
            self._warmup_history.setdefault(src, []).append((t, raw))
            self._trim_warmup_history(t)
            return

        source_corrected = raw - cal.bias
        corrected = source_corrected
        mode = self._select_noise_model(st)
        variance = cal.variance(corrected, noise_mode=mode)

        # Cold start: median is supplied by the source calibrator cache if there
        # are peers; otherwise the first source value is the only honest prior.
        # Keep source_corrected unchanged: diagnostics should always describe
        # what this source actually contributed before any bootstrap fallback.
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

        # Snapshot the exact live observation diagnostics for this source.
        # These values are never fed back into filtering/calibration.
        self._source_last_diag[src] = {
            "raw": float(raw),
            "corrected": float(source_corrected),
            "innovation": float(out.innovation),
            "z_score": float(z),
            "robust_weight": float(weight),
        }

        self._update_dynamics(t, corrected, variance, out.dt)
        self._append_level_snapshot(t)
        self._maybe_schedule_characteristic_fit(t)
        self._state = round(out.y_mean, 6)
        self._last_source = src
        self._remember_update_diag(out)
        self._build_attrs(last_out=out)
        self.async_write_ha_state()

        self._warmup_history.setdefault(src, []).append((t, raw))
        self._trim_warmup_history(t)
        if self._dynamics_bank is None:
            self._maybe_schedule_warmup_training()

        if t - self._last_save_ts >= self._save_every_s:
            self._last_save_ts = t
            await self._save_state()

    def _update_dynamics(self, t, corrected, variance, dt):
        if self._dynamics_bank is None:
            return
        self._dynamics_bank.update(t, corrected, variance)
        est = self._dynamics_bank.estimate()
        if est is None:
            return
        self._dynamics = est
        if not est.identifiable:
            return

        # The posterior may move quickly; the filter's working parameters do
        # not.  Log-space blending prevents a tau-window feedback oscillation.
        alpha = min(0.03, max(float(dt), 0.0) / max(10.0 * self.filter.tau, 1.0))
        alpha *= max(est.confidence, 0.1)
        if alpha <= 0:
            return
        self.filter.tau = math.exp(
            (1.0 - alpha) * math.log(self.filter.tau)
            + alpha * math.log(max(est.tau, 1e-6))
        )
        self.filter.q_acc = math.exp(
            (1.0 - alpha) * math.log(max(self.filter.q_acc, 1e-18))
            + alpha * math.log(max(est.q_acc, 1e-18))
        )

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
            if result.bank is not None:
                self._dynamics_bank = result.bank
                self._dynamics = result.dynamics
            if result.characteristic is not None:
                self._characteristic = result.characteristic
            if result.fused_points:
                self._level_grid_step = max(float(result.grid_step), 1.0)
                self._level_history = deque(result.fused_points[-20000:], maxlen=20000)
            # Preserve live calibration if it has more evidence; only fill
            # missing sources from warmup training.
            for src, c in result.sources.items():
                self._calibrations.setdefault(src, c)
        except asyncio.CancelledError:
            raise
        except Exception:
            _LOGGER.exception("Bayesian State Filter warmup training failed")

    # ------------------------------------------------------------------
    # Persistence / diagnostics
    # ------------------------------------------------------------------

    async def _save_state(self):
        dyn = None
        if self._dynamics is not None:
            dyn = {
                "tau": self._dynamics.tau,
                "q_acc": self._dynamics.q_acc,
                "p10": self._dynamics.p10,
                "p90": self._dynamics.p90,
                "confidence": self._dynamics.confidence,
                "entropy_confidence": self._dynamics.entropy_confidence,
                "samples": self._dynamics.samples,
                "identifiable": self._dynamics.identifiable,
                "edge_mass": self._dynamics.edge_mass,
                "boundary_limited": self._dynamics.boundary_limited,
            }
        await self._store.async_save({
            "filter": self.filter.dump_state(),
            "sources": {k: v.dump() for k, v in self._calibrations.items()},
            "dynamics": dyn,
            "characteristic": self._characteristic.dump() if self._characteristic is not None else None,
            "level_grid_step": self._level_grid_step,
        })

    async def _restore_fallback(self):
        saved = await self._store.async_load()
        if not saved:
            return
        self.filter.load_state(saved.get("filter", {}))
        self._calibrations = {
            k: SourceCalibration.load(v) for k, v in (saved.get("sources", {}) or {}).items()
        }
        self._source_cal = OnlineSourceCalibrator(self._calibrations)
        d = saved.get("dynamics") or {}
        if d:
            from .core.dynamics import DynamicsEstimate
            try:
                self._dynamics = DynamicsEstimate(**d)
            except Exception:
                self._dynamics = None
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
        velocity = float(self.filter.x[1])

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
        }

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
                ATTR_PROCESS_NOISE: f"{self.filter.q_acc:.6e}",
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

            if self._dynamics is not None:
                attrs.update({
                    ATTR_VELOCITY_TIME_P10: round(self._dynamics.p10, 3),
                    ATTR_VELOCITY_TIME_P90: round(self._dynamics.p90, 3),
                    ATTR_VELOCITY_TIME_CONFIDENCE: round(self._dynamics.confidence, 4),
                    ATTR_VELOCITY_TIME_EDGE_MASS: round(self._dynamics.edge_mass, 4),
                    ATTR_VELOCITY_TIME_BOUNDARY_LIMITED: bool(self._dynamics.boundary_limited),
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
                "calibration_samples": int(c.calibration_samples),
                "calibration_span_s": round(c.calibration_span, 1),
                "calibration_pairs": int(c.calibration_pairs),
                "live_updates": int(c.updates),
            }
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
        # ``auto`` is intentionally conservative in 0.2.1.6 and therefore
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
