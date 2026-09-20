DOMAIN = "bayesian_state_filter"

ATTR_VARIANCE = "variance"
ATTR_STDDEV = "stddev"
ATTR_VELOCITY = "velocity"
ATTR_NOISE_VELOCITY = "noise_velocity"
ATTR_INNOVATION = "innovation"
ATTR_INNOVATION_VAR = "innovation_var"
ATTR_P_VALUE = "p_value"
ATTR_Z_SCORE = "z_score"
ATTR_NOISE_MODEL = "noise_model"
ATTR_NOISE_MODEL_MODE = "noise_model_mode"
ATTR_NOISE_MODEL_PARAMS = "noise_model_params"
ATTR_NOISE_VARIANCE_SOURCE = "noise_variance_source"
ATTR_LAST_SOURCE = "last_source"
ATTR_MEASUREMENT_SIGMA = "measurement_sigma"

ATTR_ROBUST_WEIGHT = "robust_weight"

# Characteristic time of the *level process* (v2.1 variogram estimate).
ATTR_CHARACTERISTIC_TIME = "characteristic_time_s"
ATTR_CHARACTERISTIC_TIME_P10 = "characteristic_time_p10_s"
ATTR_CHARACTERISTIC_TIME_P90 = "characteristic_time_p90_s"
ATTR_CHARACTERISTIC_TIME_CONFIDENCE = "characteristic_time_confidence"
ATTR_CHARACTERISTIC_TIME_STATUS = "characteristic_time_status"
ATTR_CHARACTERISTIC_TIME_IDENTIFIABLE = "characteristic_time_identifiable"
ATTR_DYNAMICS_EDGE_MASS = "characteristic_time_edge_mass"
ATTR_DYNAMICS_BOUNDARY_LIMITED = "characteristic_time_boundary_limited"
ATTR_CHARACTERISTIC_NUGGET_VARIANCE = "characteristic_nugget_variance"
ATTR_CHARACTERISTIC_PROCESS_VARIANCE = "characteristic_process_variance"
ATTR_CHARACTERISTIC_SIGNAL_FRACTION = "characteristic_signal_fraction"
ATTR_CHARACTERISTIC_FIT_ERROR = "characteristic_fit_error"
ATTR_CHARACTERISTIC_LAG_COUNT = "characteristic_lag_count"
ATTR_CHARACTERISTIC_PAIR_COUNT = "characteristic_pair_count"
ATTR_CHARACTERISTIC_MIN_LAG = "characteristic_min_lag_s"
ATTR_CHARACTERISTIC_MAX_LAG = "characteristic_max_lag_s"

# Predictive local-slope memory used by the damped-velocity state model.
ATTR_VELOCITY_TIME = "velocity_tau_s"
ATTR_VELOCITY_TIME_P10 = "velocity_tau_p10_s"
ATTR_VELOCITY_TIME_P90 = "velocity_tau_p90_s"
ATTR_VELOCITY_TIME_CONFIDENCE = "velocity_tau_confidence"
ATTR_VELOCITY_TIME_EDGE_MASS = "velocity_tau_edge_mass"
ATTR_VELOCITY_TIME_BOUNDARY_LIMITED = "velocity_tau_boundary_limited"

ATTR_FILTER_MODE = "filter_mode"
ATTR_PROCESS_NOISE = "process_noise_q"
ATTR_SOURCE_HEALTH = "source_health"

ATTR_MEASUREMENT_VARIANCE = "measurement_variance"
ATTR_EFFECTIVE_INNOVATION_VARIANCE = "effective_innovation_variance"
ATTR_UPDATE_DT = "update_dt_s"

# Internal algorithm constants. These are deliberately not user-facing config
# knobs: changing them alters numerical robustness/freshness policy rather than
# the physical model. Keeping them named makes the policy auditable.
NUMERIC_VARIANCE_FLOOR = 1e-15
STUDENT_T_MIN_WEIGHT = 0.05
FRESHNESS_MEDIAN_DT_MULTIPLIER = 3.0
FRESHNESS_TAU_FRACTION = 0.20
FRESHNESS_MIN_S = 300.0
CHARACTERISTIC_REFIT_MIN_S = 300.0


# v0.3 full [x,v,a,j] posterior diagnostics. Derivative values are exposed in
# human-scale per-hour units while the internal state remains per-second.
ATTR_RATE_PER_HOUR = "rate_per_hour"
ATTR_CURVATURE_PER_HOUR2 = "curvature_per_hour2"
ATTR_JERK_PER_HOUR3 = "jerk_per_hour3"
ATTR_RATE_WEIGHT = "rate_weight"
ATTR_CURVATURE_WEIGHT = "curvature_weight"
ATTR_JERK_WEIGHT = "jerk_weight"
ATTR_RATE_Z = "rate_z"
ATTR_CURVATURE_Z = "curvature_z"
ATTR_JERK_Z = "jerk_z"
ATTR_GATED_TIMESCALE = "gated_timescale_s"
ATTR_GATED_LOCAL_RMSE = "gated_local_rmse"
ATTR_GATED_LOCAL_RMSE_STEP1 = "gated_local_rmse_step1"
ATTR_GATED_LOCAL_RMSE_STEP2 = "gated_local_rmse_step2"
