"""Constants for the Adaptive Flow Control integration."""

DOMAIN = "adaptive_flow_control"

# Configuration
CONF_AUSSEN_TEMP_SENSOR = "aussen_temp_sensor"
CONF_RAUM_IST_SENSOR = "raum_ist_sensor"
CONF_RAUM_SOLL_SENSOR = "raum_soll_sensor"
CONF_VORLAUF_IST_SENSOR = "vorlauf_ist_sensor"
CONF_VORLAUF_SOLL_ENTITY = "vorlauf_soll_entity"
CONF_IS_HEATING_ENTITY = "is_heating_entity"
CONF_THERMISCHE_LEISTUNG_SENSOR = "thermische_leistung_sensor"
CONF_MIN_VORLAUF = "min_vorlauf"
CONF_MAX_VORLAUF = "max_vorlauf"
CONF_MIN_THERMISCHE_LEISTUNG = "min_thermische_leistung"
CONF_MAX_THERMISCHE_LEISTUNG = "max_thermische_leistung"
CONF_UPDATE_INTERVAL = "update_interval"
CONF_LEARNING_RATE = "learning_rate"
CONF_HYSTERESE = "hysterese"
CONF_MAX_SCHRITTWEITE = "max_schrittweite"
CONF_SANITY_BAND = "sanity_band"

# Defaults
DEFAULT_MIN_VORLAUF_LO = 25.0
DEFAULT_MIN_VORLAUF_HI = 32.0
DEFAULT_MAX_VORLAUF_LO = 35.0
DEFAULT_MAX_VORLAUF_HI = 55.0
DEFAULT_UPDATE_INTERVAL = 60  # minutes
DEFAULT_LEARNING_RATE = 0.03
DEFAULT_HYSTERESE = 3.0
# Safety margin added to the hysteresis so that the rounded Tv/soll value
# does not fall below the switching rule.
HYSTERESE_MARGIN = 0.5
# Max. change of the target value per update cycle (K). Prevents the model
# from letting the target value jump by many kelvin due to one wrong
# learning event, making the heat pump run at full power for hours.
DEFAULT_MAX_SCHRITTWEITE = 1.0
# Tolerance band around the heating-curve fallback (K). Predictions outside
# the band are treated as model errors and replaced by the fallback.
# Not too tight: the model may legitimately deviate from the heating curve
# once it has learned real data (test learning shows deviations of ~5.5K
# at cold outdoor temperatures). The erratic jump 28→39 in the night
# extrapolation case (2026-11-16) was ~14K and still falls within it.
DEFAULT_SANITY_BAND = 8.0
DEFAULT_MIN_THERMISCHE_LEISTUNG = 0.6  # kW
DEFAULT_MAX_THERMISCHE_LEISTUNG = 6.0  # kW

# Debounce time for the immediate update after a raum_soll change (s).
# Otherwise, slider drags and schedule changes fire a chain of state
# changes; without a wait, each of them would trigger a full control
# cycle with a new experience.
RAUM_SOLL_DEBOUNCE_SECONDS = 60

# Learning-weight factor for experiences from raum_soll-triggered cycles.
# A target-value change carries the same information as a single
# adjustment to the mean — otherwise multiple cycles within a short time
# would pull into the model amplified N-fold (each experience is a full
# learning step). A factor < 1 damps this; hourly cycles stay at 1.0.
RAUM_SOLL_TRIGGER_GEWICHT = 0.3

# Attributes
ATTR_AUSSEN_TEMP = "aussen_temp"
ATTR_RAUM_IST = "raum_ist"
ATTR_RAUM_SOLL = "raum_soll"
ATTR_VORLAUF_IST = "vorlauf_ist"
ATTR_RAUM_ABWEICHUNG = "raum_abweichung"
ATTR_AUSSEN_TREND_1H = "aussen_trend_1h"
ATTR_MODEL_MAE = "model_mae"
ATTR_PREDICTIONS_COUNT = "predictions_count"
ATTR_LAST_UPDATE = "last_update"
ATTR_NEXT_UPDATE = "next_update"

CONF_ACTIVE_SWITCH = "active"
