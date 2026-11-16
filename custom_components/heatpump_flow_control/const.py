"""Constants for the Heatpump Flow Control integration."""

DOMAIN = "heatpump_flow_control"

# Konfiguration
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
DEFAULT_UPDATE_INTERVAL = 60  # Minuten
DEFAULT_LEARNING_RATE = 0.03
DEFAULT_HYSTERESE = 3.0
# Sicherheitsabstand zusätzlich zur Hysterese, damit der Wert nach dem
# Runden von Tv/Soll nicht unter der Schaltregel landet.
HYSTERESE_MARGIN = 0.5
# Max. Änderung des Sollwerts pro Update-Zyklus (K). Verhindert, dass das
# Modell mit einem falschen Lern-Ereignis den Sollwert um viele Kelvin
# springen lässt und die WP dann stundenlang Vollgas läuft.
DEFAULT_MAX_SCHRITTWEITE = 1.0
# Toleranzband um die Heizkurve-Fallback (K). Vorhersagen außerhalb des
# Bands werden als Modellfehler behandelt und durch den Fallback ersetzt.
# Nicht zu eng: Das Modell darf legitim von der Heizkurve abweichen,
# sobald es echte Daten gelernt hat (Testlernen zeigt Abweichungen ~5.5K
# bei kalten Außentemperaturen). Der Wildwuchs-Sprung 28→39 im Nacht-
# Extrapolationsfall (2026-11-16) lag bei ~14K und fällt trotzdem auf.
DEFAULT_SANITY_BAND = 8.0
DEFAULT_MIN_THERMISCHE_LEISTUNG = 0.6  # kW
DEFAULT_MAX_THERMISCHE_LEISTUNG = 6.0  # kW

# Attribute
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
