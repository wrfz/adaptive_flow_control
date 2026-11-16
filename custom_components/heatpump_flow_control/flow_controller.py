"""Machine Learning Controller for heat pump control."""

from datetime import datetime, timedelta
import logging
import math
import time as time_module
from typing import Any

from river import compose, linear_model, metrics, optim, preprocessing

from .const import DEFAULT_SANITY_BAND
from .types import (
    DateTimeTemperatur,
    Erfahrung,
    Features,
    HistoryBuffer,
    ModelStats,
    SensorValues,
    TempVorlauf,
    VorlaufSollAndFeatures,
    VorlaufSollWeight,
)

# pylint: disable=hass-logger-capital
# ruff: noqa: BLE001

_LOGGER = logging.getLogger(__name__)

# Pickle format version for migration
# Increment this number on incompatible code changes
PICKLE_VERSION = 5

# Weighting for synthetic training data
# Lower value = real data have a stronger influence
# With 0.001: real data dominate (>80%) after ~3 days
# Calculation: 3 days × 48 pred/day × 2.0 weight = 288 vs 70.200 × 0.001 = 70
SYNTHETIC_WEIGHT = 0.1

# Learning dynamics: large correction factor + batch evaluation of all pending
# experiences with the same current deviation caused oscillation.
KORREKTUR_FAKTOR = 1.5          # was 3.0: damped target correction
MAX_WEIGHT = 5.0                # Cap for the learning weight of an experience
KORREKTUR_DECAY_HOURS = 24.0    # the current deviation belongs mainly to young experiences
MIN_KORREKTUR_FACTOR = 0.1      # below: skip experience instead of learning self-confirmation


class FlowController:
    """Flow Controller for Vorlauf temperature control."""

    # Pickle version for migration check
    pickle_version: int = PICKLE_VERSION

    # Model attributes (initialized in _setup())
    model: compose.Pipeline
    metric: metrics.MAE

    def __init__(
        self,
        min_vorlauf: float,
        max_vorlauf: float,
        learning_rate: float,
        min_thermische_leistung: float = 0.6,
        max_thermische_leistung: float = 6.0,
        sanity_band: float = DEFAULT_SANITY_BAND,
    ) -> None:
        """Initialize the flow controller."""

        # Configuration parameters
        self.min_vorlauf = min_vorlauf
        self.max_vorlauf = max_vorlauf
        self.learning_rate = learning_rate
        self.min_thermische_leistung = min_thermische_leistung
        self.max_thermische_leistung = max_thermische_leistung
        self.sanity_band = sanity_band
        self.pickle_version = PICKLE_VERSION  # For migration check
        self.min_reward_hours = 4.0  # Learning from features that are 4h old
        self.short_history_hours = 3.0
        self.longterm_history_days = 7

        # Initialize all attributes (will be set in _setup())
        self._enabled = True  # Default: active, is overwritten by the switch entity
        self.erfahrungs_liste: list[Erfahrung] = []  # Stores all predictions with features
        self.aussen_temp_history: HistoryBuffer = HistoryBuffer()
        # Cumulative counter of all learned experiences (persisted;
        # the list itself discards learned entries on cleanup)
        self.erfahrungen_gelernt_total = 0

    def setup(self, iterations: int = 0) -> None:
        """Initialize the flow controller."""
        _LOGGER.info("setup()")

        # River online learning model
        self.model = compose.Pipeline(
            preprocessing.StandardScaler(),
            linear_model.LinearRegression(
                optimizer=optim.Adam(self.learning_rate)
            ),
        )

        # TIME-BASED history for trend calculation
        self.aussen_temp_history = HistoryBuffer()  # Short history for trends (last 2-3 hours)
        self.short_history_hours = 3.0  # Short history: 3 hours

        # Long-term history (1 week)
        self.longterm_history_days = 7

        # Metrics
        self.metric = metrics.MAE()
        self.erfahrungen_gelernt_total = 0

        # Train model with synthetic data
        self._train_synthetic_data(iterations)

        _LOGGER.info(
            "setup(): min=%s, max=%s, lr=%s",
            self.min_vorlauf,
            self.max_vorlauf,
            self.learning_rate,
        )

    def __getstate__(self) -> dict[str, Any]:
        """Return state for pickling - only save learned state, not configuration.

        This ensures that configuration changes don't cause pickle incompatibility.
        Configuration parameters (min_vorlauf, max_vorlauf, etc.) will be set
        from current config when unpickling.
        """

        _LOGGER.info("__getstate__()")

        return {
            # Pickle version for migration
            "pickle_version": self.pickle_version,

            # Learned model state
            "model": self.model,
            "metric": self.metric,

            # Experience list for delayed learning
            "erfahrungs_liste": self.erfahrungs_liste,

            # Cumulative counter of learned experiences
            "erfahrungen_gelernt_total": self.erfahrungen_gelernt_total,

            # Temperature histories for trend calculation
            "aussen_temp_history": self.aussen_temp_history,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore state from pickle.

        Configuration parameters (min_vorlauf, max_vorlauf, learning_rate, etc.)
        are NOT restored from pickle - they will be set by __init__ with current config.
        """

        _LOGGER.info("__setstate__()")

        self.pickle_version = state.get("pickle_version", PICKLE_VERSION)
        self.model = state["model"]
        self.metric = state["metric"]
        self.erfahrungs_liste = state["erfahrungs_liste"]
        self.aussen_temp_history = state["aussen_temp_history"]
        # Old pickle state without counter: start at 0
        self.erfahrungen_gelernt_total = state.get("erfahrungen_gelernt_total", 0)

        # _enabled is not pickled: initialize it safely here,
        # the switch entity sets the real value in async_added_to_hass().
        self._enabled = True

        # Note: Configuration parameters must be set via update_config() after unpickling

    @property
    def enabled(self):
        """Return whether the controller is enabled."""

        return self._enabled

    @enabled.setter
    def enabled(self, value: bool):
        """Set whether the controller is enabled."""

        self._enabled = value
        _LOGGER.debug("Controller enabled: %d", value)

    def update_config(
        self,
        min_vorlauf: float,
        max_vorlauf: float,
        learning_rate: float,
        min_thermische_leistung: float = 0.6,
        max_thermische_leistung: float = 6.0,
        sanity_band: float = DEFAULT_SANITY_BAND,
    ) -> None:
        """Update configuration parameters after unpickling.

        This is called after unpickling to set configuration parameters
        from the current Home Assistant configuration.

        Args:
            min_vorlauf: Minimum Vorlauf temperature
            max_vorlauf: Maximum Vorlauf temperature
            learning_rate: Learning rate for the model
            min_thermische_leistung: Minimum thermal power in kW
            max_thermische_leistung: Maximum thermal power in kW
            sanity_band: tolerance band around the heating curve fallback in K
        """
        self.min_vorlauf = min_vorlauf
        self.max_vorlauf = max_vorlauf
        self.learning_rate = learning_rate
        self.min_thermische_leistung = min_thermische_leistung
        self.max_thermische_leistung = max_thermische_leistung
        self.sanity_band = sanity_band

        # Set other configuration defaults
        self.min_reward_hours = 4.0
        self.short_history_hours = 3.0
        self.longterm_history_days = 7

        _LOGGER.info(
            "Config updated after unpickling: min=%s, max=%s, lr=%s, power=%s-%s kW",
            self.min_vorlauf,
            self.max_vorlauf,
            self.learning_rate,
            self.min_thermische_leistung,
            self.max_thermische_leistung,
        )

    def _heating_curve_fallback(self, aussen_temp: float, raum_abweichung: float) -> float:
        """Heating curve for synthetic training.

        Typical heating curve: Vorlauf = A - B * outdoor temperature.

        Args:
            aussen_temp: outdoor temperature in degrees Celsius
            raum_abweichung: Positive values mean room is too cold and negative too warm.
        """

        # Heating curve: 15°C → min-vorlauf up to -20°C → max-vorlauf
        p0 = TempVorlauf(15, self.min_vorlauf) # outdoor temperature, Vorlauf
        p1 = TempVorlauf(-20.0, self.max_vorlauf)

        vorlauf = p0.vorlauf + (p1.vorlauf - p0.vorlauf) / (p1.temp - p0.temp) * (aussen_temp - p0.temp)

        # Correction based on room deviation
        vorlauf += raum_abweichung * 2.0

        v_min = min(p0.vorlauf, p1.vorlauf)
        v_max = max(p0.vorlauf, p1.vorlauf)

        return max(v_min, min(vorlauf, v_max))

    def _train_synthetic_data(self, iterations: int) -> None:
        """Train the model with synthetic data from the heating curve."""

        temp_min = -20.0
        temp_max = 20.0
        synthetic_count = 0

        _LOGGER.info("_train_synthetic_data() => Temperature range %.1f to %.1f°C", temp_min, temp_max)

        # Many training runs for precise learning
        if iterations > 0:
            start_time = time_module.time()
            for _ in range(iterations):
                for t_aussen in range(int(temp_min), int(temp_max) + 1):
                    # Temperature range 16-25°C for room setpoint
                    for raum_soll in range(16, 26):
                        # Wide deviations for robust training (also large deviations as in real operation)
                        for raum_abweichung in range(-5, 5 + 1): # -5 to +5°C
                            raum_ist = raum_soll - raum_abweichung

                            # Calculate Vorlauf-Soll from heating curve
                            vorlauf_soll = self._heating_curve_fallback(
                                float(t_aussen), raum_abweichung
                            )

                            # Train the model with synthetic features
                            synthetic_features = Features(
                                aussen_temp=float(t_aussen),
                                raum_abweichung=raum_abweichung,
                                temp_diff=float(t_aussen) - raum_ist,
                                thermische_leistung_relativ=0.5,        # 50% of max. power
                                aussen_trend_1h=0.0,
                                stunde_sin=0.0,
                                stunde_cos=1.0,
                                wochentag_sin=0.0,
                                wochentag_cos=1.0,
                            )

                            self.model.learn_one(
                                synthetic_features.to_dict(),
                                vorlauf_soll,
                                sample_weight=SYNTHETIC_WEIGHT  # Weaker weighting for fast adaptation
                            )
                            synthetic_count += 1

            elapsed_time = time_module.time() - start_time

            _LOGGER.info(
                "Synthetic training completed: %d data points (weight=%.1f) in %.2f seconds",
                synthetic_count,
                SYNTHETIC_WEIGHT,
                elapsed_time,
            )

    def _create_features(
        self,
        sensor_values: SensorValues,
    ) -> Features:
        """Creates the feature dictionary for the model.

        Args:
            aussen_temp: outdoor temperature
            raum_ist: current room temperature
            raum_soll: room setpoint temperature
            vorlauf_ist: current Vorlauf
        """

        _LOGGER.info("_create_features()")

        now = datetime.now()

        self.aussen_temp_history.append(DateTimeTemperatur(timestamp=now, temperature=sensor_values.aussen_temp))

        # Remove entries older than short_history_hours
        cutoff_time = now - timedelta(hours=self.short_history_hours)

        while self.aussen_temp_history and self.aussen_temp_history.first.timestamp < cutoff_time:
            self.aussen_temp_history.popleft()

        # Room deviation
        raum_abweichung = sensor_values.raum_soll - sensor_values.raum_ist

        # Time of day as cyclic feature (default: noon)
        stunde = now.hour + now.minute / 60.0
        stunde_sin = math.sin(2 * math.pi * stunde / 24)
        stunde_cos = math.cos(2 * math.pi * stunde / 24)

        # Weekday as cyclic feature (default: Wednesday)
        wochentag = now.weekday()
        wochentag_sin = math.sin(2 * math.pi * wochentag / 7)
        wochentag_cos = math.cos(2 * math.pi * wochentag / 7)

        # Thermal power features
        thermische_leistung = sensor_values.thermische_leistung or 0.0

        # Relative value: 0 = min, 1 = max, 0.5 = middle
        if self.max_thermische_leistung > self.min_thermische_leistung:
            thermische_leistung_relativ = (
                (thermische_leistung - self.min_thermische_leistung) /
                (self.max_thermische_leistung - self.min_thermische_leistung)
            )
            # Clamp to 0-1 range
            thermische_leistung_relativ = max(0.0, min(1.0, thermische_leistung_relativ))
        else:
            thermische_leistung_relativ = 0.5

        return Features(
            aussen_temp = sensor_values.aussen_temp,
            raum_abweichung = raum_abweichung,
            temp_diff = sensor_values.aussen_temp - sensor_values.raum_ist,
            thermische_leistung_relativ = thermische_leistung_relativ,
            aussen_trend_1h = self.aussen_temp_history.get_trend(hours=1.0),
            stunde_sin = stunde_sin,
            stunde_cos = stunde_cos,
            wochentag_sin = wochentag_sin,
            wochentag_cos = wochentag_cos,
        )

    def _evaluate_erfahrung(
        self,
        erfahrung: Erfahrung,
        raum_ist_jetzt: float,
        age_hours: float,
    ) -> VorlaufSollWeight | None:
        """Calculates the corrected Vorlauf with steady, symmetric weighting.

        Fully continuous function without thresholds or jumps.
        Takes the physical limits of the thermal power into account.

        The current room deviation is weighted with a recency factor:
        the older the experience, the less the deviation belongs to it
        (it reflects many subsequent control decisions). If relevance is too
        low, None is returned and the experience is skipped, instead of
        training the model on its own prediction (self-confirmation).
        """

        # Deviation: positive = too cold, negative = too warm
        abweichung = erfahrung.raum_soll - raum_ist_jetzt
        abweichung_abs = abs(abweichung)

        # Recency: 1.0 = fresh (age ~ min_reward_hours), 0.0 = old
        recency = max(0.0, 1.0 - age_hours / KORREKTUR_DECAY_HOURS)

        if recency < MIN_KORREKTUR_FACTOR:
            _LOGGER.debug(
                "Experience %.1fh old: recency %.2f < %.2f, skipping",
                age_hours,
                recency,
                MIN_KORREKTUR_FACTOR,
            )
            return None

        # Check whether the heat pump was at the limit
        thermische_leistung_relativ = erfahrung.features.thermische_leistung_relativ
        am_oberen_limit = thermische_leistung_relativ > 0.95    # > 95% of max. power
        am_unteren_limit = thermische_leistung_relativ < 0.05   # <  5% of max. power

        weight = min(0.5 + abweichung_abs ** 1.5, MAX_WEIGHT)
        korrektur_faktor = KORREKTUR_FAKTOR

        # Reduce correction when pump was at the limit
        if am_oberen_limit and abweichung > 0:  # Too cold AND at the upper power limit
            # Pump cannot deliver more → strongly reduce correction
            korrektur_faktor *= 0.2
            weight *= 0.5  # Also reduce learning weight
            _LOGGER.debug(
                "Pump at the upper power limit (%.0f%%), reducing correction",
                thermische_leistung_relativ * 100
            )
        elif am_unteren_limit and abweichung < 0:  # Too warm AND at the lower power limit
            # Pump cannot deliver less → strongly reduce correction
            korrektur_faktor *= 0.2
            weight *= 0.5
            _LOGGER.debug(
                "Pump at the lower power limit (%.0f%%), reducing correction",
                thermische_leistung_relativ * 100
            )

        # Damp correction and weight by recency: young experiences
        # get the full correction, old ones a damped one.
        korrektur_faktor *= recency
        weight *= recency

        # Weight factor of the experience (e. g. 0.3 for raum_soll-triggered
        # immediate cycles): damps only the learning weight, never the correction.
        if erfahrung.gewicht_faktor != 1.0:
            weight *= erfahrung.gewicht_faktor
            _LOGGER.debug(
                "Weight factor %.2f applied (weight=%.2f)",
                erfahrung.gewicht_faktor,
                weight,
            )

        korrigierter_vorlauf = erfahrung.vorlauf_soll + abweichung * korrektur_faktor

        # Limit to the physically possible range
        korrigierter_vorlauf = max(
            self.min_vorlauf,
            min(self.max_vorlauf, korrigierter_vorlauf)
        )

        _LOGGER.info(
            "Experience evaluated: dev=%.2f°C, age=%.1fh, recency=%.2f, power=%.0f%%, "
            "Vorlauf %.1f→%.1f°C, weight=%.2f",
            abweichung,
            age_hours,
            recency,
            thermische_leistung_relativ * 100,
            erfahrung.vorlauf_soll,
            korrigierter_vorlauf,
            weight
        )

        return VorlaufSollWeight(vorlauf_soll=korrigierter_vorlauf, weight=weight)

    def learn_from_features(self, current_raum_ist: float) -> dict[str, int]:
        """Learns from features stored 4h ago.

        Args:
            current_raum_ist: current room temperature (for evaluation)

        Returns:
            Statistics about the learning
        """
        stats = {"gelernt": 0, "uebersprungen": 0}

        now = datetime.now()

        # Find experiences that are old enough (4h) and not yet learned
        for erfahrung in self.erfahrungs_liste:
            if erfahrung.gelernt:
                continue

            # Check whether the experience is old enough (4h)
            age_hours = (now - erfahrung.timestamp).total_seconds() / 3600

            if age_hours < self.min_reward_hours:
                stats["uebersprungen"] += 1
                continue

            # Evaluate experience with current room temperature
            try:
                korrigierter_vorlauf_weight = self._evaluate_erfahrung(
                    erfahrung=erfahrung,
                    raum_ist_jetzt=current_raum_ist,
                    age_hours=age_hours,
                )

                if korrigierter_vorlauf_weight is None:
                    # Too old for a meaningful correction:
                    # mark as processed, instead of learning self-confirmation
                    erfahrung.gelernt = True
                    stats["uebersprungen"] += 1
                    continue

                # Learn with the corrected Vorlauf
                self.model.learn_one(
                    erfahrung.features.to_dict(),
                    korrigierter_vorlauf_weight.vorlauf_soll,
                    sample_weight=korrigierter_vorlauf_weight.weight,
                )

                erfahrung.gelernt = True
                stats["gelernt"] += 1

                _LOGGER.info(
                    "✓ Learned from experience from %.1fh ago: Vorlauf %.1f°C → %.1f°C (weight=%.1f)",
                    age_hours,
                    erfahrung.vorlauf_soll,
                    korrigierter_vorlauf_weight.vorlauf_soll,
                    korrigierter_vorlauf_weight.weight,
                )

            except Exception as e:
                _LOGGER.error("Error learning from feature: %s", e)

        # Cleanup: remove learned and very old experiences
        # - Learned experiences are no longer needed
        # - Very old unlearned experiences (older than 7 days) are also removed
        max_age_days = 7
        cutoff_cleanup = now - timedelta(days=max_age_days)
        self.erfahrungs_liste = [
            e for e in self.erfahrungs_liste
            if not e.gelernt and e.timestamp > cutoff_cleanup
        ]

        # Cumulative counter: learned entries disappear from the list,
        # therefore do not count from it
        self.erfahrungen_gelernt_total += stats["gelernt"]

        if stats["gelernt"] > 0:
            _LOGGER.info(
                "Feature learning completed: %d learned, %d skipped, %d in list",
                stats["gelernt"],
                stats["uebersprungen"],
                len(self.erfahrungs_liste),
            )

        return stats

    def calculate_vorlauf_soll(
        self,
        sensor_values: SensorValues,
        gewicht_faktor: float = 1.0,
    ) -> VorlaufSollAndFeatures:
        """Calculates Vorlauf-Soll based on model and features.

        Args:
            sensor_values: sensor values for the calculation
            gewicht_faktor: learning weight factor of the new experience
                (1.0 = normal, < 1 = damped, e. g. immediate cycles after
                raum_soll change)

        Returns:
            VorlaufSollAndFeatures: Vorlauf-Soll and associated features
        """

        _LOGGER.info("calculate_vorlauf_soll()")
        features = self._create_features(sensor_values)

        # Fallback heating curve as reference for the sanity band
        fallback_vorlauf = self._heating_curve_fallback(
            sensor_values.aussen_temp, features.raum_abweichung
        )

        # Use the model for prediction
        try:
            raw_prediction = self.model.predict_one(features.to_dict())
            _LOGGER.info(
                "Model-Prediction: %.2f°C (outdoor: %.1f°C, room-dev: %.2f°C, Vorlauf-Ist: %.1f°C)",
                raw_prediction,
                sensor_values.aussen_temp,
                features.raum_abweichung,
                sensor_values.vorlauf_ist,
            )

            # Sanity band: predictions outside the band around the
            # heating curve are model errors (e. g. extrapolation to
            # feature combinations the training has not seen) and
            # are replaced by the fallback.
            band = self.sanity_band
            if fallback_vorlauf - band <= raw_prediction <= fallback_vorlauf + band:
                vorlauf_soll = raw_prediction
            else:
                _LOGGER.warning(
                    "Model returns unrealistic value %.1f°C (heating curve: %.1f°C, band ±%.1fK), "
                    "using fallback heating curve",
                    raw_prediction,
                    fallback_vorlauf,
                    band,
                )
                vorlauf_soll = fallback_vorlauf

        except Exception as e:
            _LOGGER.error("Error in model prediction: %s, using fallback heating curve", e)
            vorlauf_soll = fallback_vorlauf

        # Limit to the configured range
        vorlauf_soll = max(self.min_vorlauf, min(self.max_vorlauf, vorlauf_soll))

        # Store features for time-delayed learning (in 4h)
        erfahrung = Erfahrung(
            timestamp=datetime.now(),
            features=features,
            vorlauf_soll=vorlauf_soll,
            raum_ist_vorher=sensor_values.raum_ist,
            raum_soll=sensor_values.raum_soll,
            gelernt=False,
            gewicht_faktor=gewicht_faktor,
        )
        self.erfahrungs_liste.append(erfahrung)

        # Learn from features that are 4h old (reward-based learning)
        self.learn_from_features(sensor_values.raum_ist)

        _LOGGER.info(
            "Vorlauf-Soll calculated: %.1f°C (outdoor: %.1f°C, room: %.1f/%.1f°C, Vorlauf-Ist: %.1f°C)",
            vorlauf_soll,
            sensor_values.aussen_temp,
            sensor_values.raum_ist,
            sensor_values.raum_soll,
            sensor_values.vorlauf_ist,
        )

        _LOGGER.info(
            "calculate_vorlauf_soll() min: %.1f°C, max: %.1f°C",
            self.min_vorlauf,
            self.max_vorlauf,
        )  # noqa: HASS_LOGGER_CAPITAL

        return VorlaufSollAndFeatures(vorlauf=vorlauf_soll, features=features)

    def set_delivered_vorlauf(self, vorlauf_soll_geliefert: float) -> None:
        """Updates the most recent experience to the actually delivered value.

        clamp_vorlauf_soll() and limit_schrittweite() only take effect after
        calculate_vorlauf_soll(). So that the 4h reward evaluation is
        attributed to the actually delivered value (and not the unclamped
        prediction), the anchor of the experience must be corrected to the
        delivered value — otherwise the model learns at the wrong anchor
        and the clamping bias comes back via the reward.
        """

        if not self.erfahrungs_liste:
            return

        erfahrung = self.erfahrungs_liste[-1]
        if erfahrung.gelernt:
            return

        if erfahrung.vorlauf_soll != vorlauf_soll_geliefert:
            _LOGGER.info(
                "Experience anchor adjusted to delivered value: %.1f°C -> %.1f°C",
                erfahrung.vorlauf_soll,
                vorlauf_soll_geliefert,
            )
            erfahrung.vorlauf_soll = vorlauf_soll_geliefert

    def get_model_statistics(self) -> ModelStats:
        """Returns model statistics."""
        # Cumulative counter (the list discards learned entries on cleanup)
        wartend = sum(1 for e in self.erfahrungs_liste if not e.gelernt)

        return ModelStats(
            mae = self.metric.get(),
            history_size = len(self.aussen_temp_history),
            erfahrungen_total = len(self.erfahrungs_liste) + self.erfahrungen_gelernt_total,
            erfahrungen_gelernt = self.erfahrungen_gelernt_total,
            erfahrungen_wartend = wartend,
        )
