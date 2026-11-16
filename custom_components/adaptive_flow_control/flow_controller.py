"""Machine Learning Controller for heat pump control."""

from datetime import datetime, timedelta
import logging
import math
import time as time_module
from typing import Any

import numpy as np

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
PICKLE_VERSION = 6

# Weighting for synthetic training data
SYNTHETIC_WEIGHT = 0.1

# Learning dynamics
KORREKTUR_FAKTOR = 1.5
MAX_WEIGHT = 5.0
KORREKTUR_DECAY_HOURS = 24.0
MIN_KORREKTUR_FACTOR = 0.1

# Feature names in fixed order (must match Features dataclass)
FEATURE_NAMES = [
    "aussen_temp",
    "raum_abweichung",
    "temp_diff",
    "thermische_leistung_relativ",
    "aussen_trend_1h",
    "stunde_sin",
    "stunde_cos",
    "wochentag_sin",
    "wochentag_cos",
]

N_FEATURES = len(FEATURE_NAMES)


def _features_to_array(features: Features) -> np.ndarray:
    """Convert Features dataclass to a 1-D numpy array."""
    d = features.to_dict()
    return np.array([d[k] for k in FEATURE_NAMES], dtype=float)


class _SimpleMAE:
    """Lightweight MAE tracker (river-metric compatible)."""

    def __init__(self):
        self._sum = 0.0
        self._count = 0

    def update(self, y_true, y_pred):
        self._sum += abs(float(y_true) - float(y_pred))
        self._count += 1

    def get(self):
        return self._sum / self._count if self._count > 0 else 0.0


class FlowController:
    """Flow Controller for Vorlauf temperature control."""

    pickle_version: int = PICKLE_VERSION

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

        self.min_vorlauf = min_vorlauf
        self.max_vorlauf = max_vorlauf
        self.learning_rate = learning_rate
        self.ridge_lambda = max(learning_rate, 1e-4)
        self.min_thermische_leistung = min_thermische_leistung
        self.max_thermische_leistung = max_thermische_leistung
        self.sanity_band = sanity_band
        self.pickle_version = PICKLE_VERSION
        self.min_reward_hours = 4.0
        self.short_history_hours = 3.0
        self.longterm_history_days = 7

        # Model state (NumPy)
        self.weights = np.zeros(N_FEATURES)
        self.intercept = (min_vorlauf + max_vorlauf) / 2.0
        self.scaler_mean = np.zeros(N_FEATURES)
        self.scaler_std = np.ones(N_FEATURES)

        # Data
        self.erfahrungs_liste: list[Erfahrung] = []
        self.aussen_temp_history: HistoryBuffer = HistoryBuffer()
        self.erfahrungen_gelernt_total = 0

        # Synthetic data (arrays)
        self._synthetic_X: np.ndarray | None = None
        self._synthetic_y: np.ndarray | None = None
        self._synthetic_w: np.ndarray | None = None

        # Metric (running MAE)
        self.metric = _SimpleMAE()

        self._enabled = True

    def setup(self, iterations: int = 0) -> None:
        """Initialize the flow controller."""
        _LOGGER.info("setup()")

        self.aussen_temp_history = HistoryBuffer()
        self.short_history_hours = 3.0
        self.longterm_history_days = 7
        self.metric = _SimpleMAE()
        self.erfahrungen_gelernt_total = 0

        self._generate_synthetic_data()
        self._train()

        _LOGGER.info(
            "setup(): min=%s, max=%s, lr=%s",
            self.min_vorlauf,
            self.max_vorlauf,
            self.learning_rate,
        )

    def __getstate__(self) -> dict[str, Any]:
        """Return state for pickling."""
        return {
            "pickle_version": self.pickle_version,
            "weights": self.weights.tolist(),
            "intercept": float(self.intercept),
            "scaler_mean": self.scaler_mean.tolist(),
            "scaler_std": self.scaler_std.tolist(),
            "erfahrungs_liste": self.erfahrungs_liste,
            "erfahrungen_gelernt_total": self.erfahrungen_gelernt_total,
            "aussen_temp_history": self.aussen_temp_history,
            "_mae_sum": self.metric._sum,
            "_mae_count": self.metric._count,
            "min_vorlauf": self.min_vorlauf,
            "max_vorlauf": self.max_vorlauf,
            "learning_rate": self.learning_rate,
            "ridge_lambda": self.ridge_lambda,
            "min_thermische_leistung": self.min_thermische_leistung,
            "max_thermische_leistung": self.max_thermische_leistung,
            "sanity_band": self.sanity_band,
            "_enabled": self._enabled,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore state from pickle."""
        self.pickle_version = state.get("pickle_version", PICKLE_VERSION)
        self.weights = np.array(state["weights"], dtype=float)
        self.intercept = float(state["intercept"])
        self.scaler_mean = np.array(state["scaler_mean"], dtype=float)
        self.scaler_std = np.array(state["scaler_std"], dtype=float)
        self.erfahrungs_liste = state["erfahrungs_liste"]
        self.erfahrungen_gelernt_total = state.get("erfahrungen_gelernt_total", 0)
        self.aussen_temp_history = state["aussen_temp_history"]
        self.metric = _SimpleMAE()
        self.metric._sum = state.get("_mae_sum", 0.0)
        self.metric._count = state.get("_mae_count", 0)
        self.min_vorlauf = state.get("min_vorlauf", 25.0)
        self.max_vorlauf = state.get("max_vorlauf", 40.0)
        self.learning_rate = state.get("learning_rate", 0.01)
        self.ridge_lambda = state.get("ridge_lambda", max(self.learning_rate, 1e-4))
        self.min_thermische_leistung = state.get("min_thermische_leistung", 0.6)
        self.max_thermische_leistung = state.get("max_thermische_leistung", 6.0)
        self.sanity_band = state.get("sanity_band", DEFAULT_SANITY_BAND)
        self.min_reward_hours = 4.0
        self.short_history_hours = 3.0
        self.longterm_history_days = 7
        self._enabled = state.get("_enabled", True)
        self._synthetic_X = None
        self._synthetic_y = None
        self._synthetic_w = None

    @property
    def enabled(self):
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool):
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
        self.min_vorlauf = min_vorlauf
        self.max_vorlauf = max_vorlauf
        self.learning_rate = learning_rate
        self.ridge_lambda = max(learning_rate, 1e-4)
        self.min_thermische_leistung = min_thermische_leistung
        self.max_thermische_leistung = max_thermische_leistung
        self.sanity_band = sanity_band
        self.min_reward_hours = 4.0
        self.short_history_hours = 3.0
        self.longterm_history_days = 7
        _LOGGER.info(
            "Config updated: min=%s, max=%s, lr=%s, power=%s-%s kW",
            self.min_vorlauf,
            self.max_vorlauf,
            self.learning_rate,
            self.min_thermische_leistung,
            self.max_thermische_leistung,
        )

    def _heating_curve_fallback(self, aussen_temp: float, raum_abweichung: float) -> float:
        p0 = TempVorlauf(15, self.min_vorlauf)
        p1 = TempVorlauf(-20.0, self.max_vorlauf)
        vorlauf = p0.vorlauf + (p1.vorlauf - p0.vorlauf) / (p1.temp - p0.temp) * (aussen_temp - p0.temp)
        vorlauf += raum_abweichung * 2.0
        v_min = min(p0.vorlauf, p1.vorlauf)
        v_max = max(p0.vorlauf, p1.vorlauf)
        return max(v_min, min(vorlauf, v_max))

    def _generate_synthetic_data(self) -> None:
        """Generate synthetic training data as NumPy arrays."""
        temp_min = -20.0
        temp_max = 20.0
        rows = []
        targets = []
        for t_aussen in range(int(temp_min), int(temp_max) + 1):
            for raum_soll in range(16, 26):
                for raum_abweichung in range(-5, 6):
                    raum_ist = raum_soll - raum_abweichung
                    vorlauf_soll = self._heating_curve_fallback(
                        float(t_aussen), raum_abweichung
                    )
                    features = Features(
                        aussen_temp=float(t_aussen),
                        raum_abweichung=raum_abweichung,
                        temp_diff=float(t_aussen) - raum_ist,
                        thermische_leistung_relativ=0.5,
                        aussen_trend_1h=0.0,
                        stunde_sin=0.0,
                        stunde_cos=1.0,
                        wochentag_sin=0.0,
                        wochentag_cos=1.0,
                    )
                    rows.append(_features_to_array(features))
                    targets.append(vorlauf_soll)

        self._synthetic_X = np.array(rows, dtype=float)
        self._synthetic_y = np.array(targets, dtype=float)
        self._synthetic_w = np.full(len(targets), SYNTHETIC_WEIGHT, dtype=float)
        _LOGGER.info("_generate_synthetic_data(): %d points", len(targets))

    def _create_features(self, sensor_values: SensorValues) -> Features:
        _LOGGER.info("_create_features()")

        now = datetime.now()
        self.aussen_temp_history.append(
            DateTimeTemperatur(timestamp=now, temperature=sensor_values.aussen_temp)
        )
        cutoff_time = now - timedelta(hours=self.short_history_hours)
        while self.aussen_temp_history and self.aussen_temp_history.first.timestamp < cutoff_time:
            self.aussen_temp_history.popleft()

        raum_abweichung = sensor_values.raum_soll - sensor_values.raum_ist
        stunde = now.hour + now.minute / 60.0
        stunde_sin = math.sin(2 * math.pi * stunde / 24)
        stunde_cos = math.cos(2 * math.pi * stunde / 24)
        wochentag = now.weekday()
        wochentag_sin = math.sin(2 * math.pi * wochentag / 7)
        wochentag_cos = math.cos(2 * math.pi * wochentag / 7)

        thermische_leistung = sensor_values.thermische_leistung or 0.0
        if self.max_thermische_leistung > self.min_thermische_leistung:
            thermische_leistung_relativ = (
                (thermische_leistung - self.min_thermische_leistung)
                / (self.max_thermische_leistung - self.min_thermische_leistung)
            )
            thermische_leistung_relativ = max(0.0, min(1.0, thermische_leistung_relativ))
        else:
            thermische_leistung_relativ = 0.5

        return Features(
            aussen_temp=sensor_values.aussen_temp,
            raum_abweichung=raum_abweichung,
            temp_diff=sensor_values.aussen_temp - sensor_values.raum_ist,
            thermische_leistung_relativ=thermische_leistung_relativ,
            aussen_trend_1h=self.aussen_temp_history.get_trend(hours=1.0),
            stunde_sin=stunde_sin,
            stunde_cos=stunde_cos,
            wochentag_sin=wochentag_sin,
            wochentag_cos=wochentag_cos,
        )

    def _evaluate_erfahrung(
        self,
        erfahrung: Erfahrung,
        raum_ist_jetzt: float,
        age_hours: float,
    ) -> VorlaufSollWeight | None:
        abweichung = erfahrung.raum_soll - raum_ist_jetzt
        abweichung_abs = abs(abweichung)

        recency = max(0.0, 1.0 - age_hours / KORREKTUR_DECAY_HOURS)
        if recency < MIN_KORREKTUR_FACTOR:
            _LOGGER.debug(
                "Experience %.1fh old: recency %.2f < %.2f, skipping",
                age_hours, recency, MIN_KORREKTUR_FACTOR,
            )
            return None

        thermische_leistung_relativ = erfahrung.features.thermische_leistung_relativ
        am_oberen_limit = thermische_leistung_relativ > 0.95
        am_unteren_limit = thermische_leistung_relativ < 0.05

        weight = min(0.5 + abweichung_abs ** 1.5, MAX_WEIGHT)
        korrektur_faktor = KORREKTUR_FAKTOR

        if am_oberen_limit and abweichung > 0:
            korrektur_faktor *= 0.2
            weight *= 0.5
            _LOGGER.debug("Pump at upper power limit, reducing correction")
        elif am_unteren_limit and abweichung < 0:
            korrektur_faktor *= 0.2
            weight *= 0.5
            _LOGGER.debug("Pump at lower power limit, reducing correction")

        korrektur_faktor *= recency
        weight *= recency

        if erfahrung.gewicht_faktor != 1.0:
            weight *= erfahrung.gewicht_faktor
            _LOGGER.debug(
                "Weight factor %.2f applied (weight=%.2f)",
                erfahrung.gewicht_faktor, weight,
            )

        korrigierter_vorlauf = erfahrung.vorlauf_soll + abweichung * korrektur_faktor
        korrigierter_vorlauf = max(
            self.min_vorlauf, min(self.max_vorlauf, korrigierter_vorlauf)
        )

        _LOGGER.info(
            "Experience evaluated: dev=%.2f°C, age=%.1fh, recency=%.2f, power=%.0f%%, "
            "Vorlauf %.1f→%.1f°C, weight=%.2f",
            abweichung, age_hours, recency,
            thermische_leistung_relativ * 100,
            erfahrung.vorlauf_soll, korrigierter_vorlauf, weight,
        )

        return VorlaufSollWeight(vorlauf_soll=korrigierter_vorlauf, weight=weight)

    def learn_from_features(self, current_raum_ist: float) -> dict[str, int]:
        stats = {"gelernt": 0, "uebersprungen": 0}
        now = datetime.now()

        for erfahrung in self.erfahrungs_liste:
            if erfahrung.gelernt:
                continue

            age_hours = (now - erfahrung.timestamp).total_seconds() / 3600
            if age_hours < self.min_reward_hours:
                stats["uebersprungen"] += 1
                continue

            try:
                result = self._evaluate_erfahrung(
                    erfahrung=erfahrung,
                    raum_ist_jetzt=current_raum_ist,
                    age_hours=age_hours,
                )

                if result is None:
                    erfahrung.gelernt = True
                    stats["uebersprungen"] += 1
                    continue

                erfahrung.gelernt = True
                erfahrung.learned_target = result.vorlauf_soll
                erfahrung.learned_weight = result.weight
                stats["gelernt"] += 1

                self.metric.update(result.vorlauf_soll, erfahrung.vorlauf_soll)

                _LOGGER.info(
                    "✓ Learned from experience from %.1fh ago: Vorlauf %.1f°C → %.1f°C (weight=%.1f)",
                    age_hours, erfahrung.vorlauf_soll,
                    result.vorlauf_soll, result.weight,
                )

            except Exception as e:
                _LOGGER.error("Error learning from feature: %s", e)

        # Cleanup
        max_age_days = 7
        cutoff_cleanup = now - timedelta(days=max_age_days)
        self.erfahrungs_liste = [
            e for e in self.erfahrungs_liste
            if not e.gelernt or e.timestamp > cutoff_cleanup
        ]
        # Keep learned experiences for training (they have learned_target)
        # but remove very old ones
        self.erfahrungs_liste = [
            e for e in self.erfahrungs_liste
            if e.timestamp > cutoff_cleanup
        ]

        self.erfahrungen_gelernt_total += stats["gelernt"]

        # Retrain model with updated data
        self._train()

        if stats["gelernt"] > 0:
            _LOGGER.info(
                "Feature learning completed: %d learned, %d skipped, %d in list",
                stats["gelernt"], stats["uebersprungen"],
                len(self.erfahrungs_liste),
            )

        return stats

    def _train(self) -> None:
        """Retrain the Ridge regression model on all available data."""
        X_parts = []
        y_parts = []
        w_parts = []

        if self._synthetic_X is not None:
            X_parts.append(self._synthetic_X)
            y_parts.append(self._synthetic_y)
            w_parts.append(self._synthetic_w)

        for erf in self.erfahrungs_liste:
            if erf.gelernt and erf.learned_target is not None and erf.learned_weight is not None:
                x = _features_to_array(erf.features).reshape(1, -1)
                X_parts.append(x)
                y_parts.append(np.array([erf.learned_target]))
                w_parts.append(np.array([erf.learned_weight]))

        if hasattr(self, "_direct_X") and self._direct_X:
            X_parts.append(np.array(self._direct_X, dtype=float))
            y_parts.append(np.array(self._direct_y, dtype=float))
            w_parts.append(np.array(self._direct_w, dtype=float))

        if not X_parts:
            self.weights = np.zeros(N_FEATURES)
            self.intercept = (self.min_vorlauf + self.max_vorlauf) / 2.0
            self.scaler_mean = np.zeros(N_FEATURES)
            self.scaler_std = np.ones(N_FEATURES)
            return

        X = np.vstack(X_parts)
        y = np.concatenate(y_parts)
        w = np.concatenate(w_parts)

        # Weighted scaler params
        w_sum = w.sum()
        self.scaler_mean = np.average(X, axis=0, weights=w)
        var = np.average((X - self.scaler_mean) ** 2, axis=0, weights=w)
        self.scaler_std = np.sqrt(var)
        self.scaler_std = np.where(self.scaler_std < 1e-8, 1.0, self.scaler_std)

        X_scaled = (X - self.scaler_mean) / self.scaler_std

        # Ridge with intercept (augmented matrix, no regularization on intercept)
        n_samples, n_features = X_scaled.shape
        X_aug = np.column_stack([np.ones(n_samples), X_scaled])

        W_sqrt = np.sqrt(np.clip(w, 0, None))
        Xw = X_aug * W_sqrt[:, None]
        yw = y * W_sqrt

        reg = np.eye(n_features + 1)
        reg[0, 0] = 0.0
        reg *= self.ridge_lambda

        A = Xw.T @ Xw + reg
        b = Xw.T @ yw
        try:
            beta = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            beta = np.linalg.lstsq(A, b, rcond=None)[0]

        self.intercept = float(beta[0])
        self.weights = beta[1:]

        # Do not overwrite metric here: it tracks learning errors,
        # not training-fit residuals.

    def calculate_vorlauf_soll(
        self,
        sensor_values: SensorValues,
        gewicht_faktor: float = 1.0,
    ) -> VorlaufSollAndFeatures:
        _LOGGER.info("calculate_vorlauf_soll()")
        features = self._create_features(sensor_values)

        fallback_vorlauf = self._heating_curve_fallback(
            sensor_values.aussen_temp, features.raum_abweichung
        )

        # Predict
        try:
            x = _features_to_array(features)
            x_scaled = (x - self.scaler_mean) / self.scaler_std
            raw_prediction = float(x_scaled @ self.weights + self.intercept)
            _LOGGER.info(
                "Model-Prediction: %.2f°C (outdoor: %.1f°C, room-dev: %.2f°C, Vorlauf-Ist: %.1f°C)",
                raw_prediction, sensor_values.aussen_temp,
                features.raum_abweichung, sensor_values.vorlauf_ist,
            )

            band = self.sanity_band
            if fallback_vorlauf - band <= raw_prediction <= fallback_vorlauf + band:
                vorlauf_soll = raw_prediction
            else:
                _LOGGER.warning(
                    "Model returns unrealistic value %.1f°C (heating curve: %.1f°C, band ±%.1fK), "
                    "using fallback heating curve",
                    raw_prediction, fallback_vorlauf, band,
                )
                vorlauf_soll = fallback_vorlauf
        except Exception as e:
            _LOGGER.error("Error in model prediction: %s, using fallback heating curve", e)
            vorlauf_soll = fallback_vorlauf

        vorlauf_soll = max(self.min_vorlauf, min(self.max_vorlauf, vorlauf_soll))

        erfahrung = Erfahrung(
            timestamp=datetime.now(),
            features=features,
            vorlauf_soll=vorlauf_soll,
            raum_soll=sensor_values.raum_soll,
            gelernt=False,
            gewicht_faktor=gewicht_faktor,
        )
        self.erfahrungs_liste.append(erfahrung)

        self.learn_from_features(sensor_values.raum_ist)

        _LOGGER.info(
            "Vorlauf-Soll calculated: %.1f°C (outdoor: %.1f°C, room: %.1f/%.1f°C, Vorlauf-Ist: %.1f°C)",
            vorlauf_soll, sensor_values.aussen_temp,
            sensor_values.raum_ist, sensor_values.raum_soll,
            sensor_values.vorlauf_ist,
        )

        _LOGGER.info(
            "calculate_vorlauf_soll() min: %.1f°C, max: %.1f°C",
            self.min_vorlauf, self.max_vorlauf,
        )

        return VorlaufSollAndFeatures(vorlauf=vorlauf_soll, features=features)

    def set_delivered_vorlauf(self, vorlauf_soll_geliefert: float) -> None:
        if not self.erfahrungs_liste:
            return
        erfahrung = self.erfahrungs_liste[-1]
        if erfahrung.gelernt:
            return
        if erfahrung.vorlauf_soll != vorlauf_soll_geliefert:
            _LOGGER.info(
                "Experience anchor adjusted to delivered value: %.1f°C -> %.1f°C",
                erfahrung.vorlauf_soll, vorlauf_soll_geliefert,
            )
            erfahrung.vorlauf_soll = vorlauf_soll_geliefert

    def predict_one(self, x) -> float:
        """Predict from a feature dict or Features object (river-compat)."""
        if isinstance(x, dict):
            arr = np.array([x[k] for k in FEATURE_NAMES], dtype=float)
        elif isinstance(x, Features):
            arr = _features_to_array(x)
        else:
            arr = np.asarray(x, dtype=float).ravel()
        x_scaled = (arr - self.scaler_mean) / self.scaler_std
        return float(x_scaled @ self.weights + self.intercept)

    def learn_one(self, x, y, sample_weight: float = 1.0) -> None:
        """Add a training example and retrain (river-compat)."""
        if isinstance(x, dict):
            arr = np.array([x[k] for k in FEATURE_NAMES], dtype=float)
        elif isinstance(x, Features):
            arr = _features_to_array(x)
        else:
            arr = np.asarray(x, dtype=float).ravel()
        if not hasattr(self, "_direct_X"):
            self._direct_X = []
            self._direct_y = []
            self._direct_w = []
        self._direct_X.append(arr)
        self._direct_y.append(float(y))
        self._direct_w.append(float(sample_weight))
        self._train()

    def get_model_statistics(self) -> ModelStats:
        wartend = sum(1 for e in self.erfahrungs_liste if not e.gelernt)
        mae = self.metric.get()
        return ModelStats(
            mae=mae,
            history_size=len(self.aussen_temp_history),
            erfahrungen_total=len(self.erfahrungs_liste) + self.erfahrungen_gelernt_total,
            erfahrungen_gelernt=self.erfahrungen_gelernt_total,
            erfahrungen_wartend=wartend,
        )

    def metric_get(self) -> float:
        return self.metric.get()