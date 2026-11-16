"""Shared types."""

from collections import deque
from collections.abc import Iterator
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta
from itertools import islice
from typing import Any


@dataclass
class TempVorlauf:
    """Temperature and Vorlauf pair."""

    temp: float
    vorlauf: float

@dataclass
class DateTimeTemperatur:
    """Date-time and temperature pair."""
    timestamp: datetime
    temperature: float

class HistoryBuffer(deque):
    """A deque with convenient access to the first and last element."""

    @property
    def first(self) -> DateTimeTemperatur:
        """Returns the oldest element or None if empty."""
        return self[0]

    @property
    def last(self) -> DateTimeTemperatur:
        """Returns the newest element or None if empty."""
        return self[-1]

    @property
    def prev(self) -> DateTimeTemperatur:
        """Returns the newest element or None if empty."""
        return self[-2]

    @property
    def is_empty(self) -> bool:
        """Returns True if the buffer is empty."""
        return not self

    def reversed_without_latest(self):
            """Iterates backwards over all elements except the newest."""
            if len(self) <= 1:
                return iter([])
            # Creates an iterator that skips the last element
            return reversed(list(islice(self, 0, len(self) - 1)))

    def get_trend(self, hours: float) -> float:
            """Calculates the temperature change in kelvin per hour (K/h).

            Searches for the point closest to the time window.
            Returns None if there is not enough data.
            """
            if len(self) < 2:
                return 0.0

            latest = self.last
            # We search for the reference point X hours before the most recent entry
            cutoff = latest.timestamp - timedelta(hours=hours)

            # Search backwards for the first entry that is older than or equal to the cutoff
            reference_entry = self.first  # Defaults to the oldest value
            for entry in reversed(self):
                if entry.timestamp <= cutoff:
                    reference_entry = entry
                    break

            # Zeitdifferenz berechnen
            time_diff_h = (latest.timestamp - reference_entry.timestamp) / timedelta(hours=1)

            if time_diff_h == 0:
                return 0.0

            temp_diff = latest.temperature - reference_entry.temperature
            return temp_diff / time_diff_h

@dataclass
class SensorValues:
    """Input values for Vorlauf-Soll calculation."""
    aussen_temp: float
    raum_ist: float
    raum_soll: float
    vorlauf_ist: float
    thermische_leistung: float | None = None  # Optional: current thermal power in kW

@dataclass
class ModelStats:
    """Model statsistics."""
    mae: float
    history_size: int
    erfahrungen_total: int
    erfahrungen_gelernt: int
    erfahrungen_wartend: int

    def get(self, key: str, default: Any = None) -> Any:
        """Simulates dictionary behavior."""
        return getattr(self, key, default)

@dataclass
class Trends:
    """Model longterm trends."""

    aussen_trend_1h: float = 0.0
    aussen_trend_2h: float = 0.0
    aussen_trend_3h: float = 0.0
    aussen_trend_6h: float = 0.0

@dataclass
class Features:
    """Model features."""

    aussen_temp: float
    raum_abweichung: float
    temp_diff: float                    # t_aussen - raum_ist
    thermische_leistung_relativ: float  # Relative Leistung (0-1)
    aussen_trend_1h: float
    stunde_sin: float
    stunde_cos: float
    wochentag_sin: float
    wochentag_cos: float

    def to_dict(self) -> dict[str, float]:
        """Returns the features as a dictionary."""
        return asdict(self)

    def items(self) -> Iterator[tuple[str, float]]:
        """Returns an iterable list of key-value pairs."""
        return iter(asdict(self).items())

    def copy(self) -> "Features":
        """Creates an exact copy of the current features instance."""
        return replace(self)

    def get(self, key: str, default: Any = None) -> Any:
        """Simulates dictionary behavior."""
        return getattr(self, key, default)

@dataclass
class Erfahrung:
    """Experience for the model."""

    timestamp: datetime
    features: Features
    vorlauf_soll: float
    raum_ist_vorher: float
    raum_soll: float
    gelernt: bool = False
    # Weight factor for the learning step (1.0 = normal, < 1 = damped,
    # e.g. experiences from raum_soll-triggered immediate cycles).
    gewicht_faktor: float = 1.0

@dataclass
class VorlaufSollWeight:
    """Vorlauf-Soll with associated weight."""

    vorlauf_soll: float
    weight: float

@dataclass
class VorlaufSollAndFeatures:
    """Vorlauf and associated features."""

    vorlauf: float
    features: Features
