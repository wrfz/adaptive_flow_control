"""JSON-based snapshot store for the FlowController.

Stores the learned state as JSON: scaler params, model weights, experiences,
and temperature history. No river dependency.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .const import DEFAULT_SANITY_BAND
from .flow_controller import (
    FEATURE_NAMES,
    N_FEATURES,
    FlowController,
    _features_to_array,
)
from .types import (
    DateTimeTemperatur,
    Erfahrung,
    Features,
    HistoryBuffer,
)

_LOGGER = logging.getLogger(__name__)

MODEL_SCHEMA_VERSION = 3

FEATURE_KEYS = list(Features.__dataclass_fields__)


def _temp_als_json(entry: DateTimeTemperatur) -> dict[str, Any]:
    return {
        "timestamp": entry.timestamp.isoformat(),
        "temperature": entry.temperature,
    }


def _temp_aus_json(data: dict[str, Any]) -> DateTimeTemperatur:
    return DateTimeTemperatur(
        timestamp=datetime.fromisoformat(data["timestamp"]),
        temperature=float(data["temperature"]),
    )


def _erfahrung_als_json(erfahrung: Erfahrung) -> dict[str, Any]:
    return {
        "timestamp": erfahrung.timestamp.isoformat(),
        "features": erfahrung.features.to_dict(),
        "vorlauf_soll": erfahrung.vorlauf_soll,
        "raum_soll": erfahrung.raum_soll,
        "gelernt": erfahrung.gelernt,
        "gewicht_faktor": erfahrung.gewicht_faktor,
        "learned_target": erfahrung.learned_target,
        "learned_weight": erfahrung.learned_weight,
    }


def _erfahrung_aus_json(data: dict[str, Any]) -> Erfahrung:
    return Erfahrung(
        timestamp=datetime.fromisoformat(data["timestamp"]),
        features=Features(**{k: data["features"][k] for k in FEATURE_KEYS}),
        vorlauf_soll=float(data["vorlauf_soll"]),
        raum_soll=float(data["raum_soll"]),
        gelernt=bool(data["gelernt"]),
        gewicht_faktor=float(data.get("gewicht_faktor", 1.0)),
        learned_target=(
            float(data["learned_target"]) if data.get("learned_target") is not None else None
        ),
        learned_weight=(
            float(data["learned_weight"]) if data.get("learned_weight") is not None else None
        ),
    )


def controller_to_snapshot(controller: FlowController) -> dict[str, Any]:
    """Create a JSON-serializable snapshot of the learned state."""
    return {
        "schema_version": MODEL_SCHEMA_VERSION,
        "scaler_mean": [float(v) for v in controller.scaler_mean],
        "scaler_std": [float(v) for v in controller.scaler_std],
        "model": {
            "weights": [float(v) for v in controller.weights],
            "intercept": float(controller.intercept),
        },
        "metric": {
            "mae_sum": float(controller.metric._sum),
            "mae_count": int(controller.metric._count),
        },
        "erfahrungen": [_erfahrung_als_json(e) for e in controller.erfahrungs_liste],
        "erfahrungen_gelernt_total": int(controller.erfahrungen_gelernt_total),
        "max_age_days": int(controller.max_age_days),
        "history": [_temp_als_json(t) for t in controller.aussen_temp_history],
    }


def _migrate_v2_to_v3(snapshot: dict[str, Any], controller: FlowController) -> None:
    """Migrate a v2 (river) snapshot to v3 (numpy) in-place on the controller."""
    # Extract scaler params from river format
    scaler_data = snapshot.get("scaler", {})
    means = scaler_data.get("means", {})
    vars_ = scaler_data.get("vars", {})
    scaler_mean = np.array([float(means.get(k, 0.0)) for k in FEATURE_NAMES])
    scaler_var = np.array([float(vars_.get(k, 1.0)) for k in FEATURE_NAMES])
    scaler_std = np.sqrt(np.where(scaler_var < 1e-12, 1.0, scaler_var))

    # Extract model weights from river format
    model_data = snapshot.get("model", {})
    weights_dict = model_data.get("weights", {})
    weights = np.array([float(weights_dict.get(k, 0.0)) for k in FEATURE_NAMES])
    intercept = float(model_data.get("intercept", 0.0))

    controller.scaler_mean = scaler_mean
    controller.scaler_std = scaler_std
    controller.weights = weights
    controller.intercept = intercept

    # Metric
    metric_data = snapshot.get("metric", {})
    controller.metric._sum = float(metric_data.get("mae", 0.0)) * float(metric_data.get("n", 0.0))
    controller.metric._count = int(metric_data.get("n", 0.0))

    # Experiences: mark all as unlearned (will be re-evaluated)
    old_erf = snapshot.get("erfahrungen", [])
    controller.erfahrungs_liste = []
    for e in old_erf:
        erf = _erfahrung_aus_json(e) if "features" in e else None
        if erf is not None:
            erf.gelernt = False
            erf.learned_target = None
            erf.learned_weight = None
            controller.erfahrungs_liste.append(erf)

    controller.erfahrungen_gelernt_total = int(snapshot.get("erfahrungen_gelernt_total", 0))
    controller.max_age_days = 7  # v2 default

    # History
    controller.aussen_temp_history = HistoryBuffer(
        _temp_aus_json(t) for t in snapshot.get("history", [])
    )

    _LOGGER.info("Migrated v2 (river) snapshot to v3 (numpy)")


def snapshot_to_controller(
    snapshot: dict[str, Any],
    min_vorlauf: float,
    max_vorlauf: float,
    learning_rate: float,
    min_thermische_leistung: float = 0.6,
    max_thermische_leistung: float = 6.0,
    sanity_band: float = DEFAULT_SANITY_BAND,
    cleanup_tage: int = 30,
) -> FlowController:
    schema_version = snapshot.get("schema_version")
    if schema_version not in (2, 3):
        raise ValueError(f"Unknown schema version: {schema_version!r}")

    controller = FlowController(
        min_vorlauf=min_vorlauf,
        max_vorlauf=max_vorlauf,
        learning_rate=learning_rate,
        min_thermische_leistung=min_thermische_leistung,
        max_thermische_leistung=max_thermische_leistung,
        sanity_band=sanity_band,
        cleanup_tage=cleanup_tage,
    )

    if schema_version == 2:
        _migrate_v2_to_v3(snapshot, controller)
        return controller

    # Schema v3
    controller.scaler_mean = np.array(snapshot["scaler_mean"], dtype=float)
    controller.scaler_std = np.array(snapshot["scaler_std"], dtype=float)
    model_data = snapshot["model"]
    controller.weights = np.array(model_data["weights"], dtype=float)
    controller.intercept = float(model_data["intercept"])

    metric_data = snapshot["metric"]
    controller.metric._sum = float(metric_data.get("mae_sum", 0.0))
    controller.metric._count = int(metric_data.get("mae_count", 0))

    controller.erfahrungs_liste = [_erfahrung_aus_json(e) for e in snapshot["erfahrungen"]]
    controller.erfahrungen_gelernt_total = int(snapshot.get("erfahrungen_gelernt_total", 0))
    controller.max_age_days = int(snapshot.get("max_age_days", cleanup_tage))
    controller.aussen_temp_history = HistoryBuffer(
        _temp_aus_json(t) for t in snapshot["history"]
    )

    return controller


def lade_snapshot(pfad: Path) -> dict[str, Any] | None:
    if not pfad.exists():
        return None
    with pfad.open("r", encoding="utf-8") as f:
        return json.load(f)


def speichere_snapshot(snapshot: dict[str, Any], pfad: Path) -> None:
    tmp_pfad = pfad.with_suffix(pfad.suffix + ".tmp")
    with tmp_pfad.open("w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_pfad, pfad)