"""JSON-based snapshot store for the FlowController.

Instead of pickling river objects (unstable across river versions, since river
pickles C extension objects like VectorDict), only the numerical state is
stored as JSON and rebuilt when loading. This way the learning DB survives
river upgrades.
"""

import json
import logging
import os
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from river import compose, linear_model, optim, preprocessing, stats, metrics
from river.utils.vectordict import VectorDict

from .flow_controller import FlowController
from .types import (
    DateTimeTemperatur,
    Erfahrung,
    Features,
    HistoryBuffer,
)

_LOGGER = logging.getLogger(__name__)

MODEL_SCHEMA_VERSION = 2

FEATURE_KEYS = list(Features.__dataclass_fields__)


def _temp_als_json(entry: DateTimeTemperatur) -> dict[str, Any]:
    """Converts DateTimeTemperatur into a JSON dict."""

    return {
        "timestamp": entry.timestamp.isoformat(),
        "temperature": entry.temperature,
    }


def _temp_aus_json(data: dict[str, Any]) -> DateTimeTemperatur:
    """Converts a JSON dict into DateTimeTemperatur."""

    return DateTimeTemperatur(
        timestamp=datetime.fromisoformat(data["timestamp"]),
        temperature=float(data["temperature"]),
    )


def _erfahrung_als_json(erfahrung: Erfahrung) -> dict[str, Any]:
    """Converts Erfahrung into a JSON dict."""

    return {
        "timestamp": erfahrung.timestamp.isoformat(),
        "features": erfahrung.features.to_dict(),
        "vorlauf_soll": erfahrung.vorlauf_soll,
        "raum_ist_vorher": erfahrung.raum_ist_vorher,
        "raum_soll": erfahrung.raum_soll,
        "gelernt": erfahrung.gelernt,
        "gewicht_faktor": erfahrung.gewicht_faktor,
    }


def _erfahrung_aus_json(data: dict[str, Any]) -> Erfahrung:
    """Converts a JSON dict into Erfahrung."""

    return Erfahrung(
        timestamp=datetime.fromisoformat(data["timestamp"]),
        features=Features(**{k: data["features"][k] for k in FEATURE_KEYS}),
        vorlauf_soll=float(data["vorlauf_soll"]),
        raum_ist_vorher=float(data["raum_ist_vorher"]),
        raum_soll=float(data["raum_soll"]),
        gelernt=bool(data["gelernt"]),
        # Snapshots before the factor field do not have it: default 1.0
        gewicht_faktor=float(data.get("gewicht_faktor", 1.0)),
    )


def _metric_aus_snapshot(metric_data: dict[str, Any]) -> "metrics.MAE":
    """Builds the MAE metric from the snapshot."""

    metric = metrics.MAE()
    metric._mean = stats.Mean()
    metric._mean.n = float(metric_data["n"])
    metric._mean._mean = float(metric_data["mae"])
    return metric


def controller_to_snapshot(controller: FlowController) -> dict[str, Any]:
    """Creates a JSON-serializable snapshot of the learned state."""

    scaler = controller.model["StandardScaler"]
    reg = controller.model["LinearRegression"]
    optimizer = reg.optimizer

    return {
        "schema_version": MODEL_SCHEMA_VERSION,
        "scaler": {
            "with_std": scaler.with_std,
            "window_size": scaler.window_size,
            "counts": dict(scaler.counts),
            "means": {k: float(v) for k, v in scaler.means.items()},
            "vars": {k: float(v) for k, v in scaler.vars.items()},
        },
        "model": {
            "weights": dict(reg.weights),
            "intercept": reg.intercept,
            "optimizer": {
                "n_iterations": optimizer.n_iterations,
                "beta_1": optimizer.beta_1,
                "beta_2": optimizer.beta_2,
                "eps": optimizer.eps,
                "m": dict(optimizer.m),
                "v": dict(optimizer.v),
            },
        },
        "metric": {
            "mae": float(controller.metric.get()),
            "n": float(controller.metric._mean.n),
        },
        "erfahrungen": [_erfahrung_als_json(e) for e in controller.erfahrungs_liste],
        "erfahrungen_gelernt_total": int(getattr(controller, "erfahrungen_gelernt_total", 0)),
        "history": [_temp_als_json(t) for t in controller.aussen_temp_history],
    }


def snapshot_to_controller(
    snapshot: dict[str, Any],
    min_vorlauf: float,
    max_vorlauf: float,
    learning_rate: float,
    min_thermische_leistung: float = 0.6,
    max_thermische_leistung: float = 6.0,
    sanity_band: float = 5.0,
) -> FlowController:
    """Rebuilds a FlowController from a snapshot.

    The configuration parameters come from the current HA configuration,
    only the learned state is restored. The optimizer is rebuilt with
    the current learning_rate (this way lr changes take effect).
    """

    schema_version = snapshot.get("schema_version")
    if schema_version != MODEL_SCHEMA_VERSION:
        raise ValueError(f"Unknown schema version: {schema_version!r}")

    scaler_data = snapshot["scaler"]
    model_data = snapshot["model"]
    optimizer_data = model_data["optimizer"]

    scaler = preprocessing.StandardScaler(
        with_std=scaler_data["with_std"],
        window_size=scaler_data["window_size"],
    )
    scaler.counts.update(Counter(scaler_data["counts"]))
    scaler.means.update(scaler_data["means"])
    scaler.vars.update(scaler_data["vars"])

    optimizer = optim.Adam(
        lr=learning_rate,
        beta_1=optimizer_data["beta_1"],
        beta_2=optimizer_data["beta_2"],
        eps=optimizer_data["eps"],
    )
    optimizer.n_iterations = int(optimizer_data["n_iterations"])
    optimizer.m = VectorDict(dict(optimizer_data["m"]), float)
    optimizer.v = VectorDict(dict(optimizer_data["v"]), float)

    reg = linear_model.LinearRegression(optimizer=optimizer, intercept_lr=learning_rate)
    reg._weights = VectorDict(dict(model_data["weights"]), reg.initializer)
    reg.intercept = float(model_data["intercept"])

    controller = FlowController(
        min_vorlauf=min_vorlauf,
        max_vorlauf=max_vorlauf,
        learning_rate=learning_rate,
        min_thermische_leistung=min_thermische_leistung,
        max_thermische_leistung=max_thermische_leistung,
        sanity_band=sanity_band,
    )
    controller.model = compose.Pipeline(scaler, reg)
    controller.metric = _metric_aus_snapshot(snapshot["metric"])
    controller.erfahrungs_liste = [_erfahrung_aus_json(e) for e in snapshot["erfahrungen"]]
    controller.erfahrungen_gelernt_total = int(snapshot.get("erfahrungen_gelernt_total", 0))
    controller.aussen_temp_history = HistoryBuffer(
        _temp_aus_json(t) for t in snapshot["history"]
    )

    return controller


def lade_snapshot(pfad: Path) -> dict[str, Any] | None:
    """Loads a snapshot from disk. Returns None if not present."""

    if not pfad.exists():
        return None

    with pfad.open("r", encoding="utf-8") as f:
        return json.load(f)


def speichere_snapshot(snapshot: dict[str, Any], pfad: Path) -> None:
    """Stores a snapshot atomically as JSON (tmp file + replace)."""

    tmp_pfad = pfad.with_suffix(pfad.suffix + ".tmp")
    with tmp_pfad.open("w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_pfad, pfad)
