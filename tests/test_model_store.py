"""Tests für den JSON-Snapshot-Speicher (model_store)."""

from datetime import datetime, timedelta
import json

import pytest

from custom_components.adaptive_flow_control.flow_controller import FlowController
from custom_components.adaptive_flow_control.model_store import (
    MODEL_SCHEMA_VERSION,
    controller_to_snapshot,
    lade_snapshot,
    snapshot_to_controller,
    speichere_snapshot,
)
from custom_components.adaptive_flow_control.types import (
    DateTimeTemperatur,
    Erfahrung,
    Features,
    HistoryBuffer,
)


def _baue_controller(
    learning_rate: float = 0.01,
    erfahrungen: int = 3,
    history: int = 5,
) -> FlowController:
    """Baut einen trainierten Controller mit Erfahrungen und Historie."""

    controller = FlowController(
        min_vorlauf=26.0,
        max_vorlauf=40.0,
        learning_rate=learning_rate,
    )
    controller.setup(iterations=0)

    basis = datetime(2026, 11, 15, 12, 0, 0)
    for i in range(1, 21):
        features = Features(
            aussen_temp=float(i),
            raum_abweichung=0.5 * i,
            temp_diff=2.0 * i,
            thermische_leistung_relativ=0.1 * i,
            aussen_trend_1h=0.1,
            stunde_sin=0.5,
            stunde_cos=0.5,
            wochentag_sin=0.2,
            wochentag_cos=0.2,
        )
        y = 28.0 + 0.1 * i
        controller.model.learn_one(features.to_dict(), y)

    controller.erfahrungs_liste = [
        Erfahrung(
            timestamp=basis + timedelta(hours=i),
            features=Features(
                aussen_temp=5.0 + i,
                raum_abweichung=-1.0 + i,
                temp_diff=3.0,
                thermische_leistung_relativ=0.4,
                aussen_trend_1h=0.2,
                stunde_sin=0.0,
                stunde_cos=1.0,
                wochentag_sin=0.0,
                wochentag_cos=-1.0,
            ),
            vorlauf_soll=28.5 + 0.1 * i,
            raum_soll=22.0,
            gelernt=i % 2 == 0,
        )
        for i in range(erfahrungen)
    ]
    controller.aussen_temp_history = HistoryBuffer(
        DateTimeTemperatur(
            timestamp=basis + timedelta(minutes=10 * i),
            temperature=9.0 - 0.1 * i,
        )
        for i in range(history)
    )
    return controller


def _test_temp() -> dict[str, float]:
    return Features(
        aussen_temp=12.0,
        raum_abweichung=0.7,
        temp_diff=-11.3,
        thermische_leistung_relativ=0.5,
        aussen_trend_1h=0.05,
        stunde_sin=0.3,
        stunde_cos=0.9,
        wochentag_sin=0.1,
        wochentag_cos=0.8,
    ).to_dict()


def test_roundtrip_bit_exact():
    """Nach Snapshot+Restore sind Prognose und Zustand exakt identisch."""

    original = _baue_controller()
    x = _test_temp()

    ref = original.model.predict_one(x)

    snapshot = controller_to_snapshot(original)
    restored = snapshot_to_controller(
        snapshot,
        min_vorlauf=26.0,
        max_vorlauf=40.0,
        learning_rate=0.01,
    )

    assert restored.model.predict_one(x) == ref

    # Weiterlernen auf beiden: bleibt identisch
    y = 28.0
    original.model.learn_one(x, y)
    restored.model.learn_one(x, y)
    assert restored.model.predict_one(x) == original.model.predict_one(x)


def test_erfahrungen_und_historie_erhalten():
    """Erfahrungen und Außen-Temperatur-Historie überleben den Roundtrip."""

    original = _baue_controller(erfahrungen=4, history=7)
    restored = snapshot_to_controller(
        controller_to_snapshot(original),
        min_vorlauf=26.0,
        max_vorlauf=40.0,
        learning_rate=0.01,
    )

    assert len(restored.erfahrungs_liste) == len(original.erfahrungs_liste)
    assert len(restored.aussen_temp_history) == len(original.aussen_temp_history)
    for vor, nach in zip(original.erfahrungs_liste, restored.erfahrungs_liste):
        assert nach == vor
    for vor, nach in zip(original.aussen_temp_history, restored.aussen_temp_history):
        assert nach == vor
    assert restored.aussen_temp_history.get_trend(2.0) == original.aussen_temp_history.get_trend(2.0)


def test_metric_erhalten():
    """Die MAE-Metrik überlebt den Roundtrip."""

    original = _baue_controller()
    original.metric.update(0.0, 2.0)
    original.metric.update(0.0, 4.0)
    assert original.metric.get() == pytest.approx(3.0)

    restored = snapshot_to_controller(
        controller_to_snapshot(original),
        min_vorlauf=26.0,
        max_vorlauf=40.0,
        learning_rate=0.01,
    )
    assert restored.metric.get() == pytest.approx(3.0)
    restored.metric.update(0.0, 6.0)
    assert restored.metric.get() == pytest.approx(4.0)


def test_learning_rate_wird_aus_konfiguration_uebernommen():
    """Die learning_rate aus der HA-Konfiguration wird beim Rebuild wirksam."""

    original = _baue_controller(learning_rate=0.01)
    x = _test_temp()
    y = 28.0

    mit_lr = snapshot_to_controller(
        controller_to_snapshot(original),
        min_vorlauf=26.0,
        max_vorlauf=40.0,
        learning_rate=0.01,
    )
    mit_andrer_lr = snapshot_to_controller(
        controller_to_snapshot(original),
        min_vorlauf=26.0,
        max_vorlauf=40.0,
        learning_rate=0.05,
    )

    assert mit_andrer_lr.model["LinearRegression"].optimizer.lr.learning_rate == 0.05

    # Gleiche Lernschritte, unterschiedliche lr: Prognosen laufen auseinander
    for controller in (original, mit_lr, mit_andrer_lr):
        controller.model.learn_one(x, y)
    assert mit_andrer_lr.model.predict_one(x) != pytest.approx(original.model.predict_one(x))
    assert mit_lr.model.predict_one(x) == pytest.approx(original.model.predict_one(x))


def test_falsche_schema_version_verworfen():
    """Unbekannte Schema-Version wird mit ValueError verworfen (neu starten)."""

    original = _baue_controller()
    snapshot = controller_to_snapshot(original)
    snapshot["schema_version"] = 999

    with pytest.raises(ValueError):
        snapshot_to_controller(
            snapshot,
            min_vorlauf=26.0,
            max_vorlauf=40.0,
            learning_rate=0.01,
        )


def test_json_platte_roundtrip(tmp_path):
    """Snapshot speichern/laden von der Platte liefert identischen Controller."""

    original = _baue_controller()
    x = _test_temp()
    ref = original.model.predict_one(x)

    pfad = tmp_path / "model.json"
    speichere_snapshot(controller_to_snapshot(original), pfad)

    assert lade_snapshot(pfad) is not None
    assert lade_snapshot(pfad / "nicht_da.json") is None

    with pfad.open("r", encoding="utf-8") as f:
        assert json.load(f)["schema_version"] == MODEL_SCHEMA_VERSION

    restored = snapshot_to_controller(
        lade_snapshot(pfad),
        min_vorlauf=26.0,
        max_vorlauf=40.0,
        learning_rate=0.01,
    )
    assert restored.model.predict_one(x) == ref

    # Kein .tmp-Artefakt bleibt zurück
    assert not (tmp_path / "model.json.tmp").exists()
