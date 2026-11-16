# Heatpump Flow Control

ML-basierte Vorlauf-Sollwert-Regelung für Luftwärmepumpen als Home-Assistant-Custom-Integration.

Die Integration ersetzt (oder ergänzt) die starre Heizkurve durch ein Online-Lernmodell, das den Vorlauf-Sollwert an die reale Heizsituation anpasst. Der berechnete Wert wird als `number`-Entity bereitgestellt und z. B. per Automation, pyscript oder Bridge an die Wärmepumpe weitergereicht — unabhängig vom Reglertyp (CAN, Modbus, MQTT, …).

---

## Funktionsweise

### Überblick: Der Regelkreis

```
Sensoren                     Integration                        Wärmepumpe
─────────                    ───────────                        ──────────
Aussentemp ──┐
Raum IST  ───┤   Feature-Vektor ──► river-Modell ──► Vorhersage
Raum SOLL ───┤                                        │
Vorlauf IST ─┘                                        ▼
                                     Sicherheitsketten (Klemme,
                                     Schrittweite, Sanity-Band)
                                                      │
                                                      ▼
                                     number.heatpump_flow_control_vorlauf_soll
                                                      │
                                                      ▼  (per Automation/pyscript)
                                     VorlaufSoll-Sollwert der WP
                                                      │
                                                      ▼  (4h später)
                                     Raumtemperatur als "Reward" ──► Lernen
```

1. **Vorhersage:** Aus aktuellen Sensordaten wird ein Feature-Vektor gebaut, das river-Modell sagt den Vorlauf-Sollwert voraus.
2. **Lieferung:** Die Vorhersage läuft durch mehrere Sicherheitsketten und landet als Wert auf der Number-Entity, die die Wärmepumpe als Ziel-Vorlauf nutzt.
3. **Lernen (zeitversetzt):** Jede Vorhersage wird als *Erfahrung* gespeichert. Vier Stunden später wird sie bewertet: Wie hat sich der Raumtemperatur-Regelfehler entwickelt? Daraus lernt das Modell eine Korrektur auf die alte Vorhersage. So lernt das Modell von den *Folgen* seiner eigenen Entscheidungen (Delayed Reward) statt nur vom Ist-Zustand.### Entities

| Entity | Typ | Zweck |
|---|---|---|
| `number.heatpump_flow_control_vorlauf_soll` | Number | Der berechnete Vorlauf-Sollwert (manuell überschreibbar) |
| `switch.heatpump_flow_control_aktiv` | Switch | Regelung an/aus — aus = es wird nichts geschrieben, das Modell lernt nicht, alle Werte bleiben persistiert |

### Update-Zyklus

Ein Update läuft standardmäßig stündlich (zusätzliche Trigger: Zustandswechsel der Sensoren). Vor jeder Berechnung prüft die Integration:

1. **Schalter aktiv?** — Nein → skip.
2. **Heizt die WP?** — Das konfigurierte `is_heating`-Binary-Sensor (z. B. pumpenagnostisch aus CAN/Modbus abgeleitet: Betriebsart "Heizen" und keine Warmwasser-/Brauchwasser-Aktivität) muss `on` sein. Bei Warmwasser oder Absenkung → skip, es wird nichts geschrieben und nichts gelernt.
3. **Stabilisierung:** Direkt nach Heizbeginn wird 10 Minuten gewartet, damit sich die Anlage nach einem Start erst einpendelt (verhindert Lernen auf Übergangswerte).

Läuft ein Update, wird ein parallel anlaufendes Update übersprungen (Lock).

**Sofort-Update bei Sollwert-Änderung:** Zusätzlich zum Stundentimer reagiert die Integration auf Änderungen des `raum_soll`-Sensors — entprellt mit 60 Sekunden, damit Slider-Ziehen oder Zeitplan-Änderungen nicht eine Kette von Zyklen auslösen. Nach dem Entprellen läuft ein voller Regelzyklus mit allen Gates und Sicherheitsketten; die Wärmepumpe muss dann nicht bis zu einer Stunde auf die neue Einstellung warten. (`raum_ist`-Änderungen triggern bewusst nicht — die ändern sich laufend und würden das Lernen mit Redundanz fluten.)

### Feature-Vektor

Das Modell sieht pro Zyklus:

| Feature | Bedeutung |
|---|---|
| `aussen_temp` | aktuelle Aussentemperatur |
| `raum_abweichung` | `raum_soll − raum_ist` (negativ = Raum zu warm) |
| `aussen_trend_1h` | Aussentemperatur-Trend der letzten Stunde (°C/h) |
| `thermische_leistung_relativ` | aktuelle thermische Leistung normiert auf [min, max] (0–1) |
| `temp_diff` | Aussentemp − Raum-IST |
| `stunde_sin`, `stunde_cos` | Tageszeit zyklisch kodiert |
| `wochentag_sin`, `wochentag_cos` | Wochentag zyklisch kodiert |

*Hinweis: Die Betriebsart fließt bewusst nicht als Feature ein — der is_heating-Gate filtert Heizbetrieb vorab, und die Betriebsart ist pumpenagnostisch über den Helper gelöst.*

### Modell

- **Pipeline:** `river.compose.Pipeline(StandardScaler → LinearRegression(Adam))` — echtes Online-Lernen: jedes Lern-Ereignis passt die Koeffizienten sofort an, es gibt keine Batch-Phase.
- **Vortraining:** Vor dem echten Betrieb wird das Modell auf synthetischen Daten trainiert, die aus der Heizkurve erzeugt werden (Temperaturbereich −20 bis +20 °C, alle Raumabweichungen −5 bis +5 K). Synthetische Datenpunkte zählen nur mit Gewicht `0.1`, damit echte Betriebsdaten nach wenigen Tagen dominieren.
- **Fallback-Heizkurve:** Aus zwei Stützpunkten interpoliert: `(15 °C → min_vorlauf)` und `(−20 °C → max_vorlauf)`, plus `2 K pro Kelvin` Raumabweichung. Sie dient als Referenz für das Sanity-Band und als Notwert.

### Die Sicherheitsketten (in Ausführungsreihenfolge)

| Stufe | Was | Schutz gegen |
|---|---|---|
| 1. **Sanity-Band** (`sanity_band`, Standard 8 K) | Liegt die Vorhersage mehr als ±8 K neben der Heizkurve, gilt sie als Modellfehler (z. B. Extrapolation auf nie gesehene Feature-Kombinationen) und wird durch die Heizkurve ersetzt (Log: `Model liefert unrealistischen Wert … verwende Fallback-Heizkurve`). | Wildwuchs-Sprünge des Modells |
| 2. **Bereichsgrenzen** | Klemmung auf `[min_vorlauf, max_vorlauf]`. | physikalisch unsinnige Werte |
| 3. **Schrittweitenlimit** (`max_schrittweite`, Standard 1 K) | Max. Änderung des **Modellwerts** pro Zyklus (Log: `Sollwert-Sprung begrenzt …`). Im Normalbetrieb (stündliche Änderungen < 1 K) unsichtbar; es fängt nur Sprünge ab und ist ein permanentes Sicherheitsgeländer, kein temporärer Patch. | falsches Lern-Ereignis lässt die WP stundenlang Vollgas laufen |
| 4. **Totzonen-Klemme** (`hysterese`, Standard 3 K) | Viele WP-Regler (z. B. HPSU) starten erst, wenn `VorlaufSoll ≥ Tv + 3 K`. Bei Heizbedarf (`raum_ist < raum_soll`) wird eine Vorhersage in dieser toten Zone auf mindestens `Tv + 3,5 K` geklemmt (Log: `Heizbedarf, Vorhersage … in toter Zone … geklemmt`). Die Klemme läuft **nach** dem Schrittweitenlimit und wird deshalb **nicht gedrosselt** — die deterministische Startregel des Reglers wirkt sofort, während Modell-Sprünge weiter begrenzt bleiben. | Kurzzyklen / Nichtstart / verzögerter Start nach Sollwert-Änderung |

Das Schrittweitenlimit wird **nie automatisch ausgeschaltet**. Wenn das Modell später so verlässlich lernt, dass schnellere Anpassung gewünscht ist: Option `max_schrittweite` erhöhen (z. B. 2 K) oder auf `0` setzen (= aus).

### Zeitversetztes Lernen (Delayed Reward)

Beim Update wird eine **Erfahrung** gespeichert: Zeitstempel, Feature-Vektor und der **tatsächlich gelieferte** Sollwert (nach Klemme und Schrittweitenlimit — nicht die rohe Vorhersage). Vier Stunden später, beim nächsten Update:

1. Bewertung der Erfahrung: Wie weit liegt der Raum vom Soll weg (`abweichung = raum_soll − raum_ist_jetzt`)?
   - Raum zu kalt → die gelieferte Vorlauf-Temperatur war zu niedrig → Lernziel wird **angehoben**.
   - Raum zu warm → Lernziel wird **abgesenkt**.
2. Der Lernzielwert ist `gelieferter_vorlauf + abweichung × 1.5` (gedämpfte Korrektur, `KORREKTUR_FAKTOR`), geklemmt auf `[min_vorlauf, max_vorlauf]`.
3. Das Gewicht des Lern-Schritts sinkt mit dem Alter der Erfahrung (24-h-Decay, `KORREKTUR_DECAY_HOURS`) und steigt mit der beobachteten thermischen Leistung, maximal 5.0 (`MAX_WEIGHT`). Liegt der Korrekturfaktor unter 0.1, wird die Erfahrung übersprungen statt eine Selbstbestätigung zu lernen.
4. Gelernte und älter als 7 Tage alte Erfahrungen werden aufgeräumt; ein kumulativer Zähler (`erfahrungen_gelernt`-Attribut) bleibt erhalten.

### Persistenz

- Snapshot: `/config/heatpump_flow_control.model.json` (Modell, Erfahrungen, Zähler) — gespeichert am Ende jedes Update-Zyklus, bei Entfernen/Neuladen der Integration **und** beim Herunterfahren von Home Assistant (STOP-Event), damit ungelernte Erfahrungen jeden Restart überleben.
- Beim Start wird der Snapshot automatisch geladen (`Model restored from JSON …` im Log). Existiert er nicht, wird das Modell frisch aufgesetzt und synthetisch vortrainiert.
- Manuelle Eingriffe: Der Sollwert kann jederzeit direkt auf der Number-Entity gesetzt werden (z. B. bei Urlaubsabwesenheit). Der nächste reguläre Zyklus übernimmt die Regelung wieder.

---

## Konfiguration

Die Integration wird über *Einstellungen → Geräte & Dienste → Integration hinzufügen* eingerichtet. Alle Parameter sind später über *Konfigurieren* (Options-Flow) änderbar und werden ohne Neustart übernommen.

### Pflichtangaben (Sensoren)

| Parameter | Beschreibung |
|---|---|
| **Aussentemp-Sensor** | `sensor.*` mit Aussentemperatur in °C |
| **Raum-Ist-Sensor** | `sensor.*` mit aktueller Raumtemperatur (Referenzraum) |
| **Raum-Soll-Sensor** | `sensor.*`, `input_number.*` oder `climate.*`, der den gewünschten Sollwert liefert |
| **Vorlauf-Ist-Sensor** | `sensor.*` mit aktueller Vorlauf-Temperatur (Tv) |
| **Vorlauf-Soll-Entity** | Das Ziel: `number.*`, `input_number.*` oder `climate.*`, in das der berechnete Sollwert geschrieben wird (bei CAN-/Modbus-gekoppelten WPn typischerweise ein `number`, das pyscript/Bridge in den Regler-Sollwert schreibt) |
| **is_heating-Entity** | `binary_sensor.*` — `on` = WP heizt (nicht Warmwasser/DHW). Pumptenagnostisch bauen, z. B. aus Betriebsart + BVV-/DHW-Signal |

### Optionale Parameter

| Parameter | Standard | Bereich | Bedeutung |
|---|---|---|---|
| **Thermischer-Leistungs-Sensor** | – | – | `sensor.*` mit thermischer Leistung (kW). Verbessert die Lerngewichtung; ohne Sensor fließt der Mittelwert 0.5 ein |
| **min_vorlauf** | 25 °C | 25–32 | Untere Grenze des Sollwerts + Heizkurven-Stützpunkt bei 15 °C |
| **max_vorlauf** | 55 °C | 35–55 | Obere Grenze des Sollwerts + Heizkurven-Stützpunkt bei −20 °C |
| **min/max thermische Leistung** | 0.6 / 6.0 kW | – | Skalierungsreferenz für `thermische_leistung_relativ` |
| **update_interval** | 60 min | 1–240 | Zykluszeit der Regelung |
| **learning_rate** | 0.01 | 0.001–0.1 | Adam-Lernrate des Modells. Klein = langsam, stabil; groß = schnell, nervös |
| **hysterese** | 3 K | 0–10 | Schaltregel-Abstand des WP-Reglers für die Totzonen-Klemme (`Tv + hysterese + 0.5 K`) — an die Startschwelle des eigenen Reglers anpassen |
| **max_schrittweite** | 1 K | 0–5 | Max. Sollwert-Änderung pro Zyklus. `0` = aus |
| **sanity_band** | 8 K | 1–20 | Toleranzband um die Heizkurve. Nicht zu eng wählen — ein trainiertes Modell darf legitimerweise von der Heizkurve abweichen (real beobachtet: ~5.5 K bei kalten Aussentemperaturen) |

### Attribute der Number-Entity (Debug-Fenster)

| Attribut | Bedeutung |
|---|---|
| `aussen_temp`, `raum_ist`, `raum_soll`, `vorlauf_ist` | zuletzt verwendete Sensorwerte |
| `raum_abweichung`, `aussen_trend_1h` | aktuelle Features |
| `is_heating` | Zustand des is_heating-Gates beim letzten Update |
| `erfahrungen_gelernt` | kumulativer Zähler aller Lern-Ereignisse |
| `history_size` | Länge des Aussentemperatur-Trend-Buffers (nicht die Lernhistorie!) |
| `model_mae` | laufender Modellfehler |
| `last_update` / `next_update` | letzte/nächste geplante Regelung |

---

## Beispiele

### Beispiel 1: Normaler Winter-Tag (alle Ketten unauffällig)

Zustand: Aussentemp 9.7 °C, Raum 23.0 °C, Raum-Soll 22.5 °C, Tv 30.5 °C, letzter Sollwert 30.5 °C.

1. Heizkurven-Fallback: `25 + (40−25)/35 × (15−9.7) + 2 × (22.5−23.0)` ≈ **26.3 °C** → Sanity-Band [18.3; 34.3].
2. Modell-Vorhersage: **31.9 °C** → innerhalb des Bands → übernommen.
3. Bereichsgrenze [25; 40]: unverändert.
4. Totzonen-Klemme: Raum ist warm genug (23.0 ≥ 22.5) → **kein** Heizbedarf → keine Klemme nötig.
5. Schrittweitenlimit: 31.9 > 30.5 + 1.0 → **begrenzt auf 31.5 °C** (Log: `Sollwert-Sprung begrenzt: 31.9°C → 31.5°C (max 1.0K pro Update)`).
6. Geliefert: **31.5 °C** — und genau dieser Wert landet in der Erfahrung für das spätere Lernen.

### Beispiel 2: Totzonen-Klemme verhindert WP-Zykling und verzögerten Start

Zustand: Raum 21.8 °C, Soll 22.5 °C → Heizbedarf. Tv 30.1 °C. Modell sagt **28.0 °C**, letzter gelieferter Wert 30.5 °C.

- Die WP startet erst bei `Soll ≥ Tv + 3 K` = 33.1 K. Ein Soll von 28.0 °C läge in der Totzone: Die WP startet nie oder zykelt.
- 1. Schrittweitenlimit: 28.0 < 30.5 − 1 → auf **29.5 °C** begrenzt (Modellschutz).
- 2. Totzonen-Klemme: `min_soll = 30.1 + 3.0 + 0.5 = 33.6 °C` → geliefert wird **33.6 °C** (Log: `Heizbedarf, Vorhersage 29.5°C in toter Zone (< Tv+3.5K): geklemmt auf 33.6°C`) — **ungedrosselt**, die WP startet sofort.
- Wäre die Klemme (wie früher) vor dem Schrittweitenlimit gelaufen, hätte sie den Sollwert auf 31.5 °C gestaucht und die WP erst nach 2–3 weiteren Zyklen gestartet.
- Die Erfahrung speichert 33.6 °C (den gelieferten Wert), nicht die 28.0.

### Beispiel 3: Sanity-Band fängt Extrapolation ab

Nachts bei −2 °C Aussentemperatur liefert das Modell **39.2 °C**, obwohl die Heizkurve ~26 °C sagt.

- Band ±8 K → [18; 34]. 39.2 liegt außen → Modellfehler.
- Geliefert wird die **Heizkurve (26 °C)** (Log: `Model liefert unrealistischen Wert 39.2°C (Heizkurve: 26.0°C, Band ±8.0K), verwende Fallback-Heizkurve`).
- Für das Lernen bleibt der Vorfall unschädlich: Auch hier trägt die Erfahrung den gelieferten Wert als Anker.

### Beispiel 4: Zeitversetztes Lernen

Vor 4.1 h wurden geliefert: 28.4 °C bei Raum-Soll 22.5 °C. Jetzt: Raum 21.8 °C (zu kalt).

- Abweichung: `22.5 − 21.8 = +0.7 K` → zu niedrig geheizt.
- Lernziel: `28.4 + 0.7 × 1.5 = 29.5 °C`.
- Das Modell lernt mit den damaligen Features das Ziel 29.5 °C, gewichtet nach Alter (24-h-Decay) und thermischer Leistung (Log: `✓ Gelernt aus Erfahrung von vor 4.1h: Vorlauf 28.4°C → 29.5°C`).
- Der `erfahrungen_gelernt`-Zähler steigt um 1.

### Beispiel 5: Warmwasser bricht den Zyklus ab

12:48 — die WP springt auf Warmwasser (is_heating geht `off`).

- Nächstes Update 13:48: Gate prüft `is_heating` → `off` → **skip**: kein neuer Sollwert, keine Erfahrung, kein Lernen. Der zuletzt gelieferte Wert bleibt stehen, bis wieder Heizbetrieb läuft.
- 10 Minuten nach Rückkehr in den Heizbetrieb wird der erste Update übersprungen (Stabilisierung), danach regelt die Integration normal weiter.

### Beispiel 6: Anbindung an die Wärmepumpe

Die Integration schreibt nur ihre eigene Number-Entity. Die Weitergabe an den WP-Regler ist bewusst ausgelagert und reglerspezifisch — typische Wege: pyscript, Automation mit `number.set_value`, Modbus-/MQTT-Bridge. Beispiel CAN-gekoppelte HPSU per pyscript:

```yaml
# pyscript/heat_pump_set_vorlauf_soll_geregelt.py — vereinfacht
@state_trigger("number.heatpump_flow_control_vorlauf_soll")
def copy_to_regler():
    soll = number.heatpump_flow_control_vorlauf_soll
    number.set_value("number.hpsu_can_vorlauf_soll_geregelt", str(soll))
```

Wichtig: Manuelle Eingriffe immer auf der **ML-Number** machen, nicht auf der Kopie für den Regler — sonst überschreibt der Kopierzyklus die Eingabe innerhalb von Sekunden.

---

## Logging / Fehlersuche

DEBUG-Logging für die Integration aktivieren: *Einstellungen → Geräte & Dienste → Heatpump Flow Control → drei Punkte → Debug-Logging aktivieren* (oder per Entwicklerwerkzeuge). Ablauf eines Zyklus im Log:

```
_async_update_vorlauf_soll() controler.enabled: 1, is_heating: 1
_is_heating_mode: on, 1
_async_get_sensor_values() → Reading sensors: aussen=9.7, raum_ist=23.0, …
_erstelle_features()
Model-Prediction: 31.90°C (Außen: 9.7°C, Raum-Abw: -0.48°C, Vorlauf-Ist: 30.5°C)
Vorlauf-Soll berechnet: 31.5°C …
Erfahrung bewertet / ✓ Gelernt aus Erfahrung von vor …
Saving model to /config/heatpump_flow_control.model.json
```

Warnungen, auf die man achten sollte:

| Log-Zeile | Bedeutung |
|---|---|
| `raum_soll-Änderung: sofortiges Update (entprellt 60s)` | Sollwert wurde geändert — voller Regelzyklus läuft 60 s nach der letzten Änderung |
| `Sollwert-Sprung begrenzt` | Schrittweitenlimit griff — beobachten, ob es häufig bei Normalbetrieb auftritt (dann wäre `max_schrittweite` zu eng oder das Modell unruhig) |
| `Model liefert unrealistischen Wert` | Sanity-Band griff — häufigeres Auftreten deutet auf fehlende/verzerrte Trainingsdaten hin |
| `Heizbedarf, Vorhersage … in toter Zone` | Totzonen-Klemme griff — normal bei kleinen Raumabweichungen; hebt auf die WP-Startschwelle an |
| `Not in heating mode - skipping update` | Gate: kein Heizbetrieb (Warmwasser/Absenkung) — normal |

---

## Architektur-Dateien

| Datei | Inhalt |
|---|---|
| `custom_components/heatpump_flow_control/__init__.py` | Setup, Restore/Snapshot-IO |
| `custom_components/heatpump_flow_control/flow_controller.py` | Regelkern: Features, Modell, Heizkurve, Lernen, Erfahrungen |
| `custom_components/heatpump_flow_control/number.py` | Number/Switch-Entities, Update-Zyklus, Sicherheitsketten, Attribute |
| `custom_components/heatpump_flow_control/model_store.py` | JSON-Persistenz |
| `custom_components/heatpump_flow_control/config_flow.py` | Einrichtungs- und Optionsdialog |
| `custom_components/heatpump_flow_control/const.py` | Konstanten und Defaults |
| `tests/` | Unit-Tests (`pytest tests/`) |
